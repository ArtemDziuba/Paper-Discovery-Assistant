"""Resumable orchestration; the production summarizer and rankers remain untouched."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from pda.finetuning_data import base_id, title_agreement
from pda.fulltext import ExtractedPaper, PdfStore, TextUnit, extract_pdf
from pda.models import Paper
from pda.retrieval import ArxivClient
from pda.summary_config import PROJECT_ROOT, SummarySettings, digest
from pda.summary_eval_data import (
    WEIGHTS,
    empty_evaluation,
    error,
    new_record,
    read_record,
    recalculate,
    save_record,
    text_hash,
    timestamp,
    valid_metric,
    valid_summary,
)
from pda.summary_eval_reports import report_results


class SummaryEvaluation:
    def __init__(
        self,
        instances: list[dict],
        output_dir: Path,
        *,
        summary_settings=None,
        summarizer_factory=None,
        judge_factory=None,
        document_loader=None,
        prepared_path: Path = PROJECT_ROOT / "evaluation" / "annotations" / "prepared.json",
        log=print,
    ):
        self.instances = instances
        self.output_dir = output_dir
        self.settings = summary_settings or SummarySettings.from_env()
        self.summarizer_factory = summarizer_factory
        self.judge_factory = judge_factory
        self.document_loader = document_loader or self._document
        self.prepared_path = prepared_path
        self.log = log
        self._summarizer = None
        self._judge = None
        self._metadata = None

    def _paper(self, record: dict) -> Paper:
        identifier = record["paper"]["arxiv_id"]
        if self._metadata is None:
            self._metadata = {}
            if self.prepared_path.exists():
                snapshot = json.loads(self.prepared_path.read_text(encoding="utf-8"))
                for rows in snapshot.get("datasets", {}).values():
                    for query in rows:
                        for item in query["judgments"]:
                            paper = Paper.model_validate(item["paper"])
                            self._metadata[paper.arxiv_id] = paper
                            self._metadata[base_id(paper.arxiv_id)] = paper
        paper = self._metadata.get(identifier)
        if paper is None:
            papers, _ = ArxivClient().fetch_ids([identifier])
            for candidate in papers:
                if base_id(candidate.arxiv_id) == base_id(identifier):
                    paper = candidate
                    break
        if paper is None:
            raise ValueError(f"arXiv metadata unavailable for selected paper {identifier}.")
        if identifier != base_id(identifier) and identifier != paper.arxiv_id:
            raise ValueError(f"arXiv returned a different version than selected ID {identifier}.")
        if title_agreement(record["paper"]["title"], paper.title) < 0.85:
            raise ValueError(
                f"Selected ID {identifier} does not match its dataset title; no substitution."
            )
        return paper

    def _generate(self, record: dict, force: bool) -> dict:
        if self._summarizer is None:
            if self.summarizer_factory:
                self._summarizer = self.summarizer_factory()
            else:
                from pda.summarization import Summarizer

                self._summarizer = Summarizer(self.settings)
        result = self._summarizer.summarize(
            self._paper(record),
            record["query"],
            regenerate=force,
            progress=lambda message: self.log(f"  {message}"),
        )
        return {"summary": result.text, "metadata": result.model_dump(exclude={"text", "query"})}

    def _document(self, record: dict) -> ExtractedPaper:
        metadata = record["summarization"].get("metadata") or {}
        identifier, pdf_hash = metadata.get("arxiv_id"), metadata.get("pdf_sha256")
        if not identifier or not pdf_hash:
            raise ValueError(
                "Cached summary lacks versioned PDF provenance; source cannot be verified."
            )
        selected_id = record["paper"]["arxiv_id"]
        if base_id(identifier) != base_id(selected_id) or (
            selected_id != base_id(selected_id) and selected_id != identifier
        ):
            raise ValueError("Summary source version does not match the selected paper.")
        source_key = digest(
            {
                "arxiv_id": identifier,
                "pdf_sha256": pdf_hash,
                "parser_version": metadata.get("parser_version"),
            }
        )
        path = self.output_dir / "sources" / f"{source_key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("arxiv_id") != identifier or data.get("pdf_sha256") != pdf_hash:
                raise ValueError("Cached evidence belongs to a different paper or PDF.")
            document = ExtractedPaper(
                units=[TextUnit(**unit) for unit in data["document"]["units"]],
                **{k: v for k, v in data["document"].items() if k != "units"},
            )
            if text_hash(document.text) != data.get("text_sha256"):
                raise ValueError("Cached extracted text failed its integrity check.")
            return document
        pdf, actual_hash = PdfStore(self.settings).get(identifier)
        if actual_hash != pdf_hash:
            raise ValueError("PDF changed since summary generation; refusing unrelated evidence.")
        document = extract_pdf(pdf, self.settings)
        from pda.summary_config import PDF_PARSER_VERSION

        save_record(
            path,
            {
                "arxiv_id": identifier,
                "pdf_sha256": pdf_hash,
                "parser_version": PDF_PARSER_VERSION,
                "summary_parser_version": metadata.get("parser_version"),
                "text_sha256": text_hash(document.text),
                "document": asdict(document),
            },
        )
        return document

    def _evaluate(self, record: dict, path: Path, force: bool) -> None:
        if self._judge is None:
            if self.judge_factory is None:
                raise ValueError("A judge factory is required outside aggregate-only mode.")
            self._judge = self.judge_factory()
        judge = self._judge
        key = digest(
            {
                "judge": judge.fingerprint,
                "summary": text_hash(record["summarization"]["summary"]),
                "source": record["summarization"].get("metadata"),
                "selection": record["selection_fingerprint"],
            }
        )
        evaluation = record["evaluation"]
        if force or evaluation.get("cache_key") != key:
            evaluation = record["evaluation"] = {
                **empty_evaluation(),
                "cache_key": key,
                "judge_model": judge.settings.model,
                "configuration": judge.configuration,
            }
        metrics = evaluation["metrics"]
        if all(valid_metric(metrics.get(name)) for name in WEIGHTS):
            evaluation["status"] = "completed"
            evaluation["evaluated_at"] = evaluation.get("evaluated_at") or timestamp()
            recalculate(record)
            save_record(path, record)
            self.log("  Reusing all five cached judge scores.")
            return
        started = time.monotonic()
        previous_duration = evaluation.get("duration_seconds") or 0.0
        evaluation.update(status="incomplete", weighted_score=None, weighted_score_10=None)
        save_record(path, record)
        try:
            document = self.document_loader(record)
            evidence = judge.evidence(
                document,
                record["query"],
                record["summarization"]["summary"],
                self.output_dir / "evidence" / f"{record['evaluation_id']}-{key}.json",
            )
            evidence_metadata = {k: v for k, v in evidence.items() if k != "text"}
            # A changed extraction invalidates partial scores; completed scores remain frozen
            # with their original source hash until an explicit judge/configuration change.
            old_evidence = evaluation.get("evidence")
            if old_evidence and old_evidence != evidence_metadata:
                metrics.clear()
            evaluation["evidence"] = evidence_metadata
            save_record(path, record)
            for name in WEIGHTS:
                if valid_metric(metrics.get(name)):
                    continue
                metric_started = time.monotonic()
                try:
                    self.log(f"  Judging {name}…")
                    metrics[name] = judge.measure(
                        name,
                        record["query"],
                        record["summarization"]["summary"],
                        evidence,
                    )
                    if not valid_metric(metrics[name]):
                        raise ValueError("Judge returned an invalid metric.")
                except Exception as exc:  # noqa: BLE001 -- persist SDK failures per metric.
                    failure = error(f"judge.{name}", exc)
                    metrics[name] = {
                        "status": "failed",
                        "score": None,
                        "reason": None,
                        "error": failure,
                    }
                    evaluation["errors"].append(failure)
                    self.log(f"  {name} failed: {failure['message']}")
                metrics[name]["duration_seconds"] = time.monotonic() - metric_started
                metrics[name]["evaluated_at"] = timestamp()
                evaluation["duration_seconds"] = previous_duration + time.monotonic() - started
                save_record(path, record)
            successful = sum(valid_metric(metrics.get(name)) for name in WEIGHTS)
            evaluation["status"] = (
                "completed"
                if successful == len(WEIGHTS)
                else "incomplete"
                if successful
                else "failed"
            )
            evaluation["evaluated_at"] = timestamp()
        except Exception as exc:  # noqa: BLE001 -- retain summaries after evidence failures.
            failure = error("judge.evidence", exc)
            evaluation["errors"].append(failure)
            evaluation["status"] = (
                "incomplete" if any(valid_metric(m) for m in metrics.values()) else "failed"
            )
            self.log(f"  Evaluation failed: {failure['message']}")
        finally:
            evaluation["duration_seconds"] = previous_duration + time.monotonic() - started
            recalculate(record)
            save_record(path, record)

    def run(
        self, *, mode="default", force_evaluate=False, force_summarize=False, smoke=False
    ) -> dict:
        if mode not in {"default", "evaluation-only", "aggregate-only", "summarization-only"}:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        if force_summarize and mode not in {"default", "summarization-only"}:
            raise ValueError("--force-summarize requires default or summarization-only mode.")
        if force_evaluate and mode in {"aggregate-only", "summarization-only"}:
            raise ValueError(f"--force-evaluate cannot be used in {mode} mode.")
        records, paths = [], []
        for instance in self.instances:
            path = self.output_dir / f"{instance['evaluation_id']}.json"
            try:
                record = read_record(path, instance)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record = new_record(instance)
                record["evaluation"].update(status="failed", errors=[error("cache", exc)])
                paths.append(None)  # Never overwrite an unrelated/corrupt permanent cache.
            else:
                paths.append(path)
                save_record(path, record)
            records.append(record)
        try:
            for index, (record, path) in enumerate(zip(records, paths, strict=True)):
                if path is None:
                    continue
                if record["dataset_errors"]:
                    record["evaluation"].update(status="failed", errors=record["dataset_errors"])
                    save_record(path, record)
                    continue
                if mode == "aggregate-only":
                    recalculate(record)
                    save_record(path, record)
                    continue
                if smoke and index != 0:
                    continue
                self.log(f"{record['evaluation_id']}: {record['paper']['title']}")
                if force_summarize:
                    # Explicit regeneration retains the previous expensive output in history.
                    if valid_summary(record):
                        save_record(
                            self.output_dir
                            / "history"
                            / (
                                f"{record['evaluation_id']}-{text_hash(record['summarization']['summary'])}.json"
                            ),
                            record,
                        )
                    record["summarization"] = new_record(record)["summarization"]
                    record["evaluation"] = empty_evaluation()
                    save_record(path, record)
                if not valid_summary(record):
                    if mode == "evaluation-only":
                        record["evaluation"].update(
                            status="incomplete",
                            errors=[error("cache", "No cached summary; generation disabled.")],
                        )
                        save_record(path, record)
                        continue
                    started = time.monotonic()
                    try:
                        record["evaluation"] = empty_evaluation()
                        generated = self._generate(record, force_summarize)
                        if (
                            not isinstance(generated["summary"], str)
                            or not generated["summary"].strip()
                        ):
                            raise ValueError("Summarizer returned an empty summary.")
                        record["summarization"] = {
                            "status": "completed",
                            **generated,
                            "duration_seconds": time.monotonic() - started,
                            "generated_at": timestamp(),
                            "errors": [],
                        }
                        save_record(
                            path, record
                        )  # Mandatory durable boundary BEFORE any judge call.
                    except Exception as exc:  # noqa: BLE001 -- isolate individual paper failures.
                        failure = error("summarization", exc)
                        record["summarization"].update(
                            status="failed",
                            errors=[failure],
                            duration_seconds=time.monotonic() - started,
                        )
                        record["evaluation"].update(status="failed", errors=[failure])
                        save_record(path, record)
                        self.log(f"  Summarization failed: {failure['message']}")
                        continue
                else:
                    self.log("  Reusing permanently cached summary.")
                if mode == "summarization-only":
                    continue
                try:
                    self._evaluate(record, path, force_evaluate)
                except Exception as exc:  # noqa: BLE001 -- isolate provider setup failures.
                    failure = error("judge.configuration", exc)
                    record["evaluation"].update(status="failed", errors=[failure])
                    recalculate(record)
                    save_record(path, record)
                    self.log(f"  Judge setup failed: {failure['message']}")
        finally:
            # Also rebuild reports on Ctrl+C, after previously completed summaries are durable.
            report = report_results(records, self.output_dir)
        return report

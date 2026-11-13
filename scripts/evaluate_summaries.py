"""Evaluate the fixed 36 query-paper summaries with DeepEval and native Google Gemini."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pda.summary_config import PROJECT_ROOT
from pda.summary_eval_data import load_instances
from pda.summary_eval_judge import (
    DeepEvalJudge,
    JudgeConfigurationError,
    JudgeSettings,
    load_credentials,
)
from pda.summary_eval_reports import print_report
from pda.summary_evaluation import SummaryEvaluation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--evaluation-only", action="store_true")
    modes.add_argument("--aggregate-only", action="store_true")
    modes.add_argument(
        "--summarization-only",
        action="store_true",
        help="Generate missing summaries locally; never initialize or call the Gemini judge.",
    )
    parser.add_argument("--force-evaluate", action="store_true")
    parser.add_argument("--force-summarize", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Process only human query 1 / paper 1."
    )
    parser.add_argument(
        "--judge-model", help="Default: PDA_SUMMARY_JUDGE_MODEL or gemini-2.5-flash."
    )
    parser.add_argument("--evaluation-dir", type=Path, default=PROJECT_ROOT / "evaluation")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "summary_eval"
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    args = parser.parse_args(argv)
    mode = (
        "summarization-only"
        if args.summarization_only
        else "aggregate-only"
        if args.aggregate_only
        else "evaluation-only"
        if args.evaluation_only
        else "default"
    )
    if args.force_summarize and mode not in {"default", "summarization-only"}:
        parser.error("--force-summarize requires default or summarization-only mode.")
    if args.force_evaluate and mode in {"aggregate-only", "summarization-only"}:
        parser.error(f"--force-evaluate cannot be combined with --{mode}.")
    try:
        instances = load_instances(args.evaluation_dir)
        judge = None
        if mode not in {"aggregate-only", "summarization-only"}:
            load_credentials(args.env_file)  # Fail clearly before any expensive summary generation.
            settings = JudgeSettings.from_env()
            if args.judge_model:
                settings = JudgeSettings.model_validate(
                    {**settings.model_dump(), "model": args.judge_model}
                )
            judge = DeepEvalJudge(settings, env_file=args.env_file)
        runner = SummaryEvaluation(
            instances,
            args.output_dir,
            prepared_path=args.evaluation_dir / "annotations" / "prepared.json",
            judge_factory=lambda: judge,
        )
        report = runner.run(
            mode=mode,
            force_evaluate=args.force_evaluate,
            force_summarize=args.force_summarize,
            smoke=args.smoke,
        )
        print_report(report)
        print(f"\nArtifacts: {args.output_dir}")
        selected = report["records"][:1] if args.smoke else report["records"]
        if args.summarization_only:
            completed = sum(r["summarization"]["status"] == "completed" for r in selected)
            print(f"Summaries completed: {completed}/{len(selected)}. Judging disabled.")
            return 0 if completed == len(selected) else 1
        return (
            0
            if args.aggregate_only
            or all(r["evaluation"]["status"] == "completed" for r in selected)
            else 1
        )
    except (JudgeConfigurationError, ValueError, OSError) as exc:
        # Credential values are never printed; provider errors use the record redaction helper.
        from pda.summary_eval_data import error

        print(f"Configuration error: {error('configuration', exc)['message']}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Saved summaries and metric results will be reused on restart.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

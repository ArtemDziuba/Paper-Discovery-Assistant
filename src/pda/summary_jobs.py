"""Bounded background work shared across Streamlit sessions; workers never call Streamlit."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from pda.models import Paper
from pda.summarization import Summarizer
from pda.summary_config import (
    PIPELINE_VERSION,
    SummaryError,
    SummaryResult,
    SummarySettings,
    digest,
    normalized_query,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SummaryJob:
    status: str = "queued"
    progress: str = "Waiting for the local summarization worker…"
    result: SummaryResult | None = None
    error: str | None = None


class SummaryJobs:
    def __init__(self, settings: SummarySettings, summarizer=None):
        self.settings = settings
        self.summarizer = summarizer or Summarizer(settings)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pda-summary")
        self._lock = threading.Lock()
        self._jobs: dict[str, SummaryJob] = {}

    def key(self, paper: Paper, query: str) -> str:
        return digest(
            {
                "id": paper.arxiv_id,
                "query": normalized_query(query),
                "settings": self.settings.fingerprint(),
                "pipeline": PIPELINE_VERSION,
            }
        )

    def snapshot(self, key: str) -> SummaryJob | None:
        with self._lock:
            job = self._jobs.get(key)
            return replace(job) if job else None

    def submit(self, paper: Paper, query: str, *, regenerate: bool = False) -> str:
        key = self.key(paper, query)
        with self._lock:
            current = self._jobs.get(key)
            if current and current.status in {"queued", "running"}:
                return key
            if sum(j.status in {"queued", "running"} for j in self._jobs.values()) >= 4:
                raise SummaryError(
                    "The summary queue is full. Wait for an active paper to finish, then retry."
                )
            if len(self._jobs) >= 200:
                for old_key, job in list(self._jobs.items()):
                    if job.status not in {"queued", "running"}:
                        del self._jobs[old_key]
                        break
            self._jobs[key] = SummaryJob()
        self._executor.submit(self._run, key, paper, query, regenerate)
        return key

    def _run(self, key: str, paper: Paper, query: str, regenerate: bool) -> None:
        def progress(text):
            with self._lock:
                self._jobs[key].progress = text

        with self._lock:
            self._jobs[key].status = "running"
        try:
            result = self.summarizer.summarize(
                paper, query, regenerate=regenerate, progress=progress
            )
            with self._lock:
                self._jobs[key] = SummaryJob(status="succeeded", progress="Ready", result=result)
        except Exception as exc:
            if not isinstance(exc, SummaryError):
                LOGGER.exception("Unexpected full-paper summarization error")
            message = (
                str(exc)
                if isinstance(exc, SummaryError)
                else "Full-paper summarization failed. Check the application log and retry."
            )
            with self._lock:
                self._jobs[key] = SummaryJob(status="failed", error=message)

    def close(self):
        self._executor.shutdown(wait=True)

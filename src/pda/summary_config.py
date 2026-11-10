"""Configuration and shared records for on-demand full-paper summarization."""

import hashlib
import json
import os
import re
import unicodedata
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator

PIPELINE_VERSION = "full-paper-qwen-v4"
PDF_PARSER_VERSION = version("pymupdf")
# Output tolerance only: the model is still instructed to write 100–150 words.
SUMMARY_SHADOW_MIN_WORDS = 50
SUMMARY_SHADOW_MAX_WORDS = 180
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SummaryError(RuntimeError):
    """A summarization failure suitable for display with a retry action."""


class SummarySettings(BaseModel):
    model: str = "qwen3.5:4b"
    endpoint: str = "http://localhost:11434"
    num_ctx: int = Field(default=8192, ge=4096, le=262144)
    temperature: float = Field(default=0.2, ge=0, le=2)
    num_predict: int = Field(default=768, ge=256, le=4096)
    seed: int = 42
    think: bool = False
    keep_alive: str = "10m"
    timeout_seconds: float = Field(default=180, gt=0, le=1800)
    max_runtime_seconds: float = Field(default=1200, gt=0)
    max_pdf_bytes: int = Field(default=40 * 1024 * 1024, ge=1024)
    max_pages: int = Field(default=150, ge=1, le=1000)
    max_text_chars: int = Field(default=300_000, ge=500)
    max_chunks: int = Field(default=96, ge=1, le=1000)
    max_model_calls: int = Field(default=192, ge=1, le=2000)
    max_reduce_rounds: int = Field(default=6, ge=1, le=12)
    template_reserve: int = Field(default=512, ge=256)
    cache_dir: Path = PROJECT_ROOT / ".cache" / "full-paper"

    @model_validator(mode="after")
    def local_endpoint_and_budget(self):
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "Ollama endpoint must be a local HTTP address, such as http://localhost:11434."
            )
        if self.num_ctx - self.num_predict - self.template_reserve < 2048:
            raise ValueError(
                "Context must leave at least 2,048 tokens for input after generation reserves."
            )
        if not self.model.strip() or "cloud" in self.model.casefold():
            raise ValueError("Select an installed local Ollama model, not a cloud model.")
        return self

    @classmethod
    def from_env(cls):
        return cls.model_validate(
            {
                name: os.environ[f"PDA_SUMMARY_{name.upper()}"]
                for name in cls.model_fields
                if f"PDA_SUMMARY_{name.upper()}" in os.environ
            }
        )

    def fingerprint(self) -> str:
        return digest(
            {
                "pipeline": PIPELINE_VERSION,
                "pdf_parser_version": PDF_PARSER_VERSION,
                **self.model_dump(mode="json"),
            }
        )


class SummaryResult(BaseModel):
    text: str
    arxiv_id: str
    query: str
    model: str
    model_digest: str
    pipeline_version: str = PIPELINE_VERSION
    parser_version: str = PDF_PARSER_VERSION
    pdf_sha256: str
    pages: int
    chunks: int
    model_calls: int
    cache_hit: bool = False
    bibliography_removed: bool = False


def normalized_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).split()).casefold()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:[-’']\w+)*\b", text))


def validate_summary(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise SummaryError("The model returned an empty summary. Please retry.")
    if re.search(r"(?im)^\s*(?:#{1,6}\s|[-*]\s|\d+[.)]\s|summary\s*:)", text) or re.search(
        r"</?think\b|https?://|\[\d+(?:[,–-]\d+)*\]", text, re.IGNORECASE
    ):
        raise SummaryError(
            "The model returned headings, thinking text or citations instead of a plain summary. Please retry."
        )
    text = " ".join(text.split())
    count = word_count(text)
    if not SUMMARY_SHADOW_MIN_WORDS <= count <= SUMMARY_SHADOW_MAX_WORDS:
        raise SummaryError(
            f"The summary has {count} words and is outside the accepted length. Please retry."
        )
    return text

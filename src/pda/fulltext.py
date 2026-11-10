"""Versioned arXiv PDF storage, structural text extraction and lossless input chunking."""

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pymupdf

from pda.summary_config import SummaryError, SummarySettings

ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v[1-9]\d*)?")
BIBLIOGRAPHY = re.compile(
    r"^(?:\d+\.?\s+)?(?:references|bibliography|literature cited)\s*$", re.IGNORECASE
)
APPENDIX = re.compile(
    r"^(?:appendix|appendices|supplement(?:ary)?|supporting information)\b",
    re.IGNORECASE,
)
INLINE_BIBLIOGRAPHY = re.compile(
    r"(\*\*(?:references|bibliography|literature cited)\*\*)(?=\s*(?:\[\d+\]|\d+[.)]))",
    re.IGNORECASE,
)


def checked_id(identifier: str, *, require_version: bool = False) -> str:
    if not ARXIV_ID.fullmatch(identifier) or (
        require_version and not re.search(r"v[1-9]\d*$", identifier)
    ):
        raise SummaryError(
            "A valid arXiv identifier with its paper version is required for a full-paper summary."
        )
    return identifier


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PdfStore:
    def __init__(self, settings: SummarySettings, transport=None):
        self.settings = settings
        self.directory = settings.cache_dir / "pdf"
        self.http = httpx.Client(
            transport=transport, timeout=httpx.Timeout(90, connect=15), follow_redirects=True
        )
        self._lock = threading.Lock()
        self._last_request = 0.0

    def get(self, identifier: str) -> tuple[Path, str]:
        checked_id(identifier, require_version=True)
        name = hashlib.sha256(identifier.encode()).hexdigest()
        path = self.directory / f"{name}.pdf"
        with self._lock:
            if path.is_file() and 0 < path.stat().st_size <= self.settings.max_pdf_bytes:
                raw = path.read_bytes()
                if raw.startswith(b"%PDF-"):
                    return path, hashlib.sha256(raw).hexdigest()
            url = f"https://arxiv.org/pdf/{identifier}"
            for attempt in range(3):
                time.sleep(max(0, 3.1 - (time.monotonic() - self._last_request)))
                self._last_request = time.monotonic()
                try:
                    with self.http.stream(
                        "GET",
                        url,
                        headers={
                            "User-Agent": "PaperDiscoveryAssistant/0.1 (local course prototype)"
                        },
                    ) as response:
                        response.raise_for_status()
                        host = urlsplit(str(response.url)).hostname or ""
                        if host != "arxiv.org" and not host.endswith(".arxiv.org"):
                            raise SummaryError(
                                "The PDF request redirected outside arXiv; download stopped."
                            )
                        size = 0
                        pieces = []
                        for piece in response.iter_bytes():
                            size += len(piece)
                            if size > self.settings.max_pdf_bytes:
                                raise SummaryError(
                                    "The paper's PDF exceeds the configured download-size limit."
                                )
                            pieces.append(piece)
                    raw = b"".join(pieces)
                    if not raw.startswith(b"%PDF-"):
                        raise SummaryError(
                            "arXiv did not return a PDF. Retry or open the paper's source link."
                        )
                    atomic_write(path, raw)
                    sha = hashlib.sha256(raw).hexdigest()
                    atomic_write(
                        path.with_suffix(".json"),
                        json.dumps(
                            {
                                "arxiv_id": identifier,
                                "url": url,
                                "sha256": sha,
                                "retrieved_at": time.time(),
                            }
                        ).encode(),
                    )
                    return path, sha
                except httpx.HTTPError as exc:
                    transient = not isinstance(
                        exc, httpx.HTTPStatusError
                    ) or exc.response.status_code in {429, 500, 502, 503, 504}
                    if transient and attempt < 2:
                        continue
                    raise SummaryError(
                        "The full-paper PDF could not be downloaded from arXiv. Please retry later."
                    ) from exc
        raise SummaryError("The full-paper PDF could not be downloaded.")


@dataclass(frozen=True)
class TextUnit:
    section: str
    page: int
    text: str
    kind: str = "text"


@dataclass(frozen=True)
class ExtractedPaper:
    units: list[TextUnit]
    pages: int
    bibliography_removed: bool
    removed_artifact_lines: int

    @property
    def text(self) -> str:
        return "\n\n".join(f"[Section: {u.section}; page {u.page}]\n{u.text}" for u in self.units)


def extract_pdf(path: Path, settings: SummarySettings) -> ExtractedPaper:
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise SummaryError(
            "The downloaded PDF could not be opened. Full-paper extraction failed."
        ) from exc
    with document:
        if document.needs_pass or not 0 < document.page_count <= settings.max_pages:
            raise SummaryError("The PDF is encrypted, empty, or exceeds the configured page limit.")
        # Keep OCR explicit: structured text extraction must not silently install or
        # invoke an OCR engine, or present a scanned paper as an abstract-only summary.
        for page in document:
            if not page.get_text("text").strip() and (page.get_images() or page.get_drawings()):
                raise SummaryError(
                    f"Page {page.number + 1} contains non-extractable content. OCR is required; no abstract-only fallback is used."
                )
        try:
            pages = document.to_markdown(
                page_chunks=True,
                use_ocr=False,
                header=True,
                footer=True,
                write_images=False,
                embed_images=False,
                force_text=True,
                show_progress=False,
            )
        except Exception as exc:
            raise SummaryError(
                "PyMuPDF structured PDF extraction failed. Check the local layout installation and retry."
            ) from exc
        if not isinstance(pages, list) or len(pages) != document.page_count:
            raise SummaryError(
                "Structured extraction did not return every PDF page; no summary was generated."
            )
        units = []
        section = "Opening material"
        bibliography = False
        removed_bibliography = False
        removed_lines = 0
        for page_number, record in enumerate(pages, start=1):
            metadata = record.get("metadata") if isinstance(record, dict) else None
            if not isinstance(metadata, dict) or metadata.get("page_number") != page_number:
                raise SummaryError("Structured extraction returned invalid page provenance.")
            blocks = (
                part for kind, text in _page_blocks(record) for part in _layout_parts(kind, text)
            )
            for kind, text in blocks:
                if kind in {"page-header", "page-footer"}:
                    removed_lines += len(text.splitlines())
                    continue
                heading_text = re.sub(r"^#{1,6}\s+", "", text).strip().strip("*").strip()
                heading = kind in {"title", "section-header"} or bool(re.match(r"^#{1,6}\s", text))
                if BIBLIOGRAPHY.fullmatch(heading_text) and units:
                    bibliography = True
                    removed_bibliography = True
                    continue
                if bibliography:
                    # Layout headings also identify unnamed post-reference supplements.
                    if not heading or (
                        not APPENDIX.match(heading_text)
                        and re.search(r"\[\d+\]|https?://|\b(?:19|20)\d{2}\b", heading_text)
                    ):
                        continue
                    bibliography = False
                if heading:
                    section = " ".join(heading_text.split())
                units.append(TextUnit(section, page_number, text, kind))
        result = ExtractedPaper(units, len(pages), removed_bibliography, removed_lines)
        if len(result.text) < 800 or len(result.text.split()) < 100:
            raise SummaryError(
                "Insufficient full-paper text was extracted. The PDF may be scanned; no abstract-only summary is generated."
            )
        if len(result.text) > settings.max_text_chars:
            raise SummaryError(
                "The paper exceeds the configured text-processing limit; no text was truncated."
            )
        return result


def _page_blocks(record: dict) -> Iterator[tuple[str, str]]:
    """Read layout spans in order, retaining any unclassified text between them."""
    text = record.get("text")
    boxes = record.get("page_boxes")
    if not isinstance(text, str) or not isinstance(boxes, list) or (text.strip() and not boxes):
        raise SummaryError(
            "PyMuPDF did not return structured layout metadata; no partial summary was generated."
        )
    try:
        ordered = sorted(boxes, key=lambda box: box["index"])
        cursor = 0
        for box in ordered:
            start, stop = box["pos"]
            if (
                not isinstance(start, int)
                or not isinstance(stop, int)
                or not cursor <= start <= stop <= len(text)
            ):
                raise ValueError("Invalid or overlapping layout spans")
            if text[cursor:start].strip():
                yield "text", text[cursor:start].strip()
            content = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text[start:stop]).strip()
            if content:
                yield str(box["class"]), content
            cursor = stop
        if text[cursor:].strip():
            yield "text", text[cursor:].strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(
            "PyMuPDF returned invalid layout spans; no text was silently omitted."
        ) from exc


def _layout_parts(kind: str, text: str) -> Iterator[tuple[str, str]]:
    """Retain structural blocks, including bibliography labels merged into body text."""
    if kind in {"table", "formula", "code", "page-header", "page-footer"}:
        yield kind, text
        return
    # A native box can contain 'Acknowledgements ... **References** [1] ...'
    # on one line. A bold label immediately followed by a citation is a reliable
    # boundary even when the classifier labels the entire box as body text.
    for part in INLINE_BIBLIOGRAPHY.split(text):
        part = part.strip()
        if not part:
            continue
        if BIBLIOGRAPHY.fullmatch(part.strip("*")):
            yield "section-header", part
            continue
        if "```" in part:
            yield "code", part
            continue
        for paragraph in re.split(r"\n\s*\n", part):
            if paragraph.strip():
                yield kind, paragraph.strip()


def _table_fragments(text: str, room: int) -> list[str]:
    """Split Markdown tables only between complete rows, repeating column headers."""
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if (
        len(rows) < 3
        or not re.fullmatch(r"\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?", rows[1])
        or any(not row.startswith("|") or not row.endswith("|") for row in rows)
    ):
        raise SummaryError(
            "A structured table exceeds the context budget and cannot be safely split. Increase context size."
        )
    header = rows[:2]
    result, current = [], []
    for row in rows[2:]:
        if len("\n".join([*header, row]).encode()) > room:
            raise SummaryError(
                "A complete table row exceeds the context budget; no cells were truncated."
            )
        if len("\n".join([*header, *current, row]).encode()) > room:
            result.append("\n".join([*header, *current]))
            current = []
        current.append(row)
    if current:
        result.append("\n".join([*header, *current]))
    return result


def chunk_document(document: ExtractedPaper, max_bytes: int, max_chunks: int) -> list[str]:
    """Pack all paragraphs; split oversized paragraphs at sentences, then words if needed."""
    if max_bytes < 256:
        raise SummaryError("The configured context leaves too little room for paper text.")
    fragments = []
    for unit in document.units:
        prefix = f"[Section: {unit.section}; page {unit.page}]\n"
        room = max_bytes - len(prefix.encode())
        if room < 128:
            raise SummaryError(
                "A section heading exceeds the input budget; increase the context size."
            )
        if len(unit.text.encode()) <= room:
            fragments.append(prefix + unit.text)
            continue
        if unit.kind == "table":
            fragments.extend(prefix + part for part in _table_fragments(unit.text, room))
            continue
        if unit.kind in {"formula", "code"} or unit.text.startswith("```"):
            raise SummaryError(
                "A formula or code block exceeds the context budget; increase context size instead of splitting it."
            )
        sentences = re.split(r"(?<=[.!?])\s+", unit.text)
        for sentence in sentences:
            if len(sentence.encode()) <= room:
                fragments.append(prefix + sentence)
                continue
            words = sentence.split()
            current = []
            for word in words:
                if len(word.encode()) > room:
                    raise SummaryError(
                        "An unsplittable text element exceeds the context budget; no content was truncated."
                    )
                if len(" ".join([*current, word]).encode()) > room:
                    fragments.append(prefix + " ".join(current))
                    current = []
                current.append(word)
            if current:
                fragments.append(prefix + " ".join(current))
    chunks, current = [], []
    for fragment in fragments:
        if len("\n\n".join([*current, fragment]).encode()) > max_bytes:
            chunks.append("\n\n".join(current))
            current = []
        current.append(fragment)
    if current:
        chunks.append("\n\n".join(current))
    if len(chunks) > max_chunks:
        raise SummaryError(
            f"The paper needs {len(chunks)} chunks, exceeding the configured limit of {max_chunks}; no sections were omitted."
        )
    return chunks

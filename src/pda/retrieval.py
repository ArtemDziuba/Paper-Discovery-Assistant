"""arXiv Atom retrieval, validation, caching and serialized rate limiting."""

import hashlib
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import httpx

from pda.models import Paper

LOGGER = logging.getLogger(__name__)
NS = {"a": "http://www.w3.org/2005/Atom"}
MAX_CANDIDATES = 10_000
PAGE_SIZE = 1000


class RetrievalError(RuntimeError):
    """A retrieval failure safe to show in the interface."""


def parse_feed(content: str) -> tuple[list[Paper], int]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RetrievalError("arXiv returned malformed XML.") from exc
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise RetrievalError("arXiv did not return an Atom feed.")
    records: dict[str, Paper] = {}
    skipped = 0
    for entry in root.findall("a:entry", NS):

        def value(key: str, entry=entry) -> str:
            return " ".join((entry.findtext(f"a:{key}", "", NS)).split())

        identifier = value("id")
        if "/api/errors" in identifier:
            raise RetrievalError(f"arXiv rejected the search: {value('summary')}")
        try:
            if not identifier.startswith(("http://arxiv.org/abs/", "https://arxiv.org/abs/")):
                raise ValueError("Unexpected identifier URL")
            arxiv_id = identifier.split("/abs/", 1)[1]
            paper = Paper(
                arxiv_id=arxiv_id,
                title=value("title"),
                abstract=value("summary"),
                authors=[
                    " ".join((a.findtext("a:name", "", NS)).split())
                    for a in entry.findall("a:author", NS)
                    if a.findtext("a:name", "", NS).strip()
                ],
                published=value("published"),
                categories=[
                    c.attrib["term"] for c in entry.findall("a:category", NS) if "term" in c.attrib
                ],
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
            # Collapse versions for candidate ranking, preserving the chosen original ID.
            key = re.sub(r"v\d+$", "", arxiv_id)
            records.setdefault(key, paper)
        except (ValueError, KeyError):
            skipped += 1
    return list(records.values()), skipped


class ArxivClient:
    def __init__(self, cache_dir: Path = Path(".cache/arxiv"), ttl: int = 86400):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self._lock = threading.Lock()
        self._last_request = 0.0

    def search(
        self,
        query: str,
        limit: int = 40,
        *,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        progress: Callable[[float, str], None] | None = None,
    ) -> tuple[list[Paper], dict]:
        if not 1 <= limit <= MAX_CANDIDATES:
            raise ValueError("Candidate limit must be between 1 and 10,000.")
        records: dict[str, Paper] = {}
        pages = []
        scanned = 0
        failure = None
        while scanned < limit:
            if progress:
                progress(scanned / limit, f"Retrieving candidates: {scanned:,} / {limit:,}")
            try:
                papers, metadata = self._fetch_page(
                    query,
                    min(PAGE_SIZE, limit - scanned),
                    start=scanned,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
            except RetrievalError as exc:
                if not pages:
                    raise
                failure = str(exc)
                break
            pages.append(metadata)
            for paper in papers:
                records.setdefault(re.sub(r"v\d+$", "", paper.arxiv_id), paper)
            received = metadata["received_entries"]
            scanned += received
            total = metadata["total_results"]
            if received == 0 or (total is not None and scanned >= total):
                break
            if total is None and received < metadata["requested_entries"]:
                break
        if progress:
            progress(1, f"Retrieved {len(records):,} usable candidates")
        return list(records.values()), {
            "cache_hit": all(p["cache_hit"] for p in pages),
            "retrieved_at": min(p["retrieved_at"] for p in pages),
            "skipped_records": sum(p["skipped_records"] for p in pages),
            "requested_candidates": limit,
            "scanned_entries": scanned,
            "total_results": pages[0]["total_results"],
            "pages": pages,
            "partial_failure": failure,
        }

    def _fetch_page(
        self,
        query: str,
        limit: int,
        *,
        start: int,
        sort_by: str,
        sort_order: str,
    ) -> tuple[list[Paper], dict]:
        if sort_by not in {"relevance", "submittedDate", "lastUpdatedDate"}:
            raise ValueError("Unsupported arXiv sort field.")
        if sort_order not in {"ascending", "descending"}:
            raise ValueError("Unsupported arXiv sort order.")
        cache_key = f"pages-v1|{query}|{start}|{limit}"
        if (sort_by, sort_order) != ("relevance", "descending"):
            cache_key += f"|{sort_by}|{sort_order}"
        key = hashlib.sha256(cache_key.encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        with self._lock:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if time.time() - data["retrieved_at"] < self.ttl:
                        return [Paper.model_validate(p) for p in data["papers"]], {
                            "cache_hit": True,
                            "retrieved_at": data["retrieved_at"],
                            "skipped_records": data["skipped_records"],
                            "received_entries": data["received_entries"],
                            "requested_entries": limit,
                            "start": start,
                            "total_results": data["total_results"],
                        }
                except (ValueError, KeyError, OSError):
                    LOGGER.warning("Ignoring invalid retrieval cache", exc_info=True)
            for attempt in range(3):
                time.sleep(max(0, 3.1 - (time.monotonic() - self._last_request)))
                self._last_request = time.monotonic()
                try:
                    response = httpx.get(
                        "https://export.arxiv.org/api/query",
                        params={
                            "search_query": query,
                            "start": start,
                            "max_results": limit,
                            "sortBy": sort_by,
                            "sortOrder": sort_order,
                        },
                        headers={
                            "User-Agent": "PaperDiscoveryAssistant/0.1 (local course prototype)"
                        },
                        timeout=httpx.Timeout(90, connect=15),
                        follow_redirects=True,
                    )
                    if (response.status_code == 429 or response.status_code >= 500) and attempt < 2:
                        try:
                            delay = float(response.headers.get("Retry-After", "3.1"))
                        except ValueError:
                            delay = 3.1
                        time.sleep(min(30, max(3.1, delay)))
                        continue
                    response.raise_for_status()
                    papers, skipped = parse_feed(response.text)
                    root = ET.fromstring(response.text)
                    received = len(root.findall("a:entry", NS))
                    total_text = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
                    try:
                        total = max(0, int(total_text)) if total_text is not None else None
                    except ValueError:
                        total = None
                    data = {
                        "retrieved_at": time.time(),
                        "skipped_records": skipped,
                        "papers": [p.model_dump(mode="json") for p in papers],
                        "received_entries": received,
                        "total_results": total,
                    }
                    try:
                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(data), encoding="utf-8")
                    except OSError:
                        LOGGER.warning("Could not save retrieval cache", exc_info=True)
                    return papers, {
                        "cache_hit": False,
                        "retrieved_at": data["retrieved_at"],
                        "skipped_records": skipped,
                        "received_entries": received,
                        "requested_entries": limit,
                        "start": start,
                        "total_results": total,
                    }
                except httpx.HTTPError as exc:
                    LOGGER.warning(
                        "arXiv request failed: attempt=%s error=%s status=%s",
                        attempt + 1,
                        type(exc).__name__,
                        exc.response.status_code
                        if isinstance(exc, httpx.HTTPStatusError)
                        else None,
                        exc_info=True,
                    )
                    if attempt == 2 or isinstance(exc, httpx.HTTPStatusError):
                        if isinstance(exc, httpx.HTTPStatusError):
                            detail = f"arXiv returned HTTP {exc.response.status_code}."
                            advice = "Please try again later."
                        elif isinstance(exc, httpx.TimeoutException):
                            detail = "The arXiv request timed out."
                            advice = "Please retry or request fewer candidate papers."
                        else:
                            detail = "Could not connect to the arXiv API."
                            advice = (
                                "Check internet access and firewall/proxy permissions. "
                                "If launched by a coding assistant, run the app from your "
                                "own terminal outside its restricted sandbox."
                            )
                        raise RetrievalError(f"{detail} {advice}") from exc
            raise RetrievalError("arXiv is temporarily unavailable.")

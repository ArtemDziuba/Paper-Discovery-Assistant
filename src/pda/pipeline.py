"""UI-independent orchestration and extractive source evidence."""

import re
import time
from collections.abc import Callable

from pda.models import Match, SearchResult
from pda.query import interpret, normalize_title, terms
from pda.ranking import MODEL, REVISION, cosine_scores
from pda.retrieval import RetrievalError


def evidence(abstract: str, concepts: list[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    return max(sentences, key=lambda sentence: len(set(terms(sentence)) & set(concepts)))


def search(
    query: str,
    client,
    encoder,
    limit: int = 40,
    top_k: int = 10,
    progress: Callable[[float, str], None] | None = None,
) -> SearchResult:
    if not 1 <= top_k <= 100:
        raise ValueError("Result count must be between 1 and 100.")
    started = time.perf_counter()
    interpreted = interpret(query)
    title_papers = []
    title_metadata = None
    if interpreted.title_candidate and hasattr(client, "find_exact_titles"):
        if progress:
            progress(0, "Checking for an exact paper title…")
        try:
            title_papers, title_metadata = client.find_exact_titles(interpreted.title_candidate)
        except RetrievalError as exc:
            title_metadata = {"error": str(exc)}
    retrieval_progress = (
        (lambda fraction, text: progress(0.5 * fraction, text)) if progress else None
    )
    try:
        if progress:
            papers, retrieval_metadata = client.search(
                interpreted.search_query,
                limit,
                progress=retrieval_progress,
            )
        else:
            papers, retrieval_metadata = client.search(interpreted.search_query, limit)
    except RetrievalError as exc:
        if not title_papers:
            raise
        papers = []
        retrieval_metadata = {"partial_failure": str(exc), "cache_hit": False}
    combined = {}
    for paper in [*title_papers, *papers]:
        combined.setdefault(re.sub(r"v\d+$", "", paper.arxiv_id), paper)
    papers = list(combined.values())[:limit]
    if title_metadata is not None:
        retrieval_metadata["cache_hit"] = retrieval_metadata.get(
            "cache_hit", False
        ) and title_metadata.get("cache_hit", False)
    retrieved = time.perf_counter()
    matches = []
    exact_title = normalize_title(interpreted.title_candidate or "")
    title_matches = [normalize_title(paper.title) == exact_title for paper in papers]
    if papers:
        texts = [p.text for p in papers]
        ranking_progress = (
            (lambda fraction, text: progress(0.5 + 0.45 * fraction, text)) if progress else None
        )
        if hasattr(encoder, "score_documents"):
            scores = encoder.score_documents(query, texts, progress=ranking_progress)
        else:
            query_vector, document_vectors = encoder.encode_pair(query, texts)
            scores = cosine_scores(query_vector, document_vectors)
        order = sorted(
            range(len(papers)),
            key=lambda i: (
                -(bool(exact_title) and title_matches[i]),
                -float(scores[i]),
            ),
        )
        for index in order[:top_k]:
            paper = papers[index]
            score = float(scores[index])
            matches.append(
                Match(
                    paper=paper,
                    raw_similarity=score,
                    display_score=round(100 * max(0, score), 1),
                    evidence=evidence(paper.abstract, interpreted.concepts),
                    matched_terms=sorted(set(terms(paper.text)) & set(interpreted.concepts)),
                )
            )
    semantic = hasattr(encoder, "model")
    if progress:
        progress(1, "Search complete")
    return SearchResult(
        interpretation=interpreted,
        matches=matches,
        metadata={
            **retrieval_metadata,
            "candidate_count": len(papers),
            "encoder": getattr(encoder, "model_name", MODEL) if semantic else "tf-idf-baseline",
            "model_revision": getattr(encoder, "model_revision", REVISION) if semantic else None,
            "training_run_id": (getattr(encoder, "training_run", None) or {}).get("run_id"),
            "inference_device": getattr(encoder, "device", "cpu"),
            "exact_title_priority": any(title_matches),
            "title_lookup": title_metadata,
            "retrieval_seconds": retrieved - started,
            "ranking_seconds": time.perf_counter() - retrieved,
            "total_seconds": time.perf_counter() - started,
            "score_transform": "100 * max(0, cosine); not a probability",
        },
    )

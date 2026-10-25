"""UI-independent orchestration and extractive source evidence."""

import re
import time
from collections.abc import Callable

from pda.models import Match, SearchResult
from pda.query import interpret, terms
from pda.ranking import MODEL, REVISION, cosine_scores


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
    retrieval_progress = (
        (lambda fraction, text: progress(0.5 * fraction, text)) if progress else None
    )
    if progress:
        papers, retrieval_metadata = client.search(
            interpreted.search_query,
            limit,
            progress=retrieval_progress,
        )
    else:
        papers, retrieval_metadata = client.search(interpreted.search_query, limit)
    retrieved = time.perf_counter()
    matches = []
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
        exact_title = (interpreted.quoted_title or "").casefold()
        order = sorted(
            range(len(papers)),
            key=lambda i: (
                -(bool(exact_title) and papers[i].title.casefold() == exact_title),
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
    overview = [
        {
            "arxiv_id": m.paper.arxiv_id,
            "title": m.paper.title,
            "excerpt": m.evidence,
            "url": str(m.paper.url),
        }
        for m in matches[:5]
    ]
    semantic = hasattr(encoder, "model")
    if progress:
        progress(1, "Search complete")
    return SearchResult(
        interpretation=interpreted,
        matches=matches,
        overview=overview,
        metadata={
            **retrieval_metadata,
            "candidate_count": len(papers),
            "encoder": MODEL if semantic else "tf-idf-baseline",
            "model_revision": REVISION if semantic else None,
            "inference_device": getattr(encoder, "device", "cpu"),
            "exact_title_priority": bool(interpreted.quoted_title),
            "retrieval_seconds": retrieved - started,
            "ranking_seconds": time.perf_counter() - retrieved,
            "total_seconds": time.perf_counter() - started,
            "score_transform": "100 * max(0, cosine); not a probability",
        },
    )

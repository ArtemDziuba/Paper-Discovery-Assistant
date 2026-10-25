"""Local embeddings and a dependency-light TF-IDF baseline."""

from collections.abc import Callable

import numpy as np

from pda.query import tokenize

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def cosine_scores(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    documents = np.asarray(documents, dtype=float)
    if query.ndim != 1 or documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("Incompatible embedding shapes.")
    if not np.isfinite(query).all() or not np.isfinite(documents).all():
        raise ValueError("Embeddings must be finite.")
    denominator = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
    return np.clip(
        np.divide(
            documents @ query, denominator, out=np.zeros(len(documents)), where=denominator > 0
        ),
        -1,
        1,
    )


class LexicalEncoder:
    """Corpus-local TF-IDF vectors; fit documents only, never relevance labels."""

    def encode_pair(self, query: str, documents: list[str]) -> tuple[np.ndarray, np.ndarray]:
        queries, docs = self._vectors([query], documents)
        return queries.toarray()[0], docs.toarray()

    def _vectors(self, queries: list[str], documents: list[str]):
        from scipy.sparse import csr_matrix
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not any(tokenize(d) for d in documents):
            return csr_matrix((len(queries), 0)), csr_matrix((len(documents), 0))
        vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        docs = vectorizer.fit_transform(documents)
        return vectorizer.transform(queries), docs

    def score_queries(self, queries: list[str], documents: list[str]) -> np.ndarray:
        queries_matrix, docs = self._vectors(queries, documents)
        return (queries_matrix @ docs.T).toarray()

    def score_documents(
        self,
        query: str,
        documents: list[str],
        progress: Callable[[float, str], None] | None = None,
    ) -> np.ndarray:
        if progress:
            progress(0, f"Ranking {len(documents):,} candidates with TF-IDF")
        scores = self.score_queries([query], documents)[0]
        if progress:
            progress(1, "TF-IDF ranking complete")
        return scores


class SemanticEncoder:
    def __init__(self, device: str = "cpu"):
        import torch

        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this Python environment. Select cpu, or install "
                "the GPU environment with `uv sync --extra cuda`, then launch with "
                "`uv run --extra cuda streamlit run app.py`."
            )
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(
                MODEL,
                revision=REVISION,
                device=device,
                local_files_only=True,
            )
        except OSError:
            self.model = SentenceTransformer(MODEL, revision=REVISION, device=device)

    def encode_pair(self, query: str, documents: list[str]) -> tuple[np.ndarray, np.ndarray]:
        vectors = self.model.encode(
            [query, *documents], batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors[0], vectors[1:]

    def score_queries(self, queries: list[str], documents: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            [*queries, *documents],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.clip(vectors[: len(queries)] @ vectors[len(queries) :].T, -1, 1)

    def score_documents(
        self,
        query: str,
        documents: list[str],
        progress: Callable[[float, str], None] | None = None,
    ) -> np.ndarray:
        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.empty(len(documents), dtype=float)
        for start in range(0, len(documents), 256):
            stop = min(start + 256, len(documents))
            if progress:
                progress(
                    start / len(documents), f"Encoding candidates: {start:,} / {len(documents):,}"
                )
            vectors = self.model.encode(
                documents[start:stop],
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            scores[start:stop] = cosine_scores(query_vector, vectors)
        if progress:
            progress(1, f"Encoded {len(documents):,} candidates")
        return scores

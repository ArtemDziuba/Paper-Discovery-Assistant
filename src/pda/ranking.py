"""Local embeddings and a dependency-light TF-IDF baseline."""

import json
import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from pda.query import tokenize

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MINILM_PATH = Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"
FINETUNED_PATH = Path(__file__).resolve().parents[2] / "models" / "finetuned-minilm"
FINETUNED_LABEL = "syntetic finetune v1"
FINETUNED_MODEL = "pda/syntetic-finetune-v1"
FINETUNED_V2_PATH = MINILM_PATH.parent / "syntetic-finetune-v2"
FINETUNED_V2_LABEL = "syntetic finetune v2"
FINETUNED_V2_MODEL = "pda/syntetic-finetune-v2"
FINETUNED_OPTIONS = {FINETUNED_LABEL: FINETUNED_PATH, FINETUNED_V2_LABEL: FINETUNED_V2_PATH}
LOGGER = logging.getLogger(__name__)
_MINILM_LOCK = threading.Lock()


def _validate_minilm_source() -> None:
    try:
        source = json.loads((MINILM_PATH / "pda_source.json").read_text(encoding="utf-8"))
        if source != {"model": MODEL, "revision": REVISION}:
            raise ValueError("The model or revision does not match the configured MiniLM.")
    except PermissionError as exc:
        LOGGER.exception("Cannot read local MiniLM source metadata")
        raise RuntimeError(
            f"Access denied reading the local MiniLM folder at {MINILM_PATH}. "
            "Check Windows folder permissions for the account running the app."
        ) from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"The local MiniLM folder at {MINILM_PATH} is incomplete or has a different "
            "revision. Restore the complete folder, or rename it and retry to set up MiniLM."
        ) from exc


def load_minilm(device: str) -> "SentenceTransformer":
    """Install the pinned base model once, then load directly from project-local files."""
    from sentence_transformers import SentenceTransformer

    with _MINILM_LOCK:
        if MINILM_PATH.exists():
            _validate_minilm_source()
            LOGGER.info("Loading MiniLM from %s", MINILM_PATH)
            # Never turn a local loading failure into an implicit network request.
            return SentenceTransformer(str(MINILM_PATH), device=device, local_files_only=True)

        MINILM_PATH.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = MINILM_PATH.parent.resolve()
        staging = staging_parent / f".minilm-{uuid4().hex}"
        # On Windows, tempfile's mode 0o700 creates an owner-only ACL. Those permissions
        # survive publishing by rename and can lock out the account running Streamlit.
        # A normal directory inherits the project's permissions instead.
        staging.mkdir()
        try:
            try:
                model = SentenceTransformer(
                    MODEL, revision=REVISION, device=device, local_files_only=True
                )
            except OSError:
                LOGGER.info("Downloading %s at revision %s", MODEL, REVISION)
                model = SentenceTransformer(MODEL, revision=REVISION, device=device)
            target = staging / "model"
            model.save_pretrained(str(target), create_model_card=False)
            (target / "pda_source.json").write_text(
                json.dumps({"model": MODEL, "revision": REVISION}, indent=2), encoding="utf-8"
            )
            try:
                target.rename(MINILM_PATH)
            except FileExistsError:
                # Another process completed the same pinned installation first.
                _validate_minilm_source()
            LOGGER.info("MiniLM is ready in %s", MINILM_PATH)
            return model
        finally:
            if staging.resolve().parent != staging_parent:
                raise RuntimeError("Unexpected MiniLM staging path; cleanup stopped.")
            shutil.rmtree(staging)


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
    def __init__(self, device: str = "cpu", *, model_path: Path | None = None):
        import torch

        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this Python environment. Select cpu, or install "
                "the GPU environment with `uv sync --extra cuda`, then launch with "
                "`uv run --extra cuda streamlit run app.py`."
            )
        from sentence_transformers import SentenceTransformer

        self.model_name = MODEL
        self.model_revision = REVISION
        self.training_run = None
        if model_path is not None:
            manifest = model_path / "training_manifest.json"
            if not manifest.is_file():
                label = next(
                    (label for label, path in FINETUNED_OPTIONS.items() if path == model_path),
                    FINETUNED_LABEL,
                )
                raise RuntimeError(
                    f"The {label} model is not available. "
                    "Restore its saved local artifact, or select Semantic."
                )
            self.training_run = json.loads(manifest.read_text(encoding="utf-8"))
            self.model_name = self.training_run.get("model_name", FINETUNED_MODEL)
            self.model_revision = self.training_run["run_id"]
            self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
            return
        self.model = load_minilm(device)

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

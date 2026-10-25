"""Chunking and sparse scores must preserve the ranking calculation."""

import numpy as np

from pda.ranking import LexicalEncoder, SemanticEncoder, cosine_scores


def test_semantic_chunked_scores_and_progress():
    calls = []

    class Model:
        def encode(self, texts, **kwargs):
            calls.append((len(texts), kwargs))
            return np.array([[1.0, 2.0] for _ in texts])

    encoder = SemanticEncoder.__new__(SemanticEncoder)
    encoder.model = Model()
    progress = []
    scores = encoder.score_documents(
        "query", ["paper"] * 10000, progress=lambda fraction, _: progress.append(fraction)
    )
    np.testing.assert_allclose(scores, np.ones(10000))
    assert calls[0][0] == 1
    assert sum(size for size, _ in calls[1:]) == 10000
    assert all(size <= 256 and kwargs["batch_size"] == 16 for size, kwargs in calls[1:])
    assert progress == sorted(progress) and progress[-1] == 1


def test_sparse_document_scores_match_dense_reference():
    encoder = LexicalEncoder()
    documents = ["graph graph networks", "stellar physics", "networks in biology"]
    q, docs = encoder.encode_pair("graph learning", documents)
    np.testing.assert_allclose(
        encoder.score_documents("graph learning", documents), cosine_scores(q, docs)
    )


def test_large_sparse_vocabulary(monkeypatch):
    from scipy.sparse import csr_matrix

    original = csr_matrix.toarray

    def toarray(matrix, *args, **kwargs):
        assert matrix.shape[0] == 1, "Do not materialize the document-vocabulary matrix"
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", toarray)
    scores = LexicalEncoder().score_documents(
        "word123 graph", [f"word{i} graph" for i in range(10000)]
    )
    assert scores.shape == (10000,)
    assert np.argmax(scores) == 123

"""Verify project-local MiniLM setup, offline reload, and unchanged embedding outputs."""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

from pda.ranking import MINILM_PATH, MODEL, REVISION, SemanticEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("evaluation/model-storage.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Report already exists; choose a new --output path.")
    existed = MINILM_PATH.exists()
    started = time.perf_counter()
    first = SemanticEncoder(args.device)
    setup_seconds = time.perf_counter() - started
    texts = [
        "graph neural networks",
        "convolutional neural networks for image classification",
        "transformer architecture",
    ]
    initial = first.model.encode(texts, batch_size=16, normalize_embeddings=True)
    reference = SentenceTransformer(
        MODEL, revision=REVISION, device=args.device, local_files_only=True
    )
    reference_vectors = reference.encode(texts, batch_size=16, normalize_embeddings=True)
    del reference, first

    def no_network(*args, **kwargs):
        raise AssertionError("Project-local model loading attempted a network connection.")

    with (
        patch("socket.create_connection", no_network),
        patch("socket.socket.connect", no_network),
        patch("socket.socket.connect_ex", no_network),
    ):
        started = time.perf_counter()
        local = SemanticEncoder(args.device)
        local_load_seconds = time.perf_counter() - started
        reloaded = local.model.encode(texts, batch_size=16, normalize_embeddings=True)
    np.testing.assert_allclose(reloaded, reference_vectors, atol=1e-6, rtol=0)
    np.testing.assert_allclose(initial, reloaded, atol=1e-6, rtol=0)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "Model storage/software correctness check; not an NLP-quality evaluation.",
        "model": MODEL,
        "revision": REVISION,
        "device": args.device,
        "local_path": str(MINILM_PATH),
        "local_folder_existed_before": existed,
        "setup_or_first_load_seconds": setup_seconds,
        "subsequent_local_load_seconds": local_load_seconds,
        "timing_scope": "One process after Python/library imports; not a startup benchmark.",
        "offline_reload_and_inference_ok": True,
        "sample_texts": texts,
        "embedding_shape": list(reloaded.shape),
        "max_abs_difference_from_pinned_reference": float(
            np.max(np.abs(reloaded - reference_vectors))
        ),
        "max_abs_difference_first_vs_local_load": float(np.max(np.abs(initial - reloaded))),
        "local_weights_sha256": hashlib.sha256(
            (MINILM_PATH / "model.safetensors").read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

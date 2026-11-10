# Full ranking evaluation after cleanup

Executed: 2026-11-10T18:47:33.427893+00:00

Software tests: **248 passed in 33.03 seconds**. Evaluation device: CUDA (NVIDIA RTX 4060 Laptop GPU).

Frozen snapshot SHA-256: `bff6e4d2bb4239da6014fcdb21287c05646f9593d6743fe003460ef9c068c3ee`.

TF-IDF, Semantic (original MiniLM), syntetic finetune v1 and syntetic finetune v2 use identical fixed candidate pools. All values below are query macro averages, rounded to four decimal places.

## Human: 2 queries, 64 verified judgments

| Metric | TF-IDF | Semantic | syntetic finetune v1 | syntetic finetune v2 |
|---|---:|---:|---:|---:|
| ndcg@5 | 0.7065 | 0.7997 | 0.7128 | 0.8097 |
| ndcg@10 | 0.7558 | 0.8168 | 0.7764 | 0.8138 |
| ndcg@20 | 0.8093 | 0.8183 | 0.7716 | 0.8236 |
| ndcg@50 | 0.9056 | 0.9311 | 0.8889 | 0.9355 |
| precision@5 | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| precision@10 | 0.9000 | 0.9500 | 0.9500 | 0.9000 |
| precision@20 | 0.8000 | 0.8000 | 0.8000 | 0.8250 |
| map@10 | 0.8521 | 0.8952 | 0.8786 | 0.8627 |
| map@20 | 0.7257 | 0.7393 | 0.7272 | 0.7677 |
| map@50 | 0.8733 | 0.8842 | 0.8749 | 0.8984 |
| pairwise_accuracy@50 | 0.6587 | 0.6863 | 0.6644 | 0.7040 |
| mrr@50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Synthetic: 10 queries, 484 verified judgments

| Metric | TF-IDF | Semantic | syntetic finetune v1 | syntetic finetune v2 |
|---|---:|---:|---:|---:|
| ndcg@5 | 0.7898 | 0.8616 | 0.8972 | 0.9076 |
| ndcg@10 | 0.7953 | 0.8349 | 0.8645 | 0.8685 |
| ndcg@20 | 0.7697 | 0.8227 | 0.8532 | 0.8635 |
| ndcg@50 | 0.9168 | 0.9418 | 0.9518 | 0.9543 |
| precision@5 | 0.8600 | 0.9600 | 0.9800 | 1.0000 |
| precision@10 | 0.9000 | 0.9600 | 0.9600 | 0.9500 |
| precision@20 | 0.8650 | 0.9450 | 0.9400 | 0.9550 |
| map@10 | 0.8302 | 0.9427 | 0.9433 | 0.9390 |
| map@20 | 0.7833 | 0.9165 | 0.9158 | 0.9300 |
| map@50 | 0.8675 | 0.9456 | 0.9349 | 0.9377 |
| pairwise_accuracy@50 | 0.6905 | 0.7824 | 0.7869 | 0.8023 |
| mrr@50 | 0.9500 | 1.0000 | 1.0000 | 1.0000 |

## Regression comparison

Compared with ranking-comparison-full-v2.json: all 96 aggregate metrics and 576 per-query metrics match exactly. All 2192 ranking positions match; maximum absolute cosine difference is 0.0. Dataset/model identities and hashes are unchanged.

## Scope

Human and synthetic source judgments remain separate. One human and 16 synthetic judgments were excluded by the existing frozen title-agreement audit. Precision/MAP/MRR use grade >=2. V2 used these 12 evaluation queries for checkpoint selection, so its values are development performance. This is ranking evaluation, not live arXiv recall or Qwen summary-quality evaluation. No training occurred.

Full precision, per-query scores, complete rankings, provenance and timings: [ranking-comparison-full-2026-11-10-post-cleanup.json](ranking-comparison-full-2026-11-10-post-cleanup.json).

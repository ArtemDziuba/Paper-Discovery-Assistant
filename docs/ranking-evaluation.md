# Ranking evaluation on the supplied annotation sets

The executed comparison is stored in evaluation/ranking-comparison.json, including model
identities, source hashes, per-query rankings, raw scores, exclusions and macro averages.
The existing model weights were evaluated without training or updating them.

| Human set: 2 queries, 64 verified judgments | nDCG@10 | Precision@5 | MRR |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.7558 | 0.9000 | 1.0000 |
| Semantic | 0.8168 | 0.9000 | 1.0000 |
| Fine-tuned | 0.7764 | 0.9000 | 1.0000 |

| Synthetic set: 15 queries, 750 judgments | nDCG@10 | Precision@5 | MRR |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.8038 | 0.9200 | 1.0000 |
| Semantic | 0.8188 | 0.9467 | 1.0000 |

Each query is evaluated against its own explicitly judged papers using title plus abstract.
Synthetic batches use all 50 papers, including those removed from fine-tuning for overlap.
The human source originally has 65 judgments; the previously reported title mismatch for
2605.03953 is excluded pending review. Source labels are unchanged. No unknown cross-query
paper is assigned an artificial zero judgment.

nDCG@10 uses grades 0–3, gains of 2^grade - 1, and log2(rank + 1) position discounts.
Precision@5 and MRR treat grades >=2 as relevant. Averages give each query equal weight.
This measures ranker quality within these small judged pools, excluding live retrieval and
exact-title priority. It does not measure arXiv-wide recall or validate displayed scores as
probabilities. The synthetic comparison measures agreement with synthetic judgments; the
two human queries provide limited evidence about general human relevance.

Run from the project root:

```powershell
uv run --extra cuda python scripts/evaluate_annotations.py --device cuda
```

For CPU evaluation, keep the installed CUDA dependency selection and use `--device cpu`.
The script uses the saved verified abstracts in evaluation/finetuning/prepared.json and
checks current annotation files against its source hashes. If labels change, refresh
metadata preparation first; doing that can change metadata versions and dataset hashes, so
keep the existing prepared.json and report when reproducing this comparison.

To launch the application yourself:

```powershell
uv sync --extra cuda
uv run --extra cuda streamlit run app.py
```

Choose the device in the sidebar and open the localhost URL printed in the terminal.
Use Ctrl+C in that terminal to stop the application. The first model download and uncached
arXiv searches require internet access. An occupied port can be avoided with
`--server.port 8502` on the Streamlit command.

To run software tests:

```powershell
uv run --extra cuda pytest -q
```

The executed suite passed all 55 tests. Unit tests mock external retrieval; the software
tests and the ranking-quality evaluation are separate checks.

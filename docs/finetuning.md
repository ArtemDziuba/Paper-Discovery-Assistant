# Local fine-tuning experiment

The ranking selector offers Semantic, TF-IDF baseline, and Fine-tuned. Fine-tuned loads
models/finetuned-minilm locally and shares the same inference batches, cosine calculation,
score transformation, and exact-title rule as the original semantic encoder. It does not
silently fall back to the base model if its weights or training manifest are absent.

## Data and separation

The supplied files contain 15 synthetic queries with 750 graded judgments and two human
queries with 65 judgments. Both single-query objects and arrays of query objects are accepted;
identical duplicate query groups are deduplicated and conflicting copies are rejected.
Source hashes identify the exact files. Synthetic label generation provenance was not supplied.
These synthetic judgments are not human ground truth.

Metadata was fetched from arXiv by explicit ID in batches of at most 50, with the existing
rate limit, retries, and cache. Actual returned titles, abstracts, versions, source URLs,
and retrieval timestamps are frozen in evaluation/finetuning/prepared.json. Human labels
and supplied titles remain unchanged in their original files.

One human judgment was excluded for review: arXiv:2605.03953 was supplied as `Transformers
with Selective Access to Early Representations`, while the retrieved title is `SATFormer:
Learning When to Use Early Value Residuals in Transformers`. This may reflect a title change;
it is not automatically evidence of a fabricated paper. The comparison uses case/Unicode/
punctuation normalization and a SequenceMatcher title-agreement threshold of 0.85. Review
exclusions and false accepts before interpreting this automatic metadata audit as definitive.

Synthetic query groups are shuffled with seed 42, with three groups reserved for development.
Every paper appearing in development or human evaluation is removed from synthetic training,
including alternate arXiv version suffixes. Ninety synthetic training judgments were removed
for overlap. The resulting split is:

| Partition | Queries | Verified judgments | Role |
| --- | ---: | ---: | --- |
| Synthetic training | 12 | 510 | Gradient updates |
| Synthetic development | 3 | 150 | Checkpoint selection |
| Human test | 2 | 64 | Final comparison; no updates or checkpoint selection |

The human queries had already been used during application development. They are held out
from this fine-tuning run, not an untouched final test of all system design choices.

## Training and artifact

Base: sentence-transformers/all-MiniLM-L6-v2 at revision
c9745ed1d9f207416be6d2e6f8de32d1f16199bf. A seeded sample of 64 ordered document preferences
per training query yields 768 training examples per epoch. Preferences compare two judged
papers for the same query; no unjudged document is converted into a negative.

Loss = mean(max(0, 0.1 * (better_grade - worse_grade) - cosine(query, better)
+ cosine(query, worse))). All encoder weights are updated. This optimizes relative ranking;
it does not turn a grade or cosine into a relevance probability.

Configuration: three epochs, batch size 8, AdamW learning rate 2e-5, weight decay 0.01,
10% warmup followed by linear decay, gradient clipping at 1.0, seed 42, float32 on CUDA.
The original model input length remains unchanged. Epoch 1 had the best synthetic development
nDCG@10 and was selected; later epochs were worse despite lower training loss.

Run ec2dc397f1446876 completed training in 29.06 seconds, excluding initialization and final
baseline comparisons. Recorded training process peak working set was about 1.60 GiB;
PyTorch peak allocated CUDA memory was about 0.93 GiB and reserved memory 1.07 GiB. These
exclude other processes and total device memory overhead. PyTorch was 2.9.1+cu128; saved
model configuration also records Sentence Transformers 5.7.0 and Transformers 5.20.2.

Local weights are saved in models/finetuned-minilm, about 91 MB for model.safetensors.
models/ is ignored by Git. The saved training_manifest.json records the base revision,
configuration, input hashes, split query IDs, selection epoch, resources, and run identity.

## Executed comparison

Evaluation ranks only each query's explicitly judged candidate set, without the title rule
or live arXiv retrieval. nDCG@10 uses grades 0–3. Precision@5 and MRR use grade >=2 as relevant.
Every compared method receives the same verified candidates. This differs from the earlier
SciFact evaluation protocol; those older reports are unchanged and should not be compared
as if they measured the same task.

| Method | Human nDCG@10 | Human Precision@5 | Human MRR |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.7558 | 0.90 | 1.00 |
| Semantic | 0.8168 | 0.90 | 1.00 |
| Fine-tuned | 0.7764 | 0.90 | 1.00 |

| Human query | Semantic nDCG@10 | Fine-tuned nDCG@10 |
| --- | ---: | ---: |
| transformer architecture | 0.8820 | 0.8376 |
| cnn image recognition | 0.7516 | 0.7151 |

Synthetic development nDCG@10 increased from 0.8733 to 0.9452 for the selected checkpoint.
Human performance decreased on both queries. Synthetic label fit did not establish improved
human relevance, and two test queries cannot establish broad generalization or significance.
The original Semantic option remains the default. No later retraining or hyperparameter
selection used the observed human scores.

Full per-query rankings, scores by grade, epoch history, exclusions and configuration are in
evaluation/finetuning/training-results.json and prepared.json. The datasets do not evaluate
live retrieval recall, explanation faithfulness, or literature-overview quality.

## Reproduction

Additional requested comparison on synthetic batches 1, 5, 7, 8, 13, and 15 used all 50
verified papers per query, including papers removed from training for overlap. It reused
the frozen metadata and checked annotation hashes. No weights were changed. Results:

| Batch | TF-IDF nDCG@10 | Semantic nDCG@10 | Fine-tuned nDCG@10 |
| ---: | ---: | ---: | ---: |
| 1 | 0.7309 | 0.7955 | 0.9636 |
| 5 | 0.8256 | 0.8156 | 0.9603 |
| 7 | 0.8113 | 0.8639 | 0.9184 |
| 8 | 0.8003 | 0.6469 | 0.7836 |
| 13 | 0.7654 | 0.7985 | 1.0000 |
| 15 | 0.9552 | 0.9405 | 1.0000 |
| Macro average | 0.8148 | 0.8101 | 0.9377 |

Batches 1, 5, 7, 8, and 13 were synthetic training query groups; batch 15 was a development
group used for checkpoint selection. Their higher fine-tuned scores do not establish
held-out performance and do not supersede the human evaluation regression above.
Full rankings and provenance are in evaluation/finetuning/selected-batches.json. Reproduce
with `uv run --extra cuda python scripts/evaluate_selected_batches.py`.

From the repository root, with GPU dependencies installed:

```powershell
uv run --extra cuda python scripts/prepare_finetuning.py
uv run --extra cuda python scripts/train_ranker.py
uv run --extra cuda streamlit run app.py
```

Metadata preparation uses the internet. Training uses frozen local abstracts and a cached
base encoder. For reproducing this exact run, retain the existing prepared.json rather than
re-fetching changing metadata. Training refuses to overwrite a completed model; use a new
folder such as `--output models/reproduced-minilm` for another run. The application uses the
default models/finetuned-minilm path; restart Streamlit if that artifact is replaced.
Fixed seeds improve repeatability, but different GPU/library versions can change floats.

The generic fixed-corpus runner also accepts `--method finetuned --model-path PATH`
and `--device cuda`. Its optional `--relevant-min 2` enables the new binary threshold;
omitting the flag preserves its historical grade >0 rule, including fractional grades.

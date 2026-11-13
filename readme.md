## 1. Define the Task 
The Paper Discovery Assistant (PDA) is a tool for fast search of relevant papers on arXiv to the user query. 
The application should:
1. Interpret query and identify the key research concepts.
2. Search arXiv for potentially relevant papers.
3. Retrieve their titles, abstracts, authors, dates, categories, and links.
4. Rank the papers according to their relevance to your question.
5. Display a score, for each paper, along with an explanation of the match.
6. Let you open the original paper or abstract.
7. Generate a summary on selected paper with the respect to the query.

## 2. Prepare the Data 
Encoder model - MiniLM-L6-v2
Human evaluation dataset - 2 queries - 32 papers each - collected manually.
Synthetic evaluation dataset - 10 queries - 50 papers each - created with Undermind and ChatGpt6.
Synthetic finetuning dataset - 45 queries - 50 papers each - created with ChatGPT6.

Decoder model - Qwen3.5-4B
Synthetic 36 evaluation batches, with deepeval with Gemini API and GPT6 as judges.
The evaluation was conducted with following criteria:
1. **Factual consistency** — Are the summary's scientific claims supported by the paper?
2. **Query relevance** — Does the summary prioritize information relevant to the user's research query?
3. **Key information coverage** — Are the important query-relevant contributions, methods, results, and limitations included?
4. **Clarity and coherence** — Is the summary clear, logically organized, and understandable?
5. **Conciseness** — Is the information communicated without unnecessary detail or repetition?

### Weighted Evaluation Score

The overall summarization quality score is calculated as a weighted sum of the five metrics:

\[
S = 0.35F + 0.30R + 0.20C + 0.10H + 0.05B
\]

Where:
- \(F\) — Factual consistency (35%)
- \(R\) — Query relevance (30%)
- \(C\) — Key information coverage (20%)
- \(H\) — Clarity and coherence (10%)
- \(B\) — Conciseness (5%)

Each metric is normalized to the range \([0, 1]\), producing a weighted score \(S \in [0, 1]\).

For reporting on a 0–10 scale:

\[
S_{10} = 10S
\]

## 3. Describe the Components 
> [!IMPORTANT]
> ##### `Draw a diagram that shows the data flow. Describe the purpose, input, and output of each  component. Identify model calls and external tool calls. Explain how one input passes through  the complete system. You can use LLM for this task, but you must control the process.`

![Project diagram](diagram.png)

1. User inputs query, the rule based algorithm clears it from filler words removing filler words and extracting search terms. As a sanity check the whole query also cheked if it is a name of a paper, in that case, the exact paper would be displayed first no matter the similarity score.

2. The extraction of papers via key words is done using the official arxiv API and retrives up to 10000 candidates.

3. Candidate papers are then prepared: extracted title and abstract of each paper, and duplicate papers are deleted.

4. Selected by user algorithm encodes original query and title+abstract of each paper. (In case of TF-IDF the embedings are replaced with lexical TF-IDF vectors).

5. NumPy calculates cosine similarity between query vector and each paper.

6. The papers sorted by the similarity scores and selected ammount of them is displayed.

7. Summarization: after pressing the button on selected paper page the app downloads original pdf, extracts the content, parses it into chunks and uses Qwen locally to create a query based summarization.

## 4. Build the Prototype 
> [!IMPORTANT]
> ##### `Give the AI coding assistant a clear request and make sure you manage the instructions to your agent not only through cycles of prompting. You must create agents md file or convention md file or any other way to handle and steer agentic behaviour and context. Follow Crisp-dm by instructing your coding agent to do it properly, put instructions into the agents md or similar file.`

I managed the coding assistant through agents.md, which defined the project scope, modular architecture, CRISP-DM workflow, testing requirements, and reproducibility rules. I supplied context through existing code, design documents, datasets, training manifests, evaluation results, and observed application errors. 
Two reusable skills: `component-evaluation` and `architecture-documentation` guided evaluation and kept documentation aligned with the implementation.

## Setup

### 1. Prerequisites

The following software is required:

- **Git** — to clone the repository.
- **Python 3.13** — installed automatically through `uv`.
- **uv** — Python package and dependency manager.
- **Ollama** — to run Qwen locally for paper summarization.
- **NVIDIA GPU with CUDA-compatible drivers** — for GPU-accelerated training and evaluation.

> **Note:** All commands below are intended for **Windows PowerShell** and should be executed from the repository root, unless otherwise specified.
>
> Keep the same PowerShell session open when running the training and ranking evaluation commands, as they share environment variables.

### 2. Clone the Repository

Replace `YOUR_REPOSITORY_URL` with the repository URL:

```powershell
git clone "YOUR_REPOSITORY_URL" "Paper Discovery Assistant"
cd "Paper Discovery Assistant"
```

### 3. Install Dependencies

Install Python 3.13 and synchronize the project dependencies, including CUDA support and evaluation tools:

```powershell
uv python install 3.13
uv sync --python 3.13 --extra cuda --group evaluation
```

Verify that PyTorch can detect your NVIDIA GPU:

```powershell
uv run --extra cuda --group evaluation python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output includes:

```text
CUDA available: True
```

### 4. Configure Ollama

Paper summarization uses **Qwen 3.5 4B**, running locally through Ollama.

Start the Ollama server in a separate terminal, if it is not already running:

```powershell
ollama serve
```

Download the model (required only once):

```powershell
ollama pull qwen3.5:4b
```

> **Note:** Ollama must be running whenever the application generates paper summaries. Ranking and ranker training do not require Ollama.

### 5. Launch the Application

Start the Streamlit interface:

```powershell
uv run --extra cuda --group evaluation python -m streamlit run app.py
```

The application will display its local URL in the terminal.

### 6. Run Software Tests

Verify that the project passes its automated tests:

```powershell
uv run --extra cuda --group evaluation python -m pytest -q
```

---

## Model Training

The project supports two fine-tuned versions of the MiniLM ranking model.

- **v1** — initial fine-tuned ranking model.
- **v2** — improved fine-tuning pipeline built after v1.

### 1. Initialize Evaluation Paths

Create timestamped paths to keep experiment outputs separate:

```powershell
$run = Get-Date -Format "yyyyMMdd-HHmmss"

$rankingData = "evaluation/runs/$run/ranking-prepared.json"
$v1Data = "evaluation/runs/$run/v1/prepared.json"
$v2Data = "evaluation/runs/$run/v2/prepared.json"
```

> **Important:** Keep this PowerShell session open for the remaining training and ranking evaluation commands. The variables are used throughout the workflow.

### 2. Prepare Ranking Evaluation Data

Generate the ranking evaluation metadata:

```powershell
uv run --extra cuda --group evaluation python scripts/prepare_evaluation.py --output $rankingData
```

### 3. Train Fine-Tuned MiniLM v1

Prepare the fine-tuning dataset:

```powershell
uv run --extra cuda --group evaluation python scripts/prepare_finetuning.py --output $v1Data
```

Train the first version of the ranking model:

```powershell
uv run --extra cuda --group evaluation python scripts/train_ranker.py `
    --data $v1Data `
    --output models/finetuned-minilm `
    --epochs 3 `
    --device cuda
```

The trained model is saved to `models/finetuned-minilm`.

### 4. Train Fine-Tuned MiniLM v2

After completing v1, prepare the second fine-tuning dataset:

```powershell
uv run --extra cuda --group evaluation python scripts/prepare_finetuning_v2.py `
    --development $rankingData `
    --output $v2Data
```

Train v2:

```powershell
uv run --extra cuda --group evaluation python scripts/train_ranker_v2.py `
    --data $v2Data `
    --output models/syntetic-finetune-v2 `
    --results "evaluation/runs/$run/v2" `
    --device cuda
```

The trained model is saved to `models/syntetic-finetune-v2`.

---

## Ranking Evaluation

The evaluation pipeline compares ranking performance using human-annotated and synthetic datasets.

Two evaluation modes are available: **truncated** and **full**.

### 1. Truncated Evaluation

Run the partial ranking evaluation after training v1:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_annotations.py `
    --mode truncated `
    --device cuda `
    --prepared $rankingData `
    --training-provenance $v1Data `
    --output "evaluation/runs/$run/ranking-truncated.json"
```

**Reported metrics:**

- nDCG@10
- Precision@5
- MAP@10
- Pairwise Accuracy

### 2. Full Evaluation

Run the complete ranking evaluation:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_annotations.py `
    --mode full `
    --device cuda `
    --prepared $rankingData `
    --training-provenance $v1Data `
    --output "evaluation/runs/$run/ranking-full.json"
```

**Reported metrics:**

- nDCG@5, @10, @20, @50
- Precision@5, @10, @20
- MAP@10, @20, @50
- Pairwise Accuracy@50
- MRR@50

> **Evaluation notes:**
>
> - Human and synthetic evaluation results are reported separately.
> - Fine-tuned v1 is required for model comparison.
> - Fine-tuned v2 is included automatically when its standard model directory contains the required manifest.
> - When repeating an evaluation, use a new output filename to preserve previous results.

---

## Summarization Evaluation

The summarization pipeline uses **Qwen 3.5 4B** through Ollama to generate query-focused scientific paper summaries.

Summary quality can additionally be evaluated using **DeepEval** with **Gemini 2.5 Flash** as the judge model.

The evaluation covers **36 query–paper pairs** and includes five quality metrics:

| Metric | Weight |
|---|---:|
| Factual Consistency | 35% |
| Query Relevance | 30% |
| Key Information Coverage | 20% |
| Clarity and Coherence | 10% |
| Conciseness | 5% |

### 1. Generate Summaries Only

Generate summaries without calling Gemini:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --summarization-only
```

This mode:

- Generates summaries that are missing.
- Reuses previously generated summaries.
- Leaves new evaluation scores as `null`.
- Requires Ollama and Qwen, but **does not require a Google API key**.

### 2. Configure Gemini API Access

To enable automatic summary evaluation, create a local `.env` file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Add your Google API key:

```dotenv
GOOGLE_API_KEY=your_key_here
```

> **Security:** The `.env` file is excluded from Git. Never commit or share your API key.

### 3. Run Full Summarization Evaluation

Generate missing summaries and evaluate missing or outdated scores:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py
```

The pipeline:

1. Processes all 36 expected query–paper pairs.
2. Reuses existing summaries when available.
3. Generates missing summaries using Qwen.
4. Evaluates missing or outdated scores using Gemini 2.5 Flash.
5. Saves individual evaluation records and aggregate CSV/JSON reports.

### 4. Additional Evaluation Modes

The following commands support testing, resuming, and rebuilding results.

#### Smoke Test

Run a one-paper test to verify the evaluation pipeline:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --smoke
```

#### Evaluation Only

Evaluate previously cached summaries without generating new ones:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --evaluation-only
```

#### Aggregate Only

Rebuild aggregate reports from existing evaluation records without calling either model:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --aggregate-only
```

#### Force Re-evaluation

Explicitly recompute judge scores while preserving existing summaries:

```powershell
uv run --extra cuda --group evaluation python scripts/evaluate_summaries.py --evaluation-only --force-evaluate
```

> **API usage:** Gemini evaluation requires access to the Google Generative Language API and is subject to the provider's rate limits and quotas. The `--summarization-only` and `--aggregate-only` modes do not call Gemini.


## 5. Evaluate the System 
> [!IMPORTANT]
> ##### `Evaluation is required for every component of the project. It can be also done by an LLM but you must be able to distinguish between high and low scores, think of how to improve it and explain why scores are not as good as they could be (if the case). The selection of metrics will also be judged, so make sure you choose proper metrics.`

### Encoding and Ranking

Human evaluation dataset - 2 queries - 32 papers each - collected manually.
Synthetic evaluation dataset - 10 queries - 50 papers each - created with Undermind and ChatGpt6 with Undermind as a plugin.

#### Human evaluation: 2 queries, 64 verified judgments.

| Metric | TF-IDF | Semantic | Fine-tune v1 | Fine-tune v2 |
|---|---:|---:|---:|---:|
| nDCG@5 | 0.7065 | 0.7997 | 0.7128 | 0.8097 |
| nDCG@10 | 0.7558 | 0.8168 | 0.7764 | 0.8138 |
| nDCG@20 | 0.8093 | 0.8183 | 0.7716 | 0.8236 |
| nDCG@50 | 0.9056 | 0.9311 | 0.8889 | 0.9355 |
| Precision@5 | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| Precision@10 | 0.9000 | 0.9500 | 0.9500 | 0.9000 |
| Precision@20 | 0.8000 | 0.8000 | 0.8000 | 0.8250 |
| MAP@10 | 0.8521 | 0.8952 | 0.8786 | 0.8627 |
| MAP@20 | 0.7257 | 0.7393 | 0.7272 | 0.7677 |
| MAP@50 | 0.8733 | 0.8842 | 0.8749 | 0.8984 |
| Pairwise accuracy@50 | 0.6587 | 0.6863 | 0.6644 | 0.7040 |
| MRR@50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### Synthetic evaluation: 10 queries, 484 verified judgments.

| Metric | TF-IDF | Semantic | Fine-tune v1 | Fine-tune v2 |
|---|---:|---:|---:|---:|
| nDCG@5 | 0.7898 | 0.8616 | 0.8972 | 0.9076 |
| nDCG@10 | 0.7953 | 0.8349 | 0.8645 | 0.8685 |
| nDCG@20 | 0.7697 | 0.8227 | 0.8532 | 0.8635 |
| nDCG@50 | 0.9168 | 0.9418 | 0.9518 | 0.9543 |
| Precision@5 | 0.8600 | 0.9600 | 0.9800 | 1.0000 |
| Precision@10 | 0.9000 | 0.9600 | 0.9600 | 0.9500 |
| Precision@20 | 0.8650 | 0.9450 | 0.9400 | 0.9550 |
| MAP@10 | 0.8302 | 0.9427 | 0.9433 | 0.9390 |
| MAP@20 | 0.7833 | 0.9165 | 0.9158 | 0.9300 |
| MAP@50 | 0.8675 | 0.9456 | 0.9349 | 0.9377 |
| Pairwise accuracy@50 | 0.6905 | 0.7824 | 0.7869 | 0.8023 |
| MRR@50 | 0.9500 | 1.0000 | 1.0000 | 1.0000 |

#### Conclusions
1. Semantic ranking is effective.
- Pretrained semantic embeddings consistently outperform TF-IDF in graded relevance ranking, demonstrating the benefit of contextual text representations for research-paper discovery.

2. Synthetic-only fine-tuning has limited demonstrated transfer to human judgments.
- V1 improves nDCG on synthetic evaluation queries but decreases it on human-evaluated queries. This suggests possible differences in what the two evaluation sets reward, although the small human sample prevents a definitive explanation.

3. Expanded fine-tuning improves model performance.
- V2 substantially improves upon V1, particularly on human nDCG@5 and pairwise accuracy, while maintaining strong synthetic evaluation performance.

4. Fine-tuning appears especially effective for graded relevance ordering.
- Improvements in nDCG are generally more pronounced than improvements in MAP or Precision, suggesting that fine-tuning may improve the relative ordering of relevance levels more than binary relevance identification.

5. V2 is competitive, but its superiority is not established.
- V2 outperforms the pretrained model on several nDCG cutoffs and pairwise accuracy, but the pretrained model retains advantages in some MAP and Precision measurements.

6. More independent human evaluation is necessary.
- The findings indicate useful ranking improvements, but additional human-labeled queries and an untouched test set are required to evaluate generalization more confidently.

### Summarization

Summarizations were created on evaluation datasets for ranking. From each batch was taken query and first 3 papers. Qwen performed summarization and LLMs were used as a judge, specifically external ChatGPT6. Python library `DeepEval` can be used with the connection to Gemini API. However, Gemini API rans out of the usage quickly, so to keep evaluation relevant ChatGPT6 was used for all ratings.

#### Individual Summary Evaluation

| ID | Source | Query | Paper | Fact | Rel | Cover | Clear | Brief | Overall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 01 | human | transformer architecture | Attention Is All You Need | 0.80 | 1.00 | 0.90 | 1.00 | 1.00 | 9.10 |
| 02 | human | transformer architecture | An Introduction to Transformers | 0.60 | 1.00 | 0.70 | 0.80 | 0.90 | 7.75 |
| 03 | human | transformer architecture | A Mathematical Explanation of Transformers | 0.70 | 0.90 | 0.80 | 0.90 | 0.90 | 8.10 |
| 04 | human | cnn image recognition | Food Image Recognition by Using Convolutional Neural Networks (CNNs) | 0.90 | 1.00 | 0.90 | 1.00 | 1.00 | 9.45 |
| 05 | human | cnn image recognition | CNN-RNN: A Unified Framework for Multi-label Image Classification | 0.60 | 1.00 | 0.80 | 0.90 | 0.90 | 8.05 |
| 06 | human | cnn image recognition | Very Deep Convolutional Networks for Large-Scale Image Recognition | 0.60 | 1.00 | 0.80 | 0.90 | 0.90 | 8.05 |
| 07 | synthetic | Contrastive self-supervised learning for visual representation | A Simple Framework for Contrastive Learning of Visual Representations | 0.60 | 1.00 | 0.90 | 0.90 | 1.00 | 8.30 |
| 08 | synthetic | Contrastive self-supervised learning for visual representation | Momentum Contrast for Unsupervised Visual Representation Learning | 0.50 | 1.00 | 0.70 | 0.90 | 0.80 | 7.45 |
| 09 | synthetic | Contrastive self-supervised learning for visual representation | Improved Baselines with Momentum Contrastive Learning | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 9.65 |
| 10 | synthetic | Differential privacy in deep learning | Large Language Models Can Be Strong Differentially Private Learners | 0.70 | 1.00 | 0.80 | 0.90 | 0.80 | 8.35 |
| 11 | synthetic | Differential privacy in deep learning | Fine-Tuning Large Language Models with User-Level Differential Privacy | 0.70 | 1.00 | 0.80 | 0.90 | 0.80 | 8.35 |
| 12 | synthetic | Differential privacy in deep learning | Differentially Private Subspace Fine-Tuning for Large Language Models | 0.90 | 1.00 | 0.90 | 1.00 | 0.90 | 9.40 |
| 13 | synthetic | Learning with noisy labels for image classification | Mitigating Memorization in Sample Selection for Learning with Noisy Labels | 0.60 | 1.00 | 0.80 | 0.90 | 0.90 | 8.05 |
| 14 | synthetic | Learning with noisy labels for image classification | Training Convolutional Networks with Noisy Labels | 0.60 | 1.00 | 0.80 | 0.90 | 0.90 | 8.05 |
| 15 | synthetic | Learning with noisy labels for image classification | Training Deep Neural Networks on Noisy Labels with Bootstrapping | 0.70 | 1.00 | 0.90 | 1.00 | 0.90 | 8.70 |
| 16 | synthetic | Neural radiance fields for 3D scene reconstruction | Neural Radiance Fields (NeRFs): A Review and Some Recent Developments | 0.80 | 1.00 | 0.80 | 0.90 | 0.90 | 8.75 |
| 17 | synthetic | Neural radiance fields for 3D scene reconstruction | Learning Neural Radiance Fields from Multi-View Geometry | 0.70 | 1.00 | 0.90 | 0.80 | 0.90 | 8.50 |
| 18 | synthetic | Neural radiance fields for 3D scene reconstruction | NeRF--: Neural Radiance Fields Without Known Camera Parameters | 0.50 | 1.00 | 0.80 | 0.90 | 0.90 | 7.70 |
| 19 | synthetic | Knowledge graph completion and link prediction | A Survey on Graph Neural Networks for Knowledge Graph Completion | 0.80 | 1.00 | 0.80 | 0.80 | 0.80 | 8.60 |
| 20 | synthetic | Knowledge graph completion and link prediction | An overview of embedding models of entities and relationships for knowledge base completion | 0.50 | 1.00 | 0.50 | 0.90 | 0.90 | 7.10 |
| 21 | synthetic | Knowledge graph completion and link prediction | A Relational Tucker Decomposition for Multi-Relational Link Prediction | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 | 9.80 |
| 22 | synthetic | Deep learning for ECG arrhythmia detection | Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks | 1.00 | 1.00 | 0.90 | 1.00 | 0.90 | 9.75 |
| 23 | synthetic | Deep learning for ECG arrhythmia detection | Arrhythmia Classification from 12-Lead ECG Signals Using Convolutional and Transformer-Based Deep Learning Models | 0.70 | 1.00 | 0.90 | 0.90 | 0.90 | 8.60 |
| 24 | synthetic | Deep learning for ECG arrhythmia detection | A Lightweight CNN-Attention-BiLSTM Architecture for Multi-Class Arrhythmia Classification on Standard and Wearable ECGs | 0.70 | 1.00 | 0.90 | 1.00 | 0.90 | 8.70 |
| 25 | synthetic | Reinforcement learning for autonomous navigation | Deep Reinforcement learning for real autonomous mobile robot navigation in indoor environments | 0.60 | 1.00 | 0.90 | 0.90 | 0.90 | 8.25 |
| 26 | synthetic | Reinforcement learning for autonomous navigation | Sim-to-Real Transfer for Mobile Robots with Reinforcement Learning: from NVIDIA Isaac Sim to Gazebo and Real ROS 2 Robots | 0.80 | 1.00 | 0.80 | 1.00 | 0.90 | 8.85 |
| 27 | synthetic | Reinforcement learning for autonomous navigation | Parallel Distributional Deep Reinforcement Learning for Mapless Navigation of Terrestrial Mobile Robots | 0.50 | 1.00 | 0.80 | 0.90 | 0.90 | 7.70 |
| 28 | synthetic | Multimodal emotion recognition using audio and visual signals | Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention | 0.90 | 1.00 | 0.90 | 1.00 | 0.90 | 9.40 |
| 29 | synthetic | Multimodal emotion recognition using audio and visual signals | Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling | 0.80 | 1.00 | 0.90 | 1.00 | 0.90 | 9.05 |
| 30 | synthetic | Multimodal emotion recognition using audio and visual signals | Human Emotion Detection from Audio and Video Signals | 0.40 | 0.80 | 0.50 | 0.90 | 0.80 | 6.10 |
| 31 | synthetic | Deep learning for satellite image change detection | SiamixFormer: a fully-transformer Siamese network with temporal Fusion for accurate building detection and change detection in bi-temporal remote sensing images | 0.50 | 1.00 | 0.80 | 0.90 | 0.90 | 7.70 |
| 32 | synthetic | Deep learning for satellite image change detection | Semantic Change Detection with Asymmetric Siamese Networks | 0.80 | 1.00 | 0.90 | 0.90 | 0.90 | 8.95 |
| 33 | synthetic | Deep learning for satellite image change detection | NeXt2Former-CD: Efficient Remote Sensing Change Detection with Modern Vision Architectures | 0.90 | 1.00 | 0.90 | 1.00 | 0.90 | 9.40 |
| 34 | synthetic | Deep learning for image inpainting and missing-region reconstruction | Both Spatial and Frequency Cues Contribute to High-Fidelity Image Inpainting | 1.00 | 1.00 | 0.90 | 1.00 | 0.90 | 9.75 |
| 35 | synthetic | Deep learning for image inpainting and missing-region reconstruction | Frequency-Domain Fusion Transformer for Image Inpainting | 0.70 | 0.80 | 0.70 | 0.90 | 0.80 | 7.55 |
| 36 | synthetic | Deep learning for image inpainting and missing-region reconstruction | Feature Refinement to Improve High Resolution Image Inpainting | 0.80 | 1.00 | 0.90 | 1.00 | 0.90 | 9.05 |


### Aggregate Evaluation Metrics

| Metric | N | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| Factual Consistency | 36 | 0.72 | 0.16 | 0.40 | 1.00 |
| Query Relevance | 36 | 0.99 | 0.05 | 0.80 | 1.00 |
| Key Information Coverage | 36 | 0.83 | 0.11 | 0.50 | 1.00 |
| Clarity | 36 | 0.93 | 0.06 | 0.80 | 1.00 |
| Conciseness | 36 | 0.90 | 0.06 | 0.80 | 1.00 |
| Weighted Score (0–1) | 36 | 0.85 | 0.08 | 0.61 | 0.98 |
| Weighted Score (0–10) | 36 | 8.50 | 0.83 | 6.10 | 9.80 |


### Evaluation Results by Dataset and Query

| Dataset / Query | Completed | Overall /10 |
|---|---:|---:|
| **Human Evaluation** | **6/6** | **8.42** |
| Transformer Architecture | 3/3 | 8.32 |
| CNN Image Recognition | 3/3 | 8.52 |
| **Synthetic Evaluation** | **30/30** | **8.52** |
| Contrastive Self-Supervised Learning | 3/3 | 8.47 |
| Differential Privacy in Deep Learning | 3/3 | 8.70 |
| Learning with Noisy Labels | 3/3 | 8.27 |
| Neural Radiance Fields | 3/3 | 8.32 |
| Knowledge Graph Completion | 3/3 | 8.50 |
| ECG Arrhythmia Detection | 3/3 | 9.02 |
| Reinforcement Learning for Navigation | 3/3 | 8.27 |
| Multimodal Emotion Recognition | 3/3 | 8.18 |
| Satellite Change Detection | 3/3 | 8.68 |
| Deep Learning for Image Inpainting | 3/3 | 8.78 |
| **Total** | **36/36** | **8.50** |

Mean summarization duration: 112.98 s (n=36)
Mean evaluation duration: -- s (n=0)

#### Conclusions

1. Query-focused summarization is effective.
- The summarization pipeline achieves high query relevance (0.99), indicating that Qwen consistently generates summaries focused on the user's research question rather than providing only general descriptions of papers.

2. Factual consistency is the main limitation.
- Factual consistency received the lowest average score (0.72) and showed the highest variability (0.16). This suggests that summaries may contain unsupported or inaccurate scientific claims, particularly regarding numerical results and experimental findings.

3. Summaries are generally clear and concise.
- High clarity (0.93) and conciseness (0.90) scores indicate that the generated summaries are readable, well-structured, and relatively compact, making them suitable for quickly reviewing research papers.

4. Information coverage remains imperfect.
- The average key information coverage score of 0.83 suggests that summaries generally capture important findings but may omit relevant methodologies, experimental details, or limitations. This indicates a potential trade-off between summary length and completeness.

5. Summarization quality is relatively consistent across evaluation datasets.
- The average scores for human-selected (8.42/10) and synthetic-selected (8.52/10) papers are similar. Using the same LLM judge for all 36 summaries provides a more consistent evaluation, although the limited sample size prevents establishing performance consistency across research topics.

6. Summarization introduces considerable computational overhead.
- The average summarization time of 112.98 seconds per paper indicates that generating summaries from full research papers is computationally expensive. This limits the practicality of generating multiple summaries during interactive paper discovery.

7. Further evaluation and improvement are necessary.
- The overall score of 8.50/10 indicates promising summarization quality, but the evaluation relies on a single LLM judge rather than independent human verification. Future improvements should prioritize factual consistency, information coverage, and inference efficiency.
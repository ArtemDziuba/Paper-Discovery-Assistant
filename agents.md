# Project Instructions — Paper Discovery Assistant

## 1. Project Overview

The Paper Discovery Assistant (PDA) is an NLP-based application that helps users discover relevant academic papers on arXiv.

The application must:

1. Accept a research question or topic.
2. Interpret the query and identify key research concepts.
3. Search arXiv for potentially relevant papers.
4. Retrieve paper metadata, including titles, abstracts, authors, publication dates, categories, and links.
5. Rank papers according to their relevance to the user's query.
6. Display relevance scores and explanations for individual papers.
7. Allow users to open the original paper or abstract.
8. Generate a literature overview grounded in retrieved papers, with verifiable citations.

The primary goal is to develop a working, explainable, and evaluable NLP prototype rather than a production-scale application.

## 2. Academic Assignment Requirements

This project is developed as an NLP course assignment using an AI coding assistant.

Development must satisfy the following requirements:

- Follow the CRISP-DM methodology.
- Maintain a modular NLP architecture.
- Identify the purpose, inputs, and outputs of every component.
- Document model calls and external API calls.
- Evaluate the quality of every major NLP component.
- Implement automated software tests.
- Create reusable agent skills to support development.
- Maintain reproducible experiments and evaluation results.
- Provide architecture and data-flow diagrams.
- Document important design decisions and limitations.
- Ensure the application can be run locally and demonstrated.

Prioritize correctness, explainability, evaluation, and reproducibility over unnecessary complexity.

## 3. Technology Stack

Preferred technologies:

- Python 3.12+
- FastAPI for backend API
- Pydantic for data validation
- Sentence Transformers for semantic embeddings
- NumPy for similarity calculations
- pytest for automated testing

Additional libraries may be introduced when justified by project requirements.

Do not introduce FAISS, vector databases, orchestration frameworks, or additional infrastructure unless their benefits justify the added complexity.

Keep the system simple enough to run locally.

## 4. Architecture

Follow a modular architecture with clearly separated responsibilities.

The expected logical components are:

### Query Interpretation

- Extract key research concepts from user queries.
- Produce structured search terms or an improved search query.
- Preserve the original meaning and intent.

### Retrieval

- Communicate with the arXiv API.
- Retrieve candidate papers and their metadata.
- Handle API failures and rate limits.
- Preserve original paper identifiers and links.

### Preprocessing

- Validate and normalize retrieved metadata.
- Prepare paper titles and abstracts for embedding generation.
- Handle missing or incomplete metadata.
- Avoid unnecessary transformations that alter semantic meaning.

### Embeddings

- Generate semantic representations of user queries and papers.
- Use a consistent embedding model.
- Support reproducible embedding generation.

### Ranking

- Calculate semantic similarity between queries and candidate papers.
- Rank papers by estimated relevance.
- Produce interpretable scores.
- Keep raw similarity values separate from presentation scores.

### Match Explanation

- Explain why individual papers match the user's query.
- Ground explanations in paper titles, abstracts, and available metadata.
- Avoid unsupported claims about paper contents.

### Summarization

- Generate a literature overview from retrieved papers.
- Ground statements in the available source material.
- Provide verifiable citations.
- Distinguish findings supported by abstracts from claims requiring full-text access.

### API and Interface

- Accept user queries.
- Coordinate the processing pipeline.
- Return ranked papers, scores, explanations, and summaries.
- Provide links to original sources.
- Handle errors clearly.

### Evaluation

- Evaluate individual NLP components.
- Evaluate the end-to-end system.
- Store reproducible evaluation results.
- Document limitations and potential improvements.

These are logical components and do not necessarily require separate services or models.

Keep business logic separate from API handlers.

## 5. Development Methodology — CRISP-DM

Follow the six phases of CRISP-DM throughout development.

### 5.1 Business Understanding

- Define the problem and target users.
- Identify functional and nonfunctional requirements.
- Define project scope and limitations.
- Establish measurable success criteria.

### 5.2 Data Understanding

- Investigate available data sources.
- Examine arXiv metadata and abstract availability.
- Identify data quality issues.
- Investigate existing datasets with relevance judgments before creating synthetic evaluation data.

### 5.3 Data Preparation

- Implement data retrieval and preprocessing.
- Handle missing values and duplicates.
- Preserve source identifiers.
- Prepare evaluation datasets.
- Document data transformations.

### 5.4 Modeling

- Select appropriate NLP methods for each component.
- Establish simple baselines before introducing complex approaches.
- Document model selection and configuration.
- Explain the purpose of every model call.

### 5.5 Evaluation

- Evaluate each major NLP component independently.
- Compare results against defined baselines or reference judgments.
- Analyze errors and failure cases.
- Determine whether the system satisfies its original objectives.

### 5.6 Deployment

- Provide a locally runnable prototype.
- Document installation and execution instructions.
- Verify the complete application workflow.
- Prepare representative demonstration cases.

CRISP-DM is iterative. Revisit earlier phases when evaluation reveals problems.

Document how the phases were applied in the final report.

## 6. Evaluation Requirements

Every major NLP component must have an appropriate evaluation strategy.

Distinguish between:

- Software correctness: whether the implementation behaves as expected.
- NLP quality: whether the component produces useful and accurate results.

Suggested evaluation approaches:

| Component | Evaluation |
|---|---|
| Query interpretation | Concept preservation, concept coverage, manual or LLM-assisted judgments |
| Retrieval | Recall@k, candidate coverage, retrieval failure analysis |
| Preprocessing | Metadata completeness, validation accuracy, duplicate detection |
| Embeddings | Semantic similarity quality on labeled query-paper pairs |
| Ranking | nDCG@k, MRR, Precision@k |
| Match explanation | Faithfulness, relevance, unsupported-claim rate |
| Summarization | Factual consistency, citation correctness, coverage |
| End-to-end system | Task success, response time, usefulness |

These metrics are candidates, not mandatory choices. Select metrics based on the actual component implementation and available ground truth.

Evaluation rules:

1. Define evaluation criteria before interpreting results.
2. Prefer real datasets and human relevance judgments where feasible.
3. Use synthetic data only when justified and clearly identify it.
4. Avoid evaluating a model solely using its own judgments.
5. Do not report fabricated or unexecuted evaluation results.
6. Compare against simple baselines where appropriate.
7. Include successful cases and failure cases.
8. Document why performance may be limited.
9. Save evaluation configurations and results for reproducibility.

If a component cannot be evaluated quantitatively, propose a structured qualitative evaluation.

Do not claim that a component is effective merely because its code executes successfully.

## 7. Relevance Scoring

The application should display relevance scores on a 0–100 scale.

However:

- Raw cosine similarity is not a probability of relevance.
- A normalized similarity score is not automatically calibrated.
- Do not describe a score as a percentage probability unless calibration has been performed and evaluated.
- Document the scoring transformation.
- Keep raw model scores available for evaluation.
- Explain the meaning and limitations of displayed scores.

Evaluate whether higher scores correspond to greater relevance using labeled examples.

## 8. Coding Standards

- Follow PEP 8.
- Use Python type hints.
- Prefer small, focused functions.
- Use descriptive names.
- Keep modules loosely coupled.
- Avoid duplicated functionality.
- Separate configuration from business logic.
- Avoid unnecessary dependencies and abstractions.
- Use structured logging where appropriate.
- Never hardcode API keys or secrets.
- Keep configuration and random seeds reproducible where practical.

## 9. Testing

Use pytest for automated testing.

Testing requirements:

- Write unit tests for important functionality.
- Mock external API calls in unit tests.
- Test invalid inputs and edge cases.
- Test API response parsing and validation.
- Test component integration.
- Run relevant tests after modifications.
- Report failing tests rather than silently ignoring them.

Do not confuse unit tests with NLP quality evaluation.

## 10. AI Agent Development Workflow

For substantial development tasks, follow this process:

### Step 1 — Understand

- Inspect relevant project files.
- Read applicable documentation.
- Identify affected components.
- Clarify important ambiguities before implementation.

### Step 2 — Plan

- Propose an implementation approach.
- Identify files that need modification.
- Identify dependencies and risks.
- Define acceptance criteria.
- For major architectural changes, request approval before implementation.

### Step 3 — Implement

- Follow the approved architecture.
- Reuse existing functionality.
- Keep changes limited to the requested task.
- Avoid unrelated refactoring.
- Preserve public interfaces unless changes are explicitly approved.

### Step 4 — Verify

- Run relevant automated tests.
- Check expected behavior.
- Run component evaluations when the change affects NLP quality.
- Investigate and report failures.

### Step 5 — Document

- Update relevant documentation.
- Record important design decisions.
- Update evaluation results when applicable.
- Keep architecture descriptions consistent with implementation.

### Step 6 — Report

After completing a task, summarize:

- What was implemented.
- Which files changed.
- Which tests and evaluations were executed.
- Their actual results.
- Known limitations.
- Any remaining work.

Do not claim that tests passed unless they were executed successfully.

## 11. Agent Skills

Create and maintain reusable Codex skills for recurring development workflows.

At minimum, support:

### Component Evaluation Skill

A reusable workflow for:

- Identifying component inputs and outputs.
- Selecting appropriate evaluation metrics.
- Running available evaluation scripts.
- Comparing results with baselines.
- Analyzing failure cases.
- Producing evaluation reports.

### Architecture Documentation Skill

A reusable workflow for:

- Inspecting the implemented architecture.
- Identifying components and dependencies.
- Documenting component inputs and outputs.
- Identifying model and external API calls.
- Generating or updating data-flow diagrams.
- Checking consistency between documentation and implementation.

Skills must be implemented as actual Codex skills with valid `SKILL.md` files.

Do not treat descriptions in this file as substitutes for implementing the skills.

## 12. Documentation

Maintain documentation covering:

- Project requirements and scope.
- System architecture.
- Component responsibilities.
- End-to-end data flow.
- Model and technology selection.
- Data sources and preparation.
- Evaluation methodology and results.
- CRISP-DM development process.
- Installation and execution instructions.
- Known limitations.

Documentation must describe the actual implementation.

Update relevant documentation when architectural changes occur.

Prefer Mermaid diagrams for architecture and data-flow documentation.

## 13. Reproducibility

- Record model names and versions.
- Record important configuration parameters.
- Use fixed random seeds when applicable.
- Preserve evaluation datasets and relevance judgments.
- Store evaluation results in a structured format.
- Document external dependencies.
- Make it possible to reproduce reported metrics.

Do not commit credentials or sensitive configuration values.

## 14. Important Constraints

- Never fabricate papers, metadata, or citations.
- Preserve original arXiv identifiers.
- Verify that citations reference retrieved papers.
- Do not invent claims unsupported by the available source material.
- Do not present abstract-based summaries as full-paper analyses.
- Distinguish semantic similarity from calibrated relevance.
- Do not present cosine similarity as a probability.
- Do not fabricate experimental results.
- Do not silently change evaluation methodology.
- Do not modify unrelated functionality.
- Do not introduce unnecessary architectural complexity.

## 15. Project Priorities

When tradeoffs arise, prioritize:

1. Working end-to-end functionality.
2. Correctness and source grounding.
3. Meaningful evaluation of every major component.
4. Explainable architecture and reproducibility.
5. Code quality and maintainability.
6. Additional features and optimization.

The final system must be understandable, demonstrable, and supported by evaluation evidence.
## 16. Confirmed Prototype Constraints

- Build the first prototype within a two-week deadline. Deliver as quickly as dependencies and verification permit; suggested day ranges are not mandatory waiting periods. Preserve required testing, evaluation, and documentation.
- Use Streamlit for the initial web interface; FastAPI is not required.
- Support up to 10,000 arXiv candidate entries. Longer searches are acceptable; preserve API rate limits, paginate requests, report progress and partial failures, and keep ranking within the laptop memory budget.
- Run all NLP inference locally; do not use hosted LLM inference APIs.
- Available GPU: laptop NVIDIA RTX 4060 with 8 GB VRAM. Full GPU utilization is permitted.
- Target application system RAM usage below 8 GB; 12-13 GB is the upper allowance when justified. Measure actual usage before claiming compliance.
- Do not edit readme.md without consulting the user first. Place other project documentation in docs/.
- Include component-level quality evaluation, automated software tests, CRISP-DM evidence, architecture/data-flow diagrams, and reusable Codex skills.
- Prepare code, setup/run instructions, a short report, and a demonstration outline for a video of at most 10 minutes.
- Internet access for paper retrieval through the arXiv API is approved. Keep all NLP inference local; treat initial model downloads as setup dependencies, not hosted inference.

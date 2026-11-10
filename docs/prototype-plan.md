# Prototype plan (proposed)

Status: baseline architecture authorized by the request to implement. The Streamlit prototype,
automated tests, and fixed-corpus ranking evaluations are implemented. See architecture.md,
setup.md, evaluation.md, and implementation-results.md for actual behavior and verification.

## Scope and constraints

Build a local Streamlit application for English-language research queries and arXiv title/abstract search. Preserve paper identifiers, source links, raw similarity scores, and citation provenance. The initial scope excluded full-text analysis; the later user-requested extension adds on-demand, query-focused per-paper PDF summaries through local Qwen3.5-4B/Ollama. Accounts, cloud deployment, vector databases and hosted inference remain excluded. See summarization.md for the implemented extension and validation criteria.

Target hardware: laptop RTX 4060, 8 GB VRAM. Target system RAM below 8 GB, with an upper allowance of 12-13 GB when justified. Resource usage and latency must be measured on representative workloads. The user authorized up to 10,000 arXiv candidates and longer searches; use paginated retrieval, progress reporting, chunked semantic encoding, and sparse lexical scoring.

Internet access for paper retrieval through the arXiv API is approved. All NLP inference runs locally. Initial model downloads are setup dependencies rather than hosted inference. Propose cached retrieved records and an explicitly labeled replay mode for a repeatable offline demo.

## Baseline architecture

Streamlit calls ordinary Python components directly. Pydantic validates shared records. Use a lightweight, deterministic concept-extraction baseline, an arXiv retrieval client, metadata normalization/deduplication, a local Sentence Transformers encoder, NumPy cosine ranking, and per-paper match evidence with source links. Topic retrieval removes common conversational prefixes and requires retained concepts together; quoted phrases preserve their full wording. Optional Windows CUDA dependencies are selected with the cuda extra.

The original baseline used extractive explanations. The user subsequently selected local
`qwen3.5:4b` for explicit per-paper summary requests. Its separate PDF/extraction/inference
path does not modify retrieval, ranking or MiniLM training. The cross-paper overview remains
outside scope, and actual summary quality/resource use must be measured independently.

```mermaid
flowchart TD
    A[Research query in Streamlit] --> B[Concept extraction]
    B --> C[arXiv retrieval or labeled cache replay]
    C --> D[Metadata validation and deduplication]
    A --> E[Local query embedding]
    D --> F[Local title and abstract embeddings]
    E --> G[Cosine ranking]
    F --> G
    G --> H[Grounded match explanations]
    H --> J[Streamlit ranked results and source links]
```

The arXiv client makes an external API call. The encoder makes local model calls. Per-paper extractive explanations require no generative model. The overview feature has been removed at the user's request; focus on ranked papers, with 5–50 displayed results in increments of 5. Store configuration, model revision, retrieval time, and stage timings for reproducibility.

## Evaluation before interpreting results

Investigate real academic retrieval datasets before preparing synthetic data. Check domain fit, licenses, accessible abstracts, relevance labels, and identifier compatibility. Evaluate ranking on a fixed judged corpus separately from live arXiv candidate retrieval; do not imply that ranking metrics establish arXiv-wide recall.

| Stage | Proposed quality evidence |
| --- | --- |
| Query interpretation | Human-labeled concept coverage and intent preservation, including negation and ambiguous queries |
| Retrieval | Coverage of known relevant papers where judgments permit; failures and scope of incomplete judgments |
| Preprocessing | Metadata completeness, duplicate-detection errors, and an audited sample |
| Embeddings | Separation of judged relevant and irrelevant query-paper pairs; compare lexical baseline |
| Ranking | nDCG@10 and Precision@5 against graded/binary judgments and a lexical baseline |
| Explanations | Human rubric for evidence support, relevance, and unsupported claims |
| On-demand paper summary | Full-text extraction/processing coverage, human-reviewed claim support, query relevance, contribution/method/result/limitation coverage, and a 100–150 word target with internal tolerance of 50–180 |
| End to end | Representative task completion, measured latency, RAM/VRAM use, and failure cases |

Automated tests verify correctness separately: mocked API responses, malformed/missing metadata, identifier preservation, empty results, similarity calculations, citation membership, and pipeline integration. Extracting a source sentence does not by itself establish that it is useful or representative.

## Delivery sequence

The two-week window is a deadline, not a prescribed work duration. Proceed as quickly as dependencies and verification permit; do not wait for calendar milestones. The stages below describe a logical order and may overlap where appropriate. Preserve evaluation, testing, and documentation requirements while accelerating delivery.

1. Establish business requirements, investigate data, define evaluation criteria, document architecture, and create reusable skills.
2. Implement retrieval, metadata preparation, caching, and software tests.
3. Add embeddings, a lexical baseline, ranking, explanations, and Streamlit integration.
4. Validate source links and the end-to-end ranked search workflow.
5. Run component evaluations, analyze errors, and measure resource use.
6. Fix issues, check reproducibility, complete the report and diagrams, and rehearse the demo.

## Planned files and acceptance

Application code will live under src/, with a Streamlit entry point, tests/, evaluation datasets/configurations/results, docs/, and actual reusable skills under .agents/skills/. Create installation/run instructions in docs/ without changing readme.md.

Acceptance: a query produces real retrieved papers, ordered scores explicitly described as similarity rather than probability, per-paper evidence and source links, and clear handling of empty results/API failures. Display 5–50 papers in steps of 5, subject to available candidates. Include a reproducible evaluation command and actual per-component results, limitations, and demo cases. Do not claim quality, hardware compliance, or passing tests until measured.

Complete-title lookup supplements topic retrieval and prioritizes actual exact returned
titles, with optional ASCII/smart quotes. The auxiliary request and merge remain inside
the existing retrieval and ranking components; see architecture.md for provenance and limits.

## Risks and decisions to resolve

- arXiv API availability, rate limits, and clear handling of network failures.
- The overview was removed from scope at the user's request; retain per-paper evidence and links.
- Dataset relevance to live arXiv search and the availability of independent human judgments.
- Local model downloads, input truncation, CPU/GPU compatibility, and measured memory use.

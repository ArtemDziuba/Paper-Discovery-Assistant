---
name: architecture-documentation
description: Update PDA architecture and data-flow documentation from implemented components, including local model and external API boundaries.
---

Inspect app.py and src/pda/ before editing docs/architecture.md. Trace one query through
interpretation, arXiv retrieval/cache, validation, encoding, ranking, evidence selection,
overview, and presentation. Document actual inputs and outputs and mark external API calls
and local model calls separately. Update Mermaid diagrams to match code paths, including
empty-result and failure behavior. Check configuration, model revision, score transformation,
cache lifetime, and abstract truncation against the implementation.

Keep proposed changes distinct from implemented behavior. Check docs/prototype-plan.md for
scope consistency. Do not change readme.md without consulting the user. Report documentation
inconsistencies and remaining implementation gaps rather than documenting planned features
as complete.

---
name: Feature / Task
about: New capability, enhancement, or technical task
title: '<layer/area>: <short outcome>'
labels: ''
assignees: ''
---

## Context
<!-- What exists today, what is missing, and why this matters.
     Name the specific existing modules/files this relates to.
     Example: "retrieve_context() already dedupes candidates and optionally reranks them
     via RERANK_PROVIDER. This issue adds a per-document relevance grade that drops
     irrelevant chunks before they reach the answer prompt." -->


## Layer & module
<!-- Exact layer and module path for new code. Respects the layered pipeline in CLAUDE.md
     (API -> Services -> Ingestion/Index/Search -> LLM providers) - no business logic in
     api/, no direct index access from api/.
     Example:
       Layer:  search
       Module: src/lahmajo/search/rerank_provider.py -->

- **Layer:**
- **Module:**

## What already exists (do not recreate)
<!-- List modules/functions that already exist and must be reused or extended.
     Agents must read these before writing any code.
     Example:
       - `RerankProvider` ABC (`search/rerank_provider.py`) - implement it, do not invent a new interface
       - `get_llm()` (`llm/llm_provider.py`) - reuse the configured LLM, do not instantiate one directly
       - `retrieve_context()` (`services/retrieval_service.py`) - extend, do not fork -->


## What to build
<!-- Specific functions/classes/env vars to create or modify, with brief role.
     Example:
       - Create `CrossEncoderRerankProvider(RerankProvider)` in `search/rerank_provider.py`
       - Add a `cross_encoder` branch to `get_rerank_provider()`
       - Add `RERANK_MODEL` env var (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) -->


## Acceptance criteria
<!-- Observable, checkable outcomes. Avoid vague goals.
     Always include the test suite passing, plus anything endpoint/behaviour-specific.
     Example:
       - [ ] `RERANK_PROVIDER=cross_encoder` selects the new provider
       - [ ] An unknown provider value still raises ValueError listing the supported names
       - [ ] Default behaviour is unchanged when the env var is unset
       - [ ] `python run_tests.py` passes -->


## What NOT to touch
<!-- Explicit boundaries to prevent agents from over-refactoring.
     Example:
       - Do not change the RRF logic in `HybridRetriever.search()`
       - Do not add the new dependency to base `requirements.txt` -->


## Integration points
<!-- Which existing code calls this new code, or is called by it.
     Example:
       - `retrieve_context()` - calls `get_rerank_provider()`, no change needed
       - `GET /debug/search` (`api/routes.py`) - falls back to LLMRerankProvider when
         RERANK_PROVIDER=none; check this still behaves sensibly -->


## Configuration
<!-- New or changed environment variables, with defaults. Config is entirely env-var driven
     (no config files) - also update README.md "Configuration" and `.env.docker` where relevant.
     Leave empty if this change adds no configuration. -->


## How to test
<!-- Exact commands. -->
```
python run_tests.py
# single test module
python -m unittest tests.test_retrieval_service
```
<!-- Add specific curl / manual verification steps as needed. -->

## Depends on
<!-- Issue numbers that must be merged first. -->

## Refs
<!-- Design docs, related issues. -->

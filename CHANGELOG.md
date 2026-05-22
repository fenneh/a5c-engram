# Changelog

## v0.1.0 — 2026-05-22

Initial release.

### Pipeline

- `Profile` API: `ingest`, `remember`, `recall`, `list`, `forget`.
- Four memory types: `fact`, `event`, `instruction`, `task`.
- Supersession via `fact_key` + `version` + `superseded_by` columns.
- Content-addressed message IDs (SHA-256, 128-bit truncated).

### Extraction

- Deterministic regex pre-pass: dates, numeric assignments,
  self-reference, instruction verbs, task verbs.
- LLM dual-pass (full chunks + overlapping detail windows) with dedup.
- 8-check verifier (length, type, source token-overlap, snake_case
  fact_key, etc.).
- Write-time augmentation: 3-5 LLM-generated search-query paraphrases
  per memory, indexed by FTS and folded into the embedding.

### Retrieval

- Six channels: FTS, fact-key, raw message, direct vector, **temporal**
  (deterministic), HyDE.
- Reciprocal Rank Fusion with per-channel weights (fact-key 2.0,
  temporal 1.5, fts/vector 1.0, hyde 0.9, raw 0.6).
- Temporal channel parses `yesterday`/`today`/`tomorrow`/`last week`/
  `last N minutes|hours|days|weeks` and bypasses the HyDE LLM call.

### Storage

- `SqliteStorage` default — single file, FTS5 + sqlite-vec, WAL mode.
- `StorageAdapter` Protocol so you can plug in Postgres/etc.

### LLM and embedder

- `LLMAdapter` Protocol with `AnthropicLLM` (real) and `FakeLLM` (stub).
- `EmbedAdapter` Protocol with `BgeSmallEmbedder` (real, optional) and
  `FakeEmbedder` (stub).

### Server and UI

- FastAPI server mirrors the Python API at `/api/`.
- Built-in Jinja + HTMX inspection UI at `/ui`:
  profiles index, memory list with filters, memory detail with
  supersession chain, recall playground showing per-channel hits before
  fusion, ingest-session log.

### Tests and CI

- 83 tests, no network required, 3s on a laptop.
- GitHub Actions: ruff check + pytest on Python 3.11 and 3.12.
- `py.typed` marker shipped for downstream type checkers.

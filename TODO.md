# TODO

Known gaps in v0. Roughly ordered by what most needs doing first.

## Test coverage

70 tests cover the public API and the inspection UI. Still uncovered:

- `BgeSmallEmbedder` and `AnthropicLLM` integration tests — both need
  network/model downloads and are skipped from the default suite. The
  `examples/real_*_demo.py` scripts exercise them by hand.
- Concurrency on the FastAPI singleton `_profiles` dict — fine for
  single-process deployments, untested under multi-worker uvicorn.
- DB migrations / schema upgrades — there's nothing to migrate yet, but
  the moment we add a column we'll need a story.

## Features

- `expires_at` — the column exists but nothing reads it. Add a sweeper
  that removes expired memories, or filter them at recall time.
- LLM-driven query analysis — currently `_candidate_factkeys` derives
  factkeys from query tokens by hand. An LLM pass would produce better
  candidates.
- Temporal queries — bypass the LLM with regex + arithmetic for
  "yesterday / last week" style questions.
- Postgres adapter — `StorageAdapter` protocol is there, no implementation.
- Export / import — JSONL round-trip via CLI.

## Known issues

- The FTS5 sanitiser is conservative: each token is quoted as a literal
  phrase, so multi-word phrase queries don't preserve order. Good enough
  for v0, worth revisiting if recall quality suffers in practice.

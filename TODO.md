# TODO

## Test coverage

83 tests cover the public API, inspection UI, and the deterministic
temporal/paraphrase paths. Still uncovered:

- `BgeSmallEmbedder` and `AnthropicLLM` integration tests — both need
  network/model downloads and are skipped from the default suite. The
  `examples/real_*_demo.py` scripts exercise them by hand.
- Concurrency on the FastAPI singleton `_profiles` dict — fine for
  single-process deployments, untested under multi-worker uvicorn.
- DB migrations — there's no migration story yet. The moment we add a
  column to a deployed table we'll need one.

## Features

- `expires_at` — the column exists but nothing reads it. Add a sweeper
  that removes expired memories, or filter them at recall time.
- More temporal phrases — currently `yesterday`/`today`/`tomorrow`/
  `last week`/`last N units`. No support for "two months ago", named
  weekdays, or absolute dates as deltas.
- LLM-driven factkey candidates — `_candidate_factkeys` derives factkeys
  from query tokens by hand. An LLM pass would produce better candidates.
- Postgres adapter — `StorageAdapter` protocol is there, no impl.
- Export / import — JSONL CLI round-trip.

## Known issues

- The FTS5 sanitiser is conservative: each token is quoted as a literal
  phrase, so multi-word phrase queries don't preserve order. Good enough
  for v0, worth revisiting if recall quality suffers in practice.

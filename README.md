# a5c-engram

Self-hostable, extraction-first agent memory. Cloudflare's
[Agent Memory](https://blog.cloudflare.com/introducing-agent-memory/) as a
library you can drop into any Python agent — no Workers, no Vectorize lock-in,
no waitlist.

```python
from a5c_engram import Profile

p = Profile.open("my-agent")
p.ingest(messages, session_id="2026-05-22-conversation")
p.remember("the project uses GraphQL", type="fact", topic="api_style")

answer = p.recall("what API style do we use?")
# → "the project uses GraphQL" (newest version, supersedes older)
```

## Why

Naive vector-search memory accumulates duplicates, drifts on supersession,
and embeds raw chat (not knowledge). a5c-engram does what Cloudflare's
Agent Memory does:

1. **Extract**, don't embed — utterances become classified Facts / Events /
   Instructions / Tasks with verification.
2. **Supersede**, don't append — new facts on the same topic replace old ones
   via versioned fact-keys.
3. **Five-channel retrieval** — FTS, fact-key, raw, vector, HyDE — fused with
   Reciprocal Rank Fusion. No single channel covers every query shape.

## What's different

| | Cloudflare Agent Memory | mem0 / letta | **a5c-engram** |
|---|---|---|---|
| Self-host | ❌ | ✅ | ✅ |
| Pluggable storage | ❌ (Vectorize) | partial | ✅ (SQLite, Postgres, ...) |
| Pluggable LLM | ❌ (Workers AI) | partial | ✅ (Anthropic, OpenAI, Ollama) |
| API parity with Cloudflare | n/a | ❌ | ✅ |
| Built-in inspection UI | ❌ | ❌ | ✅ |
| Deterministic-first extraction | ❌ | ❌ | ✅ |
| Supersession chains as a column | ✅ | ❌ | ✅ |

## Install

```bash
uv add a5c-engram                     # core
uv add 'a5c-engram[embed,anthropic]'  # +local embedder, +Anthropic LLM
```

## Quickstart

```python
from a5c_engram import Profile

p = Profile.open("atlas")
p.ingest([
    {"role": "user", "content": "Atlas, we're rebuilding the v3.6 model tomorrow."},
    {"role": "assistant", "content": "Noted — v3.6 retrain on 2026-05-23."},
], session_id="standup-2026-05-22")

print(p.recall("when is the v3.6 retrain?"))
```

## HTTP server + inspection UI

```bash
uv run python -m a5c_engram.server
# → API at  http://localhost:8000/api/
# → UI  at  http://localhost:8000/
```

The UI gives you:
- A profile browser (counts by memory type).
- A memory detail view with the supersession chain rendered.
- A **recall playground** showing each retrieval channel's top-5 hits side
  by side, plus the RRF-fused final answer. This is the killer debug view
  — Cloudflare's product is a black box; here you can see exactly which
  channel surfaced which memory and why.

## Status

v0 (2026-05). Python 3.11+, MIT licensed. See [plan](docs/PLAN.md) and the
Cloudflare blog for the design lineage.

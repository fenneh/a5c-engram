# a5c-engram

Self-hostable, extraction-first agent memory for Python agents.

```python
from a5c_engram import Profile

p = Profile.open("my-agent")
p.remember("the project uses GraphQL", type="fact", topic="api_style")
p.remember("the project uses gRPC",   type="fact", topic="api_style")

print(p.recall("what api do we use?").hits[0].memory.content)
# → "the project uses gRPC"   (newest version, older one is preserved but
#                              marked superseded so it stops surfacing)
```

## Five-minute onboarding

### 1. Install

```bash
uv add a5c-engram                     # core
uv add 'a5c-engram[embed,anthropic]'  # +real local embedder, +Anthropic LLM
```

The bare install needs no API keys, no model downloads, and no extra
services — SQLite ships with Python. The optional extras only matter when
you want real semantic embeddings or real LLM-driven extraction.

### 2. Your first memory

```python
from a5c_engram import Profile

p = Profile.open("hello-engram")

# Direct write — when you, the agent author, decide what's worth remembering.
p.remember("The repo lives at ~/code/foo", type="fact", topic="repo_path")
p.remember("Always run tests before pushing.", type="instruction",
           topic="dev_policy")

# Bulk ingest from a conversation — the library extracts memories for you.
p.ingest([
    {"role": "user", "content": "My name is Alice."},
    {"role": "user", "content": "We deploy every Thursday at 14:00 UTC."},
], session_id="onboarding-chat")

# Recall — five-channel retrieval with reciprocal-rank fusion.
result = p.recall("when do we deploy?")
print(result.hits[0].memory.content)
```

By default `ingest()` runs a deterministic regex pre-pass (which already
catches dates, numeric facts, self-references, instruction verbs, task
phrases) plus a stub LLM. The pre-pass alone will catch the deploy time
above. To get the LLM extraction, see step 4.

### 3. Browse what got stored

```bash
uv run python -m a5c_engram.server
# → API at  http://localhost:8000/api/
# → UI  at  http://localhost:8000/ui
```

The UI is read-mostly and meant for debugging:

- **Profiles index** — counts by memory type, last ingest time.
- **Memories list** — filter by type, fact_key, or full-text query.
- **Memory detail** — content, source session, full supersession chain.
- **Recall playground** — type a query and see each retrieval channel's
  top hits side by side, plus the RRF-fused result. Useful when a recall
  returns the wrong thing and you want to know why.

Forget/delete buttons are off by default. Turn them on with
`A5C_ENGRAM_UI_WRITES=1`.

### 4. Plug in a real LLM

The default `Profile.open()` uses `FakeLLM`, which only recognises a
handful of canned patterns — good enough to confirm the install works,
useless on arbitrary prose. Real extraction needs a real LLM:

```python
from a5c_engram import Profile
from a5c_engram.llm.anthropic import AnthropicLLM

p = Profile.open("real", llm=AnthropicLLM())   # uses ANTHROPIC_API_KEY
```

Adding your own provider is a `LLMAdapter` protocol with four methods
(`extract`, `paraphrase`, `hyde`, `synthesise`). See
`src/a5c_engram/llm/anthropic.py` for a complete example you can copy.

### 5. Plug in a real embedder (optional)

The default `FakeEmbedder` is deterministic but hash-based — vector recall
returns essentially random results. For real semantic recall:

```python
from a5c_engram.embed.bge import BgeSmallEmbedder

p = Profile.open("real", embedder=BgeSmallEmbedder())
```

`BAAI/bge-small-en-v1.5` is ~120 MB, MIT-licensed, runs on CPU. Or write
your own `EmbedAdapter` — also a three-method protocol.

## How it works

1. **Extract, don't embed.** Utterances become classified atoms —
   `fact`, `event`, `instruction`, or `task` — with a verifier that rejects
   unsupported or malformed candidates. Two passes: a deterministic regex
   pre-pass (temporal, numeric, self-reference) that needs no LLM, then an
   LLM dual-pass over chunks and overlapping detail windows.
2. **Augment at write time.** Every non-task memory also gets 3-5
   LLM-generated search-query paraphrases stored alongside it. These are
   indexed by FTS and fed into the embedding, so the same memory is
   reachable by multiple lexical shapes.
3. **Supersede, don't append.** Facts and instructions carry a `fact_key`
   (a snake_case topic). When a new memory shares a `fact_key` with an
   existing one, the old one is marked `superseded_by` the new one and
   stops showing up in recall. The chain is preserved for inspection.
4. **Six-channel retrieval, fused with RRF.** Each `recall` runs in
   parallel across FTS5, fact-key exact lookup, raw message FTS, direct
   vector similarity, **temporal range** (deterministic, for
   "yesterday"/"last week"/"last N hours" queries), and HyDE (LLM
   hallucinated answer → embed → search). Results merge with Reciprocal
   Rank Fusion. Per-channel results are returned alongside the fused list
   — so you can see which channel found what. Temporal queries skip the
   HyDE LLM call because we already know the answer is a time window.

## Project shape

```
src/a5c_engram/
├── profile.py       # public API: Profile.open / ingest / remember / recall
├── schema.py        # Memory, MemoryType, Message dataclasses
├── extract/         # deterministic + LLM extraction, 8-check verifier
├── retrieve/        # the 5 channels and the RRF fuser
├── storage/         # SqliteStorage default; bring-your-own protocol
├── llm/             # AnthropicLLM + FakeLLM
├── embed/           # BgeSmallEmbedder + FakeEmbedder
├── server/          # FastAPI HTTP layer (mirrors the Python API)
└── ui/              # Jinja + HTMX inspection UI
```

Each of `storage`, `llm`, `embed` is a small Protocol — you can plug in
your own implementation without touching the rest.

## Examples

See [`examples/`](examples/) for runnable demos covering:

- A zero-setup round-trip (`chatbot_demo.py`)
- Wiring in a real LLM (`real_llm_demo.py`)
- Wiring in a real embedder (`real_embedder_demo.py`)
- Mounting into your own FastAPI app (`integrate_fastapi.py`)
- Backfilling from an existing event stream (`grdt_atlas_demo.py`)

## Status

v0, May 2026. Python 3.11+, MIT licensed. The library works end to end
with the stub LLM and stub embedder, and 70 tests cover the core paths.
See `TODO.md` for known gaps.

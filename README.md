# a5c-engram

Self-hostable, extraction-first agent memory for Python agents. Drop a
`Profile` into your agent, ingest conversation, recall it later — with
real semantic search and a built-in UI to inspect what got stored.

```python
from a5c_engram import Profile

p = Profile.open("my-agent")
p.remember("the project uses GraphQL", type="fact", topic="api_style")
p.remember("the project uses gRPC",    type="fact", topic="api_style")

print(p.recall("what api do we use?").hits[0].memory.content)
# → "the project uses gRPC"   (newest version; the older one is preserved
#                              but marked superseded so it stops surfacing)
```

## Install

```bash
uv add a5c-engram                     # core + local embedder
uv add 'a5c-engram[paid]'             # +OpenAI and Voyage embedders
uv add 'a5c-engram[anthropic]'        # +real Anthropic LLM for extraction
```

The bare install runs without any API key. First call downloads a
~120MB ONNX model (`BAAI/bge-small-en-v1.5`) to `~/.cache/fastembed`
and you're done.

## Quickstart

```python
from a5c_engram import Profile

p = Profile.open("hello")

p.ingest([
    {"role": "user", "content": "My name is Alice."},
    {"role": "user", "content": "We deploy every Thursday at 14:00 UTC."},
], session_id="onboarding")

p.remember("Never deploy on Fridays.", type="instruction", topic="deploy_policy")

print(p.recall("when do we deploy?").hits[0].memory.content)
```

## Inspection UI

```bash
uv run python -m a5c_engram.server
# → API at  http://localhost:8000/api/
# → UI  at  http://localhost:8000/ui
```

The UI is read-mostly and meant for debugging:

- Profiles index with counts per memory type.
- Memory browser, filter by type or fact_key, full-text search.
- Memory detail page with the supersession chain.
- **Recall playground** — type a query, see each retrieval channel's top
  hits side by side, then the fused result. Useful when recall returns
  the wrong thing and you want to know which channel surfaced it.

Forget buttons are off by default; set `A5C_ENGRAM_UI_WRITES=1` to enable.

## How it works

1. **Extract, don't embed.** Conversation becomes classified atoms —
   `fact`, `event`, `instruction`, `task` — with a verifier that drops
   unsupported candidates. A deterministic regex pre-pass catches dates,
   numeric facts, self-reference and instruction verbs without an LLM;
   the LLM dual-pass picks up everything else.
2. **Augment at write time.** Every non-task memory gets 3-5
   LLM-generated search-query paraphrases stored alongside it. Indexed
   by FTS and folded into the embedding so the same memory is reachable
   through multiple lexical and semantic shapes.
3. **Supersede, don't append.** Facts and instructions carry a
   `fact_key` (snake_case topic). New memories under the same topic mark
   the old ones `superseded_by` and stop surfacing — the chain is
   preserved for inspection.
4. **Six-channel retrieval with RRF.** Each `recall` runs FTS, fact-key
   exact lookup, raw message FTS, vector similarity, deterministic
   temporal range (for "yesterday" / "last week" / "last N hours"
   queries), and HyDE in parallel. Results merge with Reciprocal Rank
   Fusion. Temporal queries skip the HyDE LLM call.

## Embedders

The defaults work without an API key. Swap any of them by passing
`embedder=` to `Profile.open()` or setting `A5C_ENGRAM_EMBEDDER`.

| Embedder | When | Install |
|---|---|---|
| `FastEmbedder` (default) | local, CPU-only, ~120MB model | base install |
| `OpenAIEmbedder` | API, `text-embedding-3-small` default | `[paid]` + `OPENAI_API_KEY` |
| `VoyageEmbedder` | API, `voyage-3` default | `[paid]` + `VOYAGE_API_KEY` |
| `BgeSmallEmbedder` | sentence-transformers route | `[sentence-transformers]` |
| `FakeEmbedder` | hash-based stub for tests | base install |

LLMs are pluggable the same way: `LLMAdapter` is a four-method protocol
(`extract`, `paraphrase`, `hyde`, `synthesise`). `AnthropicLLM` ships in
the box; copy it to add OpenAI, Ollama, etc.

## Examples

Five runnable demos in [`examples/`](examples/):

- `chatbot_demo.py` — zero-setup end-to-end tour of all seven concepts
- `real_llm_demo.py` — wire in Anthropic for real extraction
- `real_embedder_demo.py` — Fake vs FastEmbedder vs OpenAI / Voyage on the same corpus
- `integrate_fastapi.py` — mount a Profile and the inspection UI inside your own FastAPI app
- `grdt_atlas_demo.py` — backfill from an existing Redis stream

# Examples

Each example here is a small, runnable demo of one integration shape.
Read them in roughly this order if you're new to a5c-engram.

| Example | What it teaches | Needs |
|---|---|---|
| [`chatbot_demo.py`](chatbot_demo.py) | The minimum viable round-trip: ingest, remember, supersede, recall. Uses the stub LLM and stub embedder, so it runs anywhere with zero setup. | nothing |
| [`real_llm_demo.py`](real_llm_demo.py) | Swap the stub LLM for Anthropic so `ingest()` actually extracts knowledge from free-form text. | `ANTHROPIC_API_KEY`, `a5c-engram[anthropic]` |
| [`real_embedder_demo.py`](real_embedder_demo.py) | Swap the hash-based stub embedder for a real local model (`BAAI/bge-small-en-v1.5`). Same query against both embedders so the difference is concrete. | `a5c-engram[embed]`, ~120 MB model download |
| [`integrate_fastapi.py`](integrate_fastapi.py) | Mount a Profile in your own FastAPI routes and serve the inspection UI at a sub-path of your existing app. | nothing |
| [`grdt_atlas_demo.py`](grdt_atlas_demo.py) | Read an existing Redis stream (in this case agent decisions from get-rich-or-die-tryin) and ingest it into a Profile. Template for plugging a5c-engram into an existing agent system. | `redis`, a running Redis with the streams |

## Run order for first-time users

```bash
# 1. Zero-setup demo — confirms the install works.
uv run python examples/chatbot_demo.py

# 2. Start the inspection UI and look at what got stored.
uv run python -m a5c_engram.server
# → open http://localhost:8000/ui

# 3. Plug in a real LLM. This is the moment ingest() becomes useful.
export ANTHROPIC_API_KEY=sk-ant-...
uv add 'a5c-engram[anthropic,embed]'
uv run python examples/real_llm_demo.py

# 4. See the embedder difference.
uv run python examples/real_embedder_demo.py

# 5. Adapt integrate_fastapi.py to your own app.
```

## Three common gotchas

1. **The default LLM is a stub.** `Profile.open("x")` with no `llm=` argument
   uses `FakeLLM`, which only recognises a few canned patterns. Ingest will
   look mostly empty until you wire in a real LLM (see `real_llm_demo.py`).
2. **The default embedder is hash-based.** Vector recall ranks essentially at
   random. Use `BgeSmallEmbedder` or your own (see `real_embedder_demo.py`).
3. **The default DB path is `~/.a5c-engram/engram.db`.** All profiles share
   one SQLite file by default. Pass `db_path=` to `Profile.open` for an
   isolated DB per app or per test.

"""Mount a5c-engram inside your own FastAPI app.

This shows the two integration shapes:
  1. Use a Profile in your own routes (e.g. an /ask endpoint that recalls).
  2. Mount the inspection UI at a sub-path so you can debug memories
     without running a second server.

Run:
    uv run uvicorn examples.integrate_fastapi:app --reload --port 8001
    curl http://localhost:8001/ask?q=what+do+we+use
    open http://localhost:8001/memory-ui
"""

from __future__ import annotations

from fastapi import FastAPI

from a5c_engram import Profile
from a5c_engram.ui.routes import mount_ui

# Your existing FastAPI app.
app = FastAPI(title="my-agent-with-memory")

# A single shared Profile. In a real app you'd open one per user or agent
# (Profile.open(f"user-{user_id}")) and look them up by name.
PROFILE = Profile.open("my-agent", db_path="/tmp/a5c_integrate_demo.db")

# Seed it once so the demo has something to recall against.
if not PROFILE.list(limit=1):
    PROFILE.remember("We use Postgres in production.", type="fact", topic="database")
    PROFILE.remember(
        "Always run migrations during the Sunday maintenance window.",
        type="instruction",
        topic="deploy_policy",
    )


@app.get("/ask")
def ask(q: str):
    """A simple recall endpoint backed by a5c-engram."""
    result = PROFILE.recall(q, k=5, use_hyde=False)
    return {
        "query": q,
        "top_hits": [{"content": h.memory.content, "channel": h.channel} for h in result.hits],
    }


@app.post("/note")
def note(content: str, topic: str | None = None):
    """An endpoint that lets users add a fact to memory."""
    m = PROFILE.remember(content, type="fact", topic=topic)
    return {"id": m.id}


# Mount the inspection UI at /memory-ui. Useful for debugging during dev.
# Pass a callable that resolves a profile by name — here we always return
# the shared singleton.
mount_ui(app, get_profile=lambda name: PROFILE)

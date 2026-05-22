from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from a5c_engram.profile import Profile
from a5c_engram.schema import MemoryType
from a5c_engram.ui.routes import mount_ui

_profiles: dict[str, Profile] = {}


def _get(name: str) -> Profile:
    p = _profiles.get(name)
    if p is None:
        p = Profile.open(name)
        _profiles[name] = p
    return p


class IngestBody(BaseModel):
    messages: list[dict[str, Any]]
    session_id: str | None = None
    use_llm: bool = True


class RememberBody(BaseModel):
    content: str
    type: str = "fact"
    topic: str | None = None
    session_id: str | None = None


class RecallBody(BaseModel):
    query: str
    k: int = 10
    use_hyde: bool = True
    synthesise: bool = False


app = FastAPI(title="a5c-engram", version="0.1.0")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.post("/api/profiles/{name}/ingest")
def api_ingest(name: str, body: IngestBody):
    p = _get(name)
    mems = p.ingest(body.messages, session_id=body.session_id, use_llm=body.use_llm)
    return {"committed": [m.to_dict() for m in mems]}


@app.post("/api/profiles/{name}/remember")
def api_remember(name: str, body: RememberBody):
    p = _get(name)
    try:
        m = p.remember(body.content, type=body.type, topic=body.topic, session_id=body.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return m.to_dict()


@app.post("/api/profiles/{name}/recall")
def api_recall(name: str, body: RecallBody):
    p = _get(name)
    result = p.recall(body.query, k=body.k, use_hyde=body.use_hyde, synthesise=body.synthesise)
    return {
        "query": result.query,
        "synthesis": result.synthesis,
        "hits": [
            {
                "memory": h.memory.to_dict(),
                "channel": h.channel,
                "rank": h.rank,
                "score": h.score,
                "channel_count": h.channel_count,
            }
            for h in result.hits
        ],
        "by_channel": {
            ch: [
                {
                    "memory": h.memory.to_dict(),
                    "channel": h.channel,
                    "rank": h.rank,
                    "score": h.score,
                }
                for h in hits
            ]
            for ch, hits in result.by_channel.items()
        },
    }


@app.get("/api/profiles/{name}/memories")
def api_list(
    name: str,
    type: str | None = None,
    topic: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    p = _get(name)
    t = MemoryType(type) if type else None
    mems = p.list(type=t, topic=topic, limit=limit, offset=offset)
    return [m.to_dict() for m in mems]


@app.get("/api/profiles/{name}/memories/{mem_id}")
def api_memory(name: str, mem_id: str):
    p = _get(name)
    m = p.storage.get_memory(mem_id)
    if m is None:
        raise HTTPException(status_code=404, detail="not found")
    chain = p.storage.supersession_chain(mem_id)
    return {"memory": m.to_dict(), "chain": [c.to_dict() for c in chain]}


@app.delete("/api/profiles/{name}/memories/{mem_id}")
def api_forget(name: str, mem_id: str):
    p = _get(name)
    ok = p.forget(mem_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"forgot": mem_id}


@app.get("/api/profiles")
def api_profiles():
    # Aggregate across every profile that any storage knows about. For now we
    # rely on the default singleton storage created by Profile.open.
    if not _profiles:
        _get("default")
    any_storage = next(iter(_profiles.values())).storage
    return any_storage.list_profiles()


mount_ui(app, get_profile=_get)


def main() -> None:
    import uvicorn

    host = os.environ.get("A5C_ENGRAM_HOST", "127.0.0.1")
    port = int(os.environ.get("A5C_ENGRAM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

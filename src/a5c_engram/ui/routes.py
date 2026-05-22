from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

UI_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(UI_DIR / "templates"))


def mount_ui(app: FastAPI, *, get_profile: Callable[[str], object]) -> None:
    """Mount HTML routes + static dir onto an existing FastAPI app."""
    app.mount(
        "/ui/static",
        StaticFiles(directory=str(UI_DIR / "static")),
        name="ui-static",
    )

    writes_enabled = os.environ.get("A5C_ENGRAM_UI_WRITES") == "1"
    TEMPLATES.env.globals["writes_enabled"] = writes_enabled
    TEMPLATES.env.globals["fmt_ts"] = _fmt_ts

    @app.get("/ui", response_class=HTMLResponse)
    def ui_index(request: Request):
        if not _profiles_registered(app):
            get_profile("default")
        any_storage = _any_storage(app, get_profile)
        profiles = any_storage.list_profiles() if any_storage else []
        return TEMPLATES.TemplateResponse(
            request, "profiles.html", {"profiles": profiles}
        )

    @app.get("/ui/p/{name}", response_class=HTMLResponse)
    def ui_profile(
        request: Request,
        name: str,
        type: str | None = None,
        topic: str | None = None,
        q: str | None = None,
    ):
        p = get_profile(name)
        if q:
            result = p.recall(q, k=20, use_hyde=False)
            mems = [h.memory for h in result.hits]
        else:
            mems = p.list(type=type, topic=topic, limit=200)
        return TEMPLATES.TemplateResponse(
            request,
            "memories.html",
            {
                "profile": name,
                "memories": mems,
                "filter_type": type,
                "filter_topic": topic,
                "filter_q": q or "",
            },
        )

    @app.get("/ui/p/{name}/m/{mem_id}", response_class=HTMLResponse)
    def ui_memory(request: Request, name: str, mem_id: str):
        p = get_profile(name)
        m = p.storage.get_memory(mem_id)
        if m is None:
            raise HTTPException(status_code=404)
        chain = p.storage.supersession_chain(mem_id)
        return TEMPLATES.TemplateResponse(
            request,
            "memory.html",
            {"profile": name, "memory": m, "chain": chain},
        )

    @app.get("/ui/p/{name}/recall", response_class=HTMLResponse)
    def ui_recall(request: Request, name: str, q: str | None = None, synth: int = 0):
        p = get_profile(name)
        result = None
        if q:
            result = p.recall(q, k=10, use_hyde=True, synthesise=bool(synth))
        return TEMPLATES.TemplateResponse(
            request,
            "recall.html",
            {
                "profile": name,
                "q": q or "",
                "synth": bool(synth),
                "result": result,
            },
        )

    @app.get("/ui/p/{name}/sessions", response_class=HTMLResponse)
    def ui_sessions(request: Request, name: str):
        p = get_profile(name)
        sessions = p.storage.list_sessions(name)
        return TEMPLATES.TemplateResponse(
            request,
            "sessions.html",
            {"profile": name, "sessions": sessions},
        )


def _profiles_registered(app: FastAPI) -> bool:
    # Best-effort: check the module-level singleton.
    from a5c_engram.server.app import _profiles
    return bool(_profiles)


def _any_storage(app: FastAPI, get_profile):
    from a5c_engram.server.app import _profiles
    if _profiles:
        return next(iter(_profiles.values())).storage
    p = get_profile("default")
    return p.storage


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

"""Confirm the pipeline degrades gracefully when things go wrong."""

import pytest
from fastapi.testclient import TestClient

from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.profile import Profile
from a5c_engram.storage.sqlite import SqliteStorage
from tests.helpers import MalformedLLM


def test_malformed_llm_garbage_does_not_crash(tmp_path):
    storage = SqliteStorage(path=tmp_path / "engram.db")
    storage.init()
    p = Profile(
        "t",
        storage=storage,
        embedder=FakeEmbedder(),
        llm=MalformedLLM(mode="empty_content"),
    )
    # MalformedLLM returns empty-content candidates; verifier should reject.
    mems = p.ingest(
        [{"role": "user", "content": "anything"}],
        session_id="s1",
        use_llm=True,
    )
    # The deterministic extractor may or may not match "anything"; what we
    # care about is no exception was raised and no junk got through.
    for m in mems:
        assert m.content, "empty-content memories must be rejected"


def test_malformed_llm_wrong_type_filtered(tmp_path):
    storage = SqliteStorage(path=tmp_path / "engram.db")
    storage.init()
    p = Profile(
        "t",
        storage=storage,
        embedder=FakeEmbedder(),
        llm=MalformedLLM(mode="wrong_type"),
    )
    mems = p.ingest(
        [{"role": "user", "content": "anything"}],
        session_id="s1",
        use_llm=True,
    )
    # Verifier must drop wrong-type candidates.
    assert all(m.type.value in {"fact", "event", "instruction", "task"} for m in mems)


def test_empty_query_returns_empty_recall(profile):
    result = profile.recall("")
    assert result.hits == []
    # All channels still present but empty.
    for ch in ["fts", "factkey", "raw", "vector", "hyde"]:
        assert ch in result.by_channel


def test_recall_on_empty_profile_returns_empty(profile):
    result = profile.recall("anything")
    assert result.hits == []


def test_remember_invalid_type_raises(profile):
    with pytest.raises(ValueError):
        profile.remember("x", type="bogus")


def test_api_404_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("A5C_ENGRAM_DB", str(tmp_path / "engram.db"))
    from a5c_engram.server import app as srv_module

    srv_module._profiles.clear()
    c = TestClient(srv_module.app)
    r = c.get("/api/profiles/never-seen/memories/no-such-id")
    assert r.status_code == 404

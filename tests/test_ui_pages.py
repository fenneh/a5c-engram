"""Smoke tests for the UI pages other than index + recall."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("A5C_ENGRAM_DB", str(tmp_path / "engram.db"))
    from a5c_engram.server import app as srv_module
    srv_module._profiles.clear()
    return TestClient(srv_module.app)


def _seed(client):
    client.post(
        "/api/profiles/atlas/remember",
        json={"content": "uses GraphQL", "type": "fact", "topic": "api_style"},
    )
    client.post(
        "/api/profiles/atlas/remember",
        json={"content": "uses gRPC", "type": "fact", "topic": "api_style"},
    )
    client.post(
        "/api/profiles/atlas/ingest",
        json={
            "messages": [{"role": "user", "content": "morning"}],
            "session_id": "standup-1",
            "use_llm": False,
        },
    )


def test_ui_memories_list(client):
    _seed(client)
    r = client.get("/ui/p/atlas")
    assert r.status_code == 200
    assert "uses gRPC" in r.text
    assert "api_style" in r.text


def test_ui_memories_filter_by_type(client):
    _seed(client)
    r = client.get("/ui/p/atlas", params={"type": "fact"})
    assert r.status_code == 200
    # Pages render type filter chip / selected option.
    assert "fact" in r.text


def test_ui_memory_detail_renders_chain(client):
    _seed(client)
    mems = client.get(
        "/api/profiles/atlas/memories", params={"topic": "api_style"}
    ).json()
    target = next(m for m in mems if m["content"] == "uses gRPC")
    r = client.get(f"/ui/p/atlas/m/{target['id']}")
    assert r.status_code == 200
    assert "uses gRPC" in r.text
    # The supersession chain section appears when chain length > 1.
    assert "uses GraphQL" in r.text
    assert "Supersession chain" in r.text


def test_ui_memory_detail_404_on_missing(client):
    r = client.get("/ui/p/atlas/m/does-not-exist")
    assert r.status_code == 404


def test_ui_sessions_page(client):
    _seed(client)
    r = client.get("/ui/p/atlas/sessions")
    assert r.status_code == 200
    assert "standup-1" in r.text


def test_ui_recall_no_query_still_renders(client):
    _seed(client)
    r = client.get("/ui/p/atlas/recall")
    assert r.status_code == 200
    # No query → no fused table, but the form must be there.
    assert "Recall" in r.text

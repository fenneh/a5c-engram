"""Coverage for HTTP routes other than the API-roundtrip in
test_server_and_ui.py: profiles index, list, memory detail (with chain),
forget."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("A5C_ENGRAM_DB", str(tmp_path / "engram.db"))
    from a5c_engram.server import app as srv_module
    srv_module._profiles.clear()
    return TestClient(srv_module.app)


def _seed(client, profile="atlas"):
    client.post(
        f"/api/profiles/{profile}/remember",
        json={"content": "uses GraphQL", "type": "fact", "topic": "api_style"},
    )
    client.post(
        f"/api/profiles/{profile}/remember",
        json={"content": "uses gRPC", "type": "fact", "topic": "api_style"},
    )
    client.post(
        f"/api/profiles/{profile}/remember",
        json={"content": "deployed v1 on 2026-05-21", "type": "event"},
    )


def test_profiles_index_returns_counts(client):
    _seed(client)
    r = client.get("/api/profiles")
    assert r.status_code == 200
    profiles = r.json()
    atlas = next(p for p in profiles if p["name"] == "atlas")
    assert atlas["counts"]["fact"] >= 1
    assert atlas["counts"]["event"] == 1


def test_list_memories_filters_by_type(client):
    _seed(client)
    r = client.get("/api/profiles/atlas/memories", params={"type": "event"})
    assert r.status_code == 200
    mems = r.json()
    assert mems
    assert all(m["type"] == "event" for m in mems)


def test_list_memories_filters_by_topic(client):
    _seed(client)
    r = client.get("/api/profiles/atlas/memories", params={"topic": "api_style"})
    assert r.status_code == 200
    mems = r.json()
    # supersession means only the latest under fact_key is unsuperseded,
    # but list_memories does not filter on superseded_by — returns both.
    assert any(m["content"] == "uses gRPC" for m in mems)


def test_memory_detail_returns_chain(client):
    _seed(client)
    mems = client.get("/api/profiles/atlas/memories", params={"topic": "api_style"}).json()
    latest = next(m for m in mems if m["content"] == "uses gRPC")
    r = client.get(f"/api/profiles/atlas/memories/{latest['id']}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["memory"]["content"] == "uses gRPC"
    assert len(payload["chain"]) == 2
    assert [c["content"] for c in payload["chain"]] == ["uses GraphQL", "uses gRPC"]


def test_memory_detail_404_on_missing(client):
    r = client.get("/api/profiles/atlas/memories/does-not-exist")
    assert r.status_code == 404


def test_delete_forget(client):
    _seed(client)
    mems = client.get("/api/profiles/atlas/memories", params={"topic": "api_style"}).json()
    target = mems[0]
    r = client.delete(f"/api/profiles/atlas/memories/{target['id']}")
    assert r.status_code == 200
    assert r.json()["forgot"] == target["id"]
    assert client.get(f"/api/profiles/atlas/memories/{target['id']}").status_code == 404


def test_delete_forget_404_on_missing(client):
    r = client.delete("/api/profiles/atlas/memories/does-not-exist")
    assert r.status_code == 404


def test_ingest_endpoint(client):
    r = client.post(
        "/api/profiles/atlas/ingest",
        json={
            "messages": [
                {"role": "user", "content": "Always run tests before merging."}
            ],
            "session_id": "s1",
            "use_llm": False,
        },
    )
    assert r.status_code == 200
    committed = r.json()["committed"]
    assert committed
    assert any(m["type"] == "instruction" for m in committed)

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("A5C_ENGRAM_DB", str(tmp_path / "engram.db"))
    from a5c_engram.server import app as srv_module
    srv_module._profiles.clear()
    return TestClient(srv_module.app)


def test_api_remember_recall_roundtrip(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post(
        "/api/profiles/atlas/remember",
        json={"content": "The project uses GraphQL.", "type": "fact", "topic": "api_style"},
    )
    assert r.status_code == 200, r.text
    r = c.post("/api/profiles/atlas/recall", json={"query": "what api do we use?"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["hits"], payload
    assert any("GraphQL" in h["memory"]["content"] for h in payload["hits"])


def test_ui_index_renders(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/ui")
    assert r.status_code == 200
    assert "a5c-engram" in r.text


def test_ui_recall_playground_renders(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post(
        "/api/profiles/atlas/remember",
        json={"content": "uses GraphQL", "type": "fact", "topic": "api_style"},
    )
    r = c.get("/ui/p/atlas/recall", params={"q": "api"})
    assert r.status_code == 200
    assert "Fused" in r.text
    assert "factkey" in r.text

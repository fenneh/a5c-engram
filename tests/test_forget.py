"""forget(id) must clean the memory out of every retrieval path. Raw
messages are deliberately *not* deleted — they're the ingest log."""


def test_forget_removes_from_get(profile):
    m = profile.remember("uses GraphQL", type="fact", topic="api_style")
    assert profile.forget(m.id) is True
    assert profile.storage.get_memory(m.id) is None


def test_forget_idempotent_returns_false_second_time(profile):
    m = profile.remember("ephemeral", type="task")
    assert profile.forget(m.id) is True
    assert profile.forget(m.id) is False


def test_forget_removes_from_fts(profile):
    m = profile.remember("uses GraphQL", type="fact", topic="api_style")
    assert profile.storage.search_fts("test", "GraphQL")
    profile.forget(m.id)
    assert profile.storage.search_fts("test", "GraphQL") == []


def test_forget_removes_from_vector(profile):
    m = profile.remember("uses GraphQL", type="fact", topic="api_style")
    qvec = profile.embedder.embed("GraphQL")
    before = profile.storage.search_vector("test", qvec, k=5)
    assert any(h.id == m.id for h in before)
    profile.forget(m.id)
    after = profile.storage.search_vector("test", qvec, k=5)
    assert all(h.id != m.id for h in after)


def test_forget_does_not_touch_raw_messages(profile):
    profile.ingest(
        [{"role": "user", "content": "We use GraphQL for the API."}],
        session_id="s1",
        use_llm=False,
    )
    mems = profile.list()
    for m in mems:
        profile.forget(m.id)
    # Raw messages are the ingest log — they survive memory deletion.
    raw_hits = profile.storage.search_raw("test", "GraphQL")
    assert raw_hits, "raw messages must survive forget()"


def test_forget_unknown_id(profile):
    assert profile.forget("does-not-exist") is False

def test_remember_then_recall(profile):
    profile.remember("The project uses GraphQL.", type="fact", topic="api_style")
    result = profile.recall("what api do we use?")
    assert result.hits, "expected at least one hit"
    assert any("GraphQL" in h.memory.content for h in result.hits)


def test_supersession_via_remember(profile):
    profile.remember("uses GraphQL", type="fact", topic="api_style")
    profile.remember("uses gRPC", type="fact", topic="api_style")
    latest = profile.storage.latest_for_factkey("test", "api_style")
    assert latest is not None and latest.content == "uses gRPC"
    assert latest.version == 2


def test_ingest_deterministic_only(profile):
    profile.ingest(
        [
            {"role": "user", "content": "My name is Alice Walker."},
            {"role": "user", "content": "Always run tests before merging."},
        ],
        session_id="s1",
        use_llm=False,
    )
    mems = profile.list()
    types = {m.type.value for m in mems}
    assert "fact" in types
    assert "instruction" in types


def test_recall_returns_per_channel_breakdown(profile):
    profile.remember("uses GraphQL", type="fact", topic="api_style")
    result = profile.recall("api style")
    assert "fts" in result.by_channel
    assert "factkey" in result.by_channel
    assert "vector" in result.by_channel
    assert "hyde" in result.by_channel
    assert "raw" in result.by_channel


def test_forget(profile):
    m = profile.remember("ephemeral", type="task")
    assert profile.forget(m.id)
    assert profile.storage.get_memory(m.id) is None

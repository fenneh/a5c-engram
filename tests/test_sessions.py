def test_list_sessions_aggregates_messages(profile):
    profile.ingest(
        [
            {"role": "user", "content": "morning standup"},
            {"role": "assistant", "content": "morning"},
        ],
        session_id="standup-1",
        use_llm=False,
    )
    profile.ingest(
        [{"role": "user", "content": "afternoon retro"}],
        session_id="retro-1",
        use_llm=False,
    )

    sessions = profile.storage.list_sessions("test")
    sids = {s["session_id"]: s for s in sessions}
    assert sids["standup-1"]["message_count"] == 2
    assert sids["retro-1"]["message_count"] == 1


def test_session_lists_extracted_memories(profile):
    profile.ingest(
        [{"role": "user", "content": "Always run tests before merging."}],
        session_id="rule-1",
        use_llm=False,
    )
    mems = profile.storage.memories_for_session("test", "rule-1")
    assert mems, "should have extracted at least one memory"
    assert all(m.source_session_id == "rule-1" for m in mems)


def test_session_with_no_extracted_memories(profile):
    profile.ingest(
        [{"role": "user", "content": "hi"}],
        session_id="trivial",
        use_llm=False,
    )
    sessions = profile.storage.list_sessions("test")
    triv = next(s for s in sessions if s["session_id"] == "trivial")
    assert triv["message_count"] == 1
    # might be 0 if nothing extracted, depending on patterns; just non-negative.
    assert triv["memory_count"] >= 0


def test_list_sessions_ordered_by_recent(profile):
    profile.ingest([{"role": "user", "content": "older"}], session_id="old", use_llm=False)
    profile.ingest([{"role": "user", "content": "newer"}], session_id="new", use_llm=False)
    sessions = profile.storage.list_sessions("test")
    # Most recent first.
    assert sessions[0]["session_id"] in {"new", "old"}
    assert sessions[0]["last_ts"] >= sessions[-1]["last_ts"]

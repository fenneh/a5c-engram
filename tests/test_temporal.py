"""Deterministic temporal channel: 'yesterday'/'last week'/'last 24 hours'
parses to a time window, recall returns memories within it, HyDE is skipped."""

import time

from a5c_engram.extract.deterministic import parse_temporal_range
from tests.helpers import CountingLLM


def test_parse_yesterday():
    now = time.time()
    r = parse_temporal_range("what did I say yesterday?", now=now)
    assert r is not None
    start, end = r
    assert start < now - 23 * 3600
    assert end <= now


def test_parse_today():
    r = parse_temporal_range("anything today?")
    assert r is not None


def test_parse_last_n_hours():
    now = time.time()
    r = parse_temporal_range("show me notes from the last 24 hours", now=now)
    assert r is not None
    start, end = r
    assert end - start >= 24 * 3600 - 1
    assert end - start <= 24 * 3600 + 5


def test_parse_last_n_days():
    now = time.time()
    r = parse_temporal_range("ingest from the last 7 days", now=now)
    assert r is not None
    start, end = r
    assert (end - start) // 86400 >= 6  # 7 days minus the +1s epsilon


def test_parse_last_week():
    r = parse_temporal_range("last week's deploys")
    assert r is not None


def test_parse_no_temporal():
    assert parse_temporal_range("what API do we use?") is None
    assert parse_temporal_range("any GraphQL facts") is None


def test_temporal_channel_in_recall(profile):
    old_mem = profile.remember("old fact", type="fact")
    # Backdate it 2 days.
    conn = profile.storage._connect()
    conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (time.time() - 2 * 86400, old_mem.id),
    )
    conn.commit()

    fresh = profile.remember("fresh fact", type="fact")

    result = profile.recall("what did I say in the last 24 hours?")
    temporal_ids = [h.memory.id for h in result.by_channel["temporal"]]
    assert fresh.id in temporal_ids
    assert old_mem.id not in temporal_ids


def test_temporal_bypasses_hyde(tmp_path):
    from a5c_engram.embed.base import FakeEmbedder
    from a5c_engram.llm.fake import FakeLLM
    from a5c_engram.profile import Profile
    from a5c_engram.storage.sqlite import SqliteStorage

    storage = SqliteStorage(path=tmp_path / "engram.db")
    storage.init()
    counter = CountingLLM(FakeLLM())
    p = Profile("t", storage=storage, embedder=FakeEmbedder(), llm=counter)
    p.remember("yesterday I shipped a thing", type="event")

    counter.hyde_calls = 0
    p.recall("what did I say yesterday?")
    assert counter.hyde_calls == 0, "temporal queries must not invoke HyDE"


def test_temporal_does_not_run_on_normal_query(profile):
    result = profile.recall("what API do we use?")
    assert result.by_channel["temporal"] == []

"""Profile.remember and Memory.new must be idempotent — re-running a
backfill that calls remember() twice with the same content should not
grow the row count.

Two paths exercised:
- no fact_key: dedup via deterministic Memory.id under storage's
  INSERT OR REPLACE
- with fact_key: the supersede chain no-ops when the latest version's
  content already matches
"""

from __future__ import annotations

from a5c_engram.schema import Memory, MemoryType


def test_memory_new_id_is_deterministic():
    a = Memory.new(profile="p", type=MemoryType.FACT, content="hello")
    b = Memory.new(profile="p", type=MemoryType.FACT, content="hello")
    assert a.id == b.id


def test_memory_new_id_changes_with_content():
    a = Memory.new(profile="p", type=MemoryType.FACT, content="hello")
    b = Memory.new(profile="p", type=MemoryType.FACT, content="hello world")
    assert a.id != b.id


def test_memory_new_id_changes_with_session():
    a = Memory.new(profile="p", type=MemoryType.FACT, content="x", source_session_id="s1")
    b = Memory.new(profile="p", type=MemoryType.FACT, content="x", source_session_id="s2")
    assert a.id != b.id


def test_remember_twice_no_factkey_collapses(profile):
    profile.remember("hello world", type=MemoryType.EVENT)
    profile.remember("hello world", type=MemoryType.EVENT)
    assert len(profile.list(limit=100)) == 1


def test_remember_twice_same_factkey_same_content_no_ops(profile):
    a = profile.remember("uses GraphQL", type=MemoryType.FACT, topic="api")
    b = profile.remember("uses GraphQL", type=MemoryType.FACT, topic="api")
    # Same memory returned, no new version.
    assert a.id == b.id
    assert b.version == 1
    chain = profile.storage.supersession_chain(b.id)
    assert len(chain) == 1


def test_remember_factkey_different_content_does_supersede(profile):
    a = profile.remember("uses GraphQL", type=MemoryType.FACT, topic="api")
    b = profile.remember("uses gRPC", type=MemoryType.FACT, topic="api")
    assert b.version == 2
    chain = profile.storage.supersession_chain(b.id)
    assert [m.content for m in chain] == ["uses GraphQL", "uses gRPC"]
    # First version still exists, marked superseded.
    a_now = profile.storage.get_memory(a.id)
    assert a_now is not None and a_now.superseded_by == b.id

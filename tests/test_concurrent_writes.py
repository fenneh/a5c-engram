"""Concurrent writes must not crash or drop memories.

SqliteStorage runs WAL + synchronous=NORMAL and shares one connection
across threads with check_same_thread=False. Python's sqlite3 module
serialises writes on the same connection, so this test is the contract:
N threads each call remember() repeatedly, and the final memory count
matches the number of distinct contents written.
"""

from __future__ import annotations

import threading

from a5c_engram.schema import MemoryType


def test_concurrent_remember_no_lost_writes(profile):
    threads = 8
    per_thread = 25
    barrier = threading.Barrier(threads)

    def worker(tid: int):
        barrier.wait()
        for i in range(per_thread):
            profile.remember(
                f"thread {tid} content number {i}",
                type=MemoryType.EVENT,
            )

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    rows = profile.list(limit=10_000)
    assert len(rows) == threads * per_thread


def test_concurrent_remember_same_content_collapses(profile):
    """Many threads writing identical content end up with one row, not N
    rows. Deterministic id + INSERT OR REPLACE is the contract.
    """
    threads = 6
    barrier = threading.Barrier(threads)

    def worker():
        barrier.wait()
        for _ in range(10):
            profile.remember("the one true memory", type=MemoryType.FACT)

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    rows = profile.list(limit=100)
    assert len(rows) == 1
    assert rows[0].content == "the one true memory"

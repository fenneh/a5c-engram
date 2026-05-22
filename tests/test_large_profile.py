"""Sanity-perf test at non-trivial profile sizes.

5000 entries is well above what the grdt agents have today (~400) but
well below what production-scale agent memory could reach (10k+). The
test exists to catch regressions where someone adds an N+1 query or
drops an index. Wallclock budgets are generous to stay stable on slow
CI runners.
"""

from __future__ import annotations

import time

from a5c_engram.schema import MemoryType


def test_5k_inserts_and_recall_under_budget(profile):
    start = time.perf_counter()
    for i in range(5000):
        profile.remember(
            f"entry {i} mentions widget {i % 50}",
            type=MemoryType.EVENT,
            session_id=f"s{i % 20}",
        )
    insert_s = time.perf_counter() - start
    # 5000 inserts including FTS index trigger should land well under
    # 30s on any sane box. Real runs locally are ~3-5s.
    assert insert_s < 30, f"5k inserts took {insert_s:.1f}s"

    start = time.perf_counter()
    result = profile.recall("widget 7", k=10, use_hyde=False)
    recall_s = time.perf_counter() - start
    # Recall is FTS-bounded; should be sub-100ms even at 5k rows.
    assert recall_s < 1.0, f"recall took {recall_s:.3f}s"
    assert len(result.hits) > 0


def test_list_at_scale_paginated_correctly(profile):
    for i in range(200):
        profile.remember(f"row {i}", type=MemoryType.FACT)
    page1 = profile.list(limit=50, offset=0)
    page2 = profile.list(limit=50, offset=50)
    assert len(page1) == 50
    assert len(page2) == 50
    # Pages must not overlap.
    ids1 = {m.id for m in page1}
    ids2 = {m.id for m in page2}
    assert ids1.isdisjoint(ids2)

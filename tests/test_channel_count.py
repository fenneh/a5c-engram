"""RecallHit.channel_count surfaces how many channels agreed on a hit.

The RRF score itself is rank-based and tightly clustered, so absolute
scores are not useful for thresholding. Channel count is the actual
"this is a strong hit" signal: if FTS + vector + factkey all agree,
that's a sharper match than any single-channel hit.
"""

from __future__ import annotations

from a5c_engram.retrieve.fuse import rrf_fuse
from a5c_engram.schema import Memory, MemoryType


def _mem(content: str) -> Memory:
    return Memory.new(profile="p", type=MemoryType.FACT, content=content)


def test_single_channel_hit_count_is_one():
    a = _mem("alpha")
    fused, _ = rrf_fuse({"fts": [a]})
    assert len(fused) == 1
    assert fused[0].channel_count == 1


def test_multi_channel_agreement_counted():
    a = _mem("alpha")
    b = _mem("beta")
    fused, _ = rrf_fuse(
        {
            "fts": [a, b],
            "vector": [a],
            "factkey": [a],
        }
    )
    by_id = {h.memory.id: h for h in fused}
    assert by_id[a.id].channel_count == 3  # all three channels surfaced a
    assert by_id[b.id].channel_count == 1  # only fts surfaced b


def test_channel_count_breaks_score_ties():
    """A lone factkey hit (weight 2.0) and two unit-weight hits (fts +
    vector at rank 1) produce numerically identical RRF scores:
    2/(60+1) == 1/(60+1) + 1/(60+1). That's exactly the case where the
    raw score gives the caller no signal and channel_count is the only
    way to tell the strong cross-channel match from the lone hit.
    """
    a = _mem("alpha")
    b = _mem("beta")
    fused, _ = rrf_fuse({"factkey": [a], "fts": [b], "vector": [b]})
    by_id = {h.memory.id: h for h in fused}
    assert by_id[a.id].score == by_id[b.id].score
    assert by_id[a.id].channel_count == 1
    assert by_id[b.id].channel_count == 2


def test_same_id_returned_twice_in_one_channel_counts_once():
    """Defensive: a buggy adapter that returns the same memory id twice
    in a single channel must not inflate channel_count.
    """
    a = _mem("alpha")
    fused, _ = rrf_fuse({"fts": [a, a]})
    assert fused[0].channel_count == 1

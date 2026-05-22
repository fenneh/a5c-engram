from a5c_engram.retrieve.fuse import rrf_fuse
from a5c_engram.schema import Memory, MemoryType


def _m(i: str) -> Memory:
    return Memory.new(profile="t", type=MemoryType.FACT, content=f"content {i}")


def test_rrf_fact_key_outranks_single_channel_top():
    """A memory hit by factkey alone should beat one ranked #1 by fts alone,
    because factkey weight is 2x."""
    a, b = _m("a"), _m("b")
    fused, _ = rrf_fuse({"factkey": [a], "fts": [b]})
    assert fused[0].memory.id == a.id


def test_rrf_aggregates_across_channels():
    """A memory appearing in multiple non-factkey channels beats one that
    only appears in a single one. (Factkey is intentionally weighted higher,
    so this asserts the among-equal-weight property.)"""
    a, b = _m("a"), _m("b")
    fused, _ = rrf_fuse({
        "fts": [b, a],     # b: 1.0/61, a: 1.0/62
        "vector": [a],     # a: 1.0/61
        "hyde": [a],       # a: 0.9/61
    })
    # b score:           1.0/61                                 = 0.01639
    # a score: 1.0/62 + 1.0/61 + 0.9/61                          = 0.04752
    assert fused[0].memory.id == a.id


def test_rrf_per_channel_returned():
    a = _m("a")
    fused, per = rrf_fuse({"fts": [a], "vector": [a]})
    assert per["fts"][0].memory.id == a.id
    assert per["vector"][0].memory.id == a.id
    assert per["fts"][0].rank == 1
    assert fused[0].channel == "fused"

from __future__ import annotations

from a5c_engram.schema import Memory, RecallHit

# Channel weight defaults — fact_key matches dominate because they are exact
# topic hits, not heuristics. Tuneable.
CHANNEL_WEIGHTS: dict[str, float] = {
    "factkey": 2.0,
    "temporal": 1.5,  # natural-language time queries are precise when they hit
    "fts": 1.0,
    "vector": 1.0,
    "hyde": 0.9,
    "raw": 0.6,
}

RRF_K = 60  # standard RRF constant


def rrf_fuse(
    by_channel: dict[str, list[Memory]],
    *,
    weights: dict[str, float] | None = None,
    k: int = RRF_K,
    top_n: int = 10,
) -> tuple[list[RecallHit], dict[str, list[RecallHit]]]:
    """Reciprocal Rank Fusion: score(d) = sum_c weight(c) / (k + rank_c(d)).

    Returns (fused top_n, per-channel hits with rank+score)."""
    weights = weights or CHANNEL_WEIGHTS

    scores: dict[str, float] = {}
    best_memory: dict[str, Memory] = {}
    per_channel: dict[str, list[RecallHit]] = {}

    for channel, mems in by_channel.items():
        w = weights.get(channel, 1.0)
        ch_hits: list[RecallHit] = []
        for rank, mem in enumerate(mems, start=1):
            rrf = w / (k + rank)
            scores[mem.id] = scores.get(mem.id, 0.0) + rrf
            best_memory[mem.id] = mem
            ch_hits.append(RecallHit(memory=mem, channel=channel, rank=rank, score=rrf))
        per_channel[channel] = ch_hits

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    fused = [
        RecallHit(memory=best_memory[mid], channel="fused", rank=i + 1, score=s)
        for i, (mid, s) in enumerate(ranked)
    ]
    return fused, per_channel

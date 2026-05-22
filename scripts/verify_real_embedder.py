"""End-to-end verification with the actual FastEmbedder (no stubs).

Runs assertions on real behaviour. Exit code 0 = working, 1 = something
broke. Skipped from CI by design (downloads a model on first run) — run
locally before claiming "it works."
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"  FAIL {label}: got {actual!r}, expected {expected!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"  FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def main() -> None:
    # Make sure the env var doesn't override us — we want the actual default.
    os.environ.pop("A5C_ENGRAM_EMBEDDER", None)

    print("\n[1] env var dispatch")
    from a5c_engram.embed import default_embedder
    from a5c_engram.embed.base import FakeEmbedder
    from a5c_engram.embed.fastembed import FastEmbedder

    os.environ["A5C_ENGRAM_EMBEDDER"] = "fake"
    assert_true(isinstance(default_embedder(), FakeEmbedder), "kind=fake → FakeEmbedder")

    os.environ.pop("A5C_ENGRAM_EMBEDDER")
    emb = default_embedder()
    assert_true(isinstance(emb, FastEmbedder), "no env var → FastEmbedder")
    assert_eq(emb.dim, 384, "default dim is 384")

    print("\n[2] real embeddings are normalised and non-trivial")
    v = emb.embed("The database supports concurrent reads.")
    assert_eq(len(v), 384, "embedding length matches dim")
    norm = sum(x * x for x in v) ** 0.5
    assert_true(0.9 < norm < 1.1, f"embedding L2-normalised (norm={norm:.4f})")

    # Two different texts should produce different vectors.
    v2 = emb.embed("Roast the chicken for forty minutes.")
    assert_true(v != v2, "different inputs → different vectors")

    # Semantically similar texts should be closer than unrelated ones.
    v_db = emb.embed("SQLite uses WAL for concurrent reads.")
    v_food = emb.embed("Bake the bread at 200C.")
    sim_close = sum(a * b for a, b in zip(v, v_db))
    sim_far = sum(a * b for a, b in zip(v, v_food))
    assert_true(
        sim_close > sim_far,
        f"semantic-near > semantic-far (close={sim_close:.3f}, far={sim_far:.3f})",
    )

    print("\n[3] Profile.open uses FastEmbedder + correct storage dim")
    from a5c_engram import Profile

    db = Path(tempfile.mkdtemp()) / "verify.db"
    p = Profile.open("verify", db_path=str(db))
    assert_true(isinstance(p.embedder, FastEmbedder), "default profile uses FastEmbedder")
    assert_eq(p.storage.dim, 384, "storage dim matches embedder dim")

    print("\n[4] full write→recall cycle with real embeddings")
    p.remember("Postgres uses MVCC for read isolation.", type="fact")
    p.remember("SQLite supports WAL mode for concurrent reads.", type="fact")
    p.remember("Roast the chicken at 200C for 45 minutes.", type="fact")
    p.remember("The marathon route follows the river.", type="fact")

    result = p.recall("How does the database handle concurrent reads?", use_hyde=False)
    vec_top = result.by_channel["vector"][0].memory.content
    print(f"  vector top: {vec_top!r}")
    assert_true(
        "WAL" in vec_top or "MVCC" in vec_top,
        "vector top is database-related (WAL or MVCC)",
    )

    # Cooking memory should NOT be the top hit.
    assert_true(
        "chicken" not in vec_top and "marathon" not in vec_top,
        "vector top is not the cooking or marathon memory",
    )

    print("\n[5] paraphrases land in FTS index and are reachable")
    m = p.remember("uses gRPC", type="fact", topic="api_style")
    assert_true(m.search_keywords != "", "paraphrases generated and stored")
    # FakeLLM paraphrases include "about uses grpc" — query for that.
    fts_hits = p.storage.search_fts("verify", "about uses grpc")
    hit_ids = [h.id for h in fts_hits]
    assert_true(m.id in hit_ids, "FTS hits via paraphrase-only tokens")

    print("\n[6] supersession survives real-embedder vector index")
    p.remember("uses GraphQL", type="fact", topic="endpoint_style")
    p.remember("uses REST", type="fact", topic="endpoint_style")
    latest = p.storage.latest_for_factkey("verify", "endpoint_style")
    assert_eq(latest.content, "uses REST", "latest fact_key wins")
    qv = p.embedder.embed("what endpoint style?")
    vhits = p.storage.search_vector("verify", qv, k=5)
    # Superseded memory must not show up.
    assert_true(
        all(h.content != "uses GraphQL" for h in vhits),
        "superseded memory hidden from vector channel",
    )

    print("\n[7] temporal channel still works alongside real vector channel")
    tr = p.recall("what changed in the last 24 hours?", use_hyde=True)
    assert_true(len(tr.by_channel["temporal"]) > 0, "temporal hits non-empty")
    assert_eq(len(tr.by_channel["hyde"]), 0, "HyDE bypassed for temporal query")

    print("\n[8] OpenAI and Voyage classes import + error helpfully without keys")
    # We don't actually call the APIs — just confirm the classes can be
    # imported and that they raise clearly without a key.
    from a5c_engram.embed.openai import OpenAIEmbedder
    from a5c_engram.embed.voyage import VoyageEmbedder

    # OpenAI's client constructor raises if no key in env — that's enough.
    saved_oa = os.environ.pop("OPENAI_API_KEY", None)
    saved_vo = os.environ.pop("VOYAGE_API_KEY", None)
    try:
        try:
            OpenAIEmbedder()
            print("  warn OpenAIEmbedder accepted missing key (depends on openai version)")
        except Exception as e:
            print(f"  ok   OpenAIEmbedder raised without key: {type(e).__name__}")
        try:
            VoyageEmbedder()
            print("  warn VoyageEmbedder accepted missing key")
        except Exception as e:
            print(f"  ok   VoyageEmbedder raised without key: {type(e).__name__}")
    finally:
        if saved_oa:
            os.environ["OPENAI_API_KEY"] = saved_oa
        if saved_vo:
            os.environ["VOYAGE_API_KEY"] = saved_vo

    print("\nALL OK — verified end-to-end with the real FastEmbedder.")


if __name__ == "__main__":
    main()

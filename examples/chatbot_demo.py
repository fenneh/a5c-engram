"""End-to-end tour of a5c-engram with no API keys required.

Runs entirely on FakeLLM + FakeEmbedder so it works straight after
`uv add a5c-engram`. Demonstrates, in order:

  1. ingest()   — extracts memories from a conversation
  2. remember() — direct write
  3. supersession — a second write under the same fact_key
  4. paraphrase index — search_keywords visible on every memory
  5. recall() with the per-channel breakdown
  6. temporal channel — a time-shaped query bypasses HyDE
  7. forget()

Run:
    uv run python examples/chatbot_demo.py
"""

from __future__ import annotations

import time

from a5c_engram import Profile


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    p = Profile.open("chatbot-demo", db_path="/tmp/a5c_engram_demo.db")

    # ------------------------------------------------------------------
    banner("1. ingest a short conversation")
    committed = p.ingest(
        [
            {"role": "user", "content": "Hey, my name is Alice Walker."},
            {"role": "assistant", "content": "Got it, Alice."},
            {"role": "user", "content": "Always run tests before pushing."},
        ],
        session_id="2026-05-22-morning",
    )
    for m in committed:
        print(f"  [{m.type.value:11}] {m.fact_key or '—':14}  {m.content}")

    # ------------------------------------------------------------------
    banner("2. remember() — direct writes")
    for content, kind, topic in [
        ("uses GraphQL", "fact", "api_style"),
        ("Never deploy on Fridays.", "instruction", "deploy_policy"),
    ]:
        m = p.remember(content, type=kind, topic=topic)
        print(f"  wrote {kind:11} fact_key={m.fact_key:14}  {m.content}")

    # ------------------------------------------------------------------
    banner("3. supersession — change the api style")
    p.remember("uses gRPC", type="fact", topic="api_style")
    latest = p.storage.latest_for_factkey("chatbot-demo", "api_style")
    chain = p.storage.supersession_chain(latest.id)
    for c in chain:
        marker = "← current" if c.id == latest.id else ""
        print(f"  v{c.version}  {c.content}  {marker}")

    # ------------------------------------------------------------------
    banner("4. paraphrase index — what FTS actually sees")
    sample = chain[-1]
    print(f"  content        : {sample.content!r}")
    print(f"  search_keywords: {sample.search_keywords!r}")
    print("  (search_keywords are LLM-generated paraphrases added to FTS")
    print("   and folded into the embedding — only content is displayed)")

    # ------------------------------------------------------------------
    banner("5. recall — per-channel breakdown")
    result = p.recall("which api do we use?")
    for h in result.hits[:3]:
        print(f"  fused #{h.rank}  score={h.score:.4f}  {h.memory.content}")
    print("  by channel:")
    for ch, hits in result.by_channel.items():
        if hits:
            top = hits[0].memory.content
            print(f"    {ch:8}: {len(hits)} hit(s) — top: {top[:50]}")
        else:
            print(f"    {ch:8}: 0 hits")

    # ------------------------------------------------------------------
    banner("6. temporal channel — bypasses HyDE")
    # Drop a memory dated 'a long time ago' so the time window can filter it.
    fresh = p.remember("just shipped v1.0 to prod", type="event")
    old = p.remember("ancient history fact", type="event")
    conn = p.storage._connect()
    conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        (time.time() - 10 * 86400, old.id),
    )
    conn.commit()

    temporal_result = p.recall("what happened in the last 24 hours?")
    print(f"  temporal hits: {len(temporal_result.by_channel['temporal'])}")
    print(f"  hyde hits   : {len(temporal_result.by_channel['hyde'])}  (0 = bypassed)")
    for h in temporal_result.by_channel["temporal"][:3]:
        print(f"    - {h.memory.content}")

    # ------------------------------------------------------------------
    banner("7. forget — and the chain survives for inspection")
    p.forget(fresh.id)
    print(f"  forgot {fresh.id[:8]} — get_memory now returns:",
          p.storage.get_memory(fresh.id))
    print(f"  supersession chain for {latest.id[:8]} still has {len(p.storage.supersession_chain(latest.id))} entries")

    # ------------------------------------------------------------------
    print("\nDone. Launch the inspector to browse this profile:")
    print("  A5C_ENGRAM_DB=/tmp/a5c_engram_demo.db uv run python -m a5c_engram.server")
    print("  open http://localhost:8000/ui")


if __name__ == "__main__":
    main()

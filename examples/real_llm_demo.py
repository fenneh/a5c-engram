"""Use a real Anthropic LLM for extraction instead of the FakeLLM stub.

The FakeLLM only knows a handful of canned patterns. Wire in a real LLM
and ingest() will produce far richer extractions over arbitrary text.

Setup:
    uv add 'a5c-engram[anthropic,embed]'
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    uv run python examples/real_llm_demo.py
"""

from __future__ import annotations

import os
import sys

from a5c_engram import Profile


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first, e.g.:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # Swap the default FakeLLM for a real Anthropic client. The model arg
    # accepts any Anthropic model id; default is Haiku 4.5 (cheap, fast).
    from a5c_engram.llm.anthropic import AnthropicLLM

    p = Profile.open(
        "real-llm-demo",
        llm=AnthropicLLM(),  # uses ANTHROPIC_API_KEY automatically
        db_path="/tmp/a5c_real_llm_demo.db",
    )

    # A few paragraphs the canned FakeLLM patterns would miss entirely.
    transcript = [
        {
            "role": "user",
            "content": (
                "Quick context: we just finished migrating the auth service from "
                "JWT to opaque session tokens stored in Redis. Token TTL is 24 "
                "hours, refresh extends by another 24. The migration shipped on "
                "2026-05-18 and so far we haven't seen any reports of session loss."
            ),
        },
        {
            "role": "user",
            "content": (
                "Going forward, never store JWT secrets in environment variables — "
                "always pull from Vault at boot. We learned this the hard way when "
                "the staging keys leaked into a CI log in March."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Got it. I'll remember: opaque-token auth, 24h TTL, Vault for "
                "secrets, no env-var JWT keys."
            ),
        },
    ]

    print("Ingesting with real LLM (this calls Anthropic — costs a few cents)…")
    committed = p.ingest(transcript, session_id="auth-migration-debrief")

    print(f"\nExtracted {len(committed)} memories:")
    for m in committed:
        tag = m.type.value.ljust(11)
        key = (m.fact_key or "—").ljust(18)
        print(f"  [{tag}] {key}  {m.content[:80]}")

    print("\nrecall('how do we handle auth secrets?'):")
    result = p.recall("how do we handle auth secrets?")
    for h in result.hits[:3]:
        print(f"  [{h.channel}] rank={h.rank}  {h.memory.content[:80]}")


if __name__ == "__main__":
    main()

"""5-message round trip showing the supersession story.

Run with: uv run python examples/chatbot_demo.py
"""

from __future__ import annotations

from a5c_engram import Profile


def main() -> None:
    p = Profile.open("chatbot-demo", db_path="/tmp/a5c_engram_demo.db")

    # First conversation — establishes initial facts.
    p.ingest(
        [
            {"role": "user", "content": "Hey, my name is Alice Walker."},
            {"role": "assistant", "content": "Got it, Alice."},
            {"role": "user", "content": "We use GraphQL for the API."},
        ],
        session_id="2026-05-22-morning",
    )

    # Later — fact changes.
    p.remember("uses gRPC", type="fact", topic="api_style")

    # Add a behavioural instruction.
    p.remember("Never deploy on Fridays.", type="instruction", topic="deploy_policy")

    print("\n=== latest 'api_style' ===")
    latest = p.storage.latest_for_factkey("chatbot-demo", "api_style")
    print(f"  {latest.content} (v{latest.version})")

    print("\n=== recall: 'what api are we using?' ===")
    result = p.recall("what api are we using?")
    for h in result.hits[:5]:
        print(f"  [{h.channel}] rank={h.rank} score={h.score:.4f}  {h.memory.content}")

    print("\n=== per-channel hits ===")
    for ch, hits in result.by_channel.items():
        print(f"  {ch}: {len(hits)} hit(s)")
        for h in hits[:2]:
            print(f"     - {h.memory.content[:60]}")


if __name__ == "__main__":
    main()

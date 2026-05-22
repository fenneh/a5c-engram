"""Read agent:atlas:memory:* Redis streams from get-rich-or-die-tryin and
ingest them into an a5c-engram profile.

This is illustrative — run from the grdt host where redis is reachable.

Usage:
    uv run python examples/grdt_atlas_demo.py --redis redis://localhost:6379
"""

from __future__ import annotations

import argparse
import json

from a5c_engram import Profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default="redis://localhost:6379")
    ap.add_argument("--role", default="atlas")
    ap.add_argument("--profile-name", default=None)
    ap.add_argument("--db-path", default=None)
    args = ap.parse_args()

    try:
        import redis  # type: ignore
    except ImportError as e:
        raise SystemExit("This example needs `pip install redis`.") from e

    r = redis.Redis.from_url(args.redis, decode_responses=True)
    profile_name = args.profile_name or f"grdt-{args.role}"
    p = Profile.open(profile_name, db_path=args.db_path)

    keys = r.keys(f"agent:{args.role}:memory:*")
    total_msgs = 0
    total_mems = 0

    for key in keys:
        instrument = key.split(":")[-1]
        entries = r.xrange(key)
        if not entries:
            continue
        messages = []
        for _stream_id, fields in entries:
            try:
                action_json = fields.get("proposed_action_json") or "{}"
                action = json.loads(action_json)
            except json.JSONDecodeError:
                action = {}
            text_parts = [
                fields.get("text") or "",
                f"decision: {fields.get('decision', 'unknown')}",
                f"action: {action}",
            ]
            messages.append(
                {"role": "assistant", "content": "\n".join(t for t in text_parts if t)}
            )
        committed = p.ingest(messages, session_id=instrument, use_llm=False)
        total_msgs += len(messages)
        total_mems += len(committed)
        print(f"  {instrument}: {len(messages)} msgs → {len(committed)} memories")

    print(f"\ningested {total_msgs} messages → {total_mems} memories into '{profile_name}'")
    print(f"start the inspector with: uv run python -m a5c_engram.server")


if __name__ == "__main__":
    main()

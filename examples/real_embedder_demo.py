"""Compare embedders on the same corpus and query.

By default we ship FastEmbedder (ONNX-based, ~120MB model, runs on CPU,
no API key). This demo shows it next to FakeEmbedder so you can see what
hash-based fallback looks like, and lets you opt in to OpenAI or Voyage
if you have the keys.

Run:
    uv run python examples/real_embedder_demo.py
    OPENAI_API_KEY=sk-... uv run python examples/real_embedder_demo.py
    VOYAGE_API_KEY=pa-...  uv run python examples/real_embedder_demo.py
"""

from __future__ import annotations

import os

from a5c_engram import Profile
from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.llm.fake import FakeLLM
from a5c_engram.storage.sqlite import SqliteStorage

CORPUS = [
    "Postgres uses MVCC for read isolation.",
    "SQLite supports WAL mode for concurrent reads.",
    "Vault rotates database credentials hourly.",
    "Roast the chicken at 200C for 45 minutes.",
    "The marathon route follows the river.",
    "Redis stores opaque session tokens with a 24h TTL.",
]

QUERY = "How does the database handle concurrent reads?"


def run(label: str, embedder) -> None:
    storage = SqliteStorage(path=f"/tmp/a5c_emb_{label}.db", dim=embedder.dim)
    storage.init()
    p = Profile(label, storage=storage, embedder=embedder, llm=FakeLLM())
    for line in CORPUS:
        p.remember(line, type="fact")
    result = p.recall(QUERY, use_hyde=False)
    print(f"\n=== {label} ({embedder.dim}d) ===")
    for h in result.by_channel["vector"][:3]:
        print(f"  rank={h.rank}  {h.memory.content}")


def main() -> None:
    print(f"Query: {QUERY!r}")

    # Hash-based stub — shows what "no semantics" looks like.
    run("fake", FakeEmbedder())

    # Default: local ONNX bge-small via fastembed.
    from a5c_engram.embed.fastembed import FastEmbedder

    run("fastembed (default)", FastEmbedder())

    # Opt-in paid options.
    if os.getenv("OPENAI_API_KEY"):
        from a5c_engram.embed.openai import OpenAIEmbedder

        run("openai text-embedding-3-small", OpenAIEmbedder())
    else:
        print("\n(skip openai — set OPENAI_API_KEY to include it)")

    if os.getenv("VOYAGE_API_KEY"):
        from a5c_engram.embed.voyage import VoyageEmbedder

        run("voyage-3", VoyageEmbedder())
    else:
        print("(skip voyage — set VOYAGE_API_KEY to include it)")

    print(
        "\nReal embedders cluster semantically similar content, so the "
        "WAL/MVCC memories surface at the top of vector recall. FakeEmbedder "
        "is hash-based and effectively random."
    )


if __name__ == "__main__":
    main()

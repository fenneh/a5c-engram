"""Switch from FakeEmbedder (hash-based, no semantics) to a real local
embedder. Shows the difference concretely on the same query.

Setup:
    uv add 'a5c-engram[embed]'
    # First run downloads ~120MB for BAAI/bge-small-en-v1.5.

Run:
    uv run python examples/real_embedder_demo.py
"""

from __future__ import annotations

from a5c_engram import Profile
from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.storage.sqlite import SqliteStorage

CORPUS = [
    "Postgres uses MVCC for read isolation.",
    "SQLite supports WAL mode for concurrent reads.",
    "Vault rotates database credentials hourly.",
    "Roast the chicken at 200C for 45 minutes.",
    "The marathon route follows the river.",
    "Redis stores opaque session tokens with a 24h TTL.",
]


def seed(profile: Profile) -> None:
    for line in CORPUS:
        profile.remember(line, type="fact")


def run(query: str, label: str, embedder) -> None:
    from a5c_engram.llm.fake import FakeLLM

    storage = SqliteStorage(path=f"/tmp/a5c_emb_{label}.db", dim=embedder.dim)
    storage.init()
    p = Profile(label, storage=storage, embedder=embedder, llm=FakeLLM())
    seed(p)
    print(f"\n=== {label} — query: {query!r} ===")
    result = p.recall(query, use_hyde=False)
    for h in result.by_channel["vector"][:3]:
        print(f"  rank={h.rank}  {h.memory.content}")


def main() -> None:
    query = "How does the database handle concurrent reads?"

    # Run the same query against each embedder.
    run(query, "fake", FakeEmbedder())

    try:
        from a5c_engram.embed.bge import BgeSmallEmbedder
    except ImportError:
        print("\nBgeSmallEmbedder unavailable. Install with:")
        print("  uv add 'a5c-engram[embed]'")
        return

    run(query, "bge", BgeSmallEmbedder())

    print(
        "\nFakeEmbedder is hash-based — vector results are essentially random. "
        "bge-small-en-v1.5 actually clusters semantically similar content, so "
        "the WAL/MVCC memories surface at the top."
    )


if __name__ == "__main__":
    main()

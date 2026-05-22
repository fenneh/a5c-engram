from a5c_engram.embed.base import EmbedAdapter, FakeEmbedder

__all__ = ["EmbedAdapter", "FakeEmbedder", "default_embedder"]


def default_embedder(kind: str | None = None):
    """Return the embedder chosen by A5C_ENGRAM_EMBEDDER (or `kind` override).

    Values:
      - "fastembed" (default) → FastEmbedder, ~120MB local model
      - "fake"               → FakeEmbedder, hash-based, tests only
      - "openai"             → OpenAIEmbedder, needs OPENAI_API_KEY
      - "voyage"             → VoyageEmbedder, needs VOYAGE_API_KEY
      - "bge"                → BgeSmallEmbedder (sentence-transformers)
    """
    import os

    kind = (kind or os.environ.get("A5C_ENGRAM_EMBEDDER", "fastembed")).lower()
    if kind == "fake":
        return FakeEmbedder()
    if kind == "openai":
        from a5c_engram.embed.openai import OpenAIEmbedder

        return OpenAIEmbedder()
    if kind == "voyage":
        from a5c_engram.embed.voyage import VoyageEmbedder

        return VoyageEmbedder()
    if kind == "bge":
        from a5c_engram.embed.bge import BgeSmallEmbedder

        return BgeSmallEmbedder()
    # default
    from a5c_engram.embed.fastembed import FastEmbedder

    return FastEmbedder()

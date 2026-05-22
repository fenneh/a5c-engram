from __future__ import annotations


class BgeSmallEmbedder:
    """sentence-transformers BAAI/bge-small-en-v1.5 — 384-dim, MIT-licensed,
    ~120MB, runs CPU-fine. Optional dependency (a5c-engram[embed])."""

    dim = 384

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "BgeSmallEmbedder needs sentence-transformers. "
                "Install with: uv add 'a5c-engram[embed]'"
            ) from e
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, normalize_embeddings=True).tolist()

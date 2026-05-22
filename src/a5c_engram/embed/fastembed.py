"""ONNX-backed local embedder via qdrant's fastembed. Default model is
BAAI/bge-small-en-v1.5 — 384d, CPU-only, ~120MB downloaded on first use.

This is the default embedder for Profile.open() because it works without
a GPU, doesn't need an API key, and produces real semantic embeddings
(unlike FakeEmbedder)."""

from __future__ import annotations

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


class FastEmbedder:
    """Default local embedder. First call downloads the ONNX weights (~120MB)
    to ~/.cache/fastembed; subsequent calls run entirely on CPU."""

    def __init__(self, model_name: str = DEFAULT_MODEL, dim: int | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "FastEmbedder needs the fastembed package. It should have been "
                "installed automatically. Try: uv add fastembed"
            ) from e
        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim or _dim_for(model_name)

    def embed(self, text: str) -> list[float]:
        # fastembed returns a generator of numpy arrays.
        return next(self._model.embed([text])).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._model.embed(texts)]


# Known dimensions for the fastembed models we mention by name. Fallback to
# 384 (bge-small) if unknown — caller can override via dim=.
_KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "intfloat/multilingual-e5-large": 1024,
}


def _dim_for(model_name: str) -> int:
    return _KNOWN_DIMS.get(model_name, DEFAULT_DIM)

"""Voyage embedder. voyage-3 by default — currently top of MTEB. voyage-3-large
for the SOTA tier. Both 1024d.

Install:
    uv add 'a5c-engram[paid]'
Auth:
    export VOYAGE_API_KEY=pa-..."""

from __future__ import annotations

import os

_DIMS = {
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
}


class VoyageEmbedder:
    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        dim: int | None = None,
        input_type: str = "document",
    ):
        try:
            import voyageai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "VoyageEmbedder needs the voyageai package. Install with: uv add 'a5c-engram[paid]'"
            ) from e
        self._client = voyageai.Client(api_key=api_key or os.getenv("VOYAGE_API_KEY"))
        self._model = model
        self._input_type = input_type
        self.dim = dim or _DIMS.get(model, 1024)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embed([text], model=self._model, input_type=self._input_type)
        return resp.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(texts, model=self._model, input_type=self._input_type)
        return resp.embeddings

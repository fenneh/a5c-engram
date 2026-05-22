"""OpenAI embedder. text-embedding-3-small (1536d) by default — cheap
($0.02/M tokens) and solid. Pass model='text-embedding-3-large' for the
larger 3072d variant.

Install:
    uv add 'a5c-engram[paid]'
Auth:
    export OPENAI_API_KEY=sk-..."""

from __future__ import annotations

import os

_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dim: int | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder needs the openai package. Install with: uv add 'a5c-engram[paid]'"
            ) from e
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model
        self.dim = dim or _DIMS.get(model, 1536)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

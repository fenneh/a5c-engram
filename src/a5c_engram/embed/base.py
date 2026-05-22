from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbedAdapter(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hash-based embedder for tests. Same input → same vector.
    Not semantically meaningful but enough for unit tests of the pipeline."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        rng_seed = hashlib.sha256(text.encode()).digest()
        out = []
        for i in range(self.dim):
            b = rng_seed[i % len(rng_seed)]
            angle = (b / 255.0) * 2.0 * math.pi
            out.append(math.sin(angle + i))
        # L2 normalise
        n = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / n for x in out]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

"""Test-only stubs that are predictable enough to assert on."""

from __future__ import annotations

import math

from a5c_engram.embed.base import FakeEmbedder
from a5c_engram.llm.base import ExtractionCandidate


class MockEmbedder:
    """Embedder where every (text, vector) pair is declared up front. Anything
    not in the map falls through to a deterministic hash. Lets vector-channel
    tests assert which memory the query lands closest to."""

    dim = 8

    def __init__(self, mapping: dict[str, list[float]] | None = None):
        self._mapping = dict(mapping or {})
        self._fallback = FakeEmbedder(dim=self.dim)

    def add(self, text: str, vec: list[float]) -> None:
        self._mapping[text] = _norm(vec)

    def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return self._mapping[text]
        # If any key is a substring, prefer that — handy for HyDE phrases.
        for k, v in self._mapping.items():
            if k in text:
                return v
        v = self._fallback.embed(text)
        return v[: self.dim] if len(v) >= self.dim else v + [0.0] * (self.dim - len(v))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class CountingLLM:
    """Wraps another LLMAdapter and counts calls to extract / hyde / synthesise."""

    def __init__(self, inner):
        self._inner = inner
        self.extract_calls: list[str] = []
        self.hyde_calls = 0
        self.synth_calls = 0

    def extract(self, text):
        self.extract_calls.append(text)
        return self._inner.extract(text)

    def hyde(self, query):
        self.hyde_calls += 1
        return self._inner.hyde(query)

    def synthesise(self, query, memories):
        self.synth_calls += 1
        return self._inner.synthesise(query, memories)


class ScriptedLLM:
    """LLM that returns a queued list of extraction results per extract() call.
    Useful for verifying dual-pass logic and dedup."""

    def __init__(self, scripted: list[list[ExtractionCandidate]]):
        self._queue = list(scripted)

    def extract(self, text):
        if not self._queue:
            return []
        return self._queue.pop(0)

    def hyde(self, query):
        return "hypothetical"

    def synthesise(self, query, memories):
        return memories[0] if memories else ""


class MalformedLLM:
    """An LLM whose extract() raises or returns garbage. Used to confirm
    the pipeline degrades gracefully."""

    def __init__(self, mode: str = "garbage"):
        self.mode = mode

    def extract(self, text):
        if self.mode == "raise":
            raise RuntimeError("simulated LLM failure")
        if self.mode == "wrong_type":
            return [ExtractionCandidate(type="banana", content="weird")]
        if self.mode == "empty_content":
            return [ExtractionCandidate(type="fact", content="")]
        return []

    def hyde(self, query):
        return ""

    def synthesise(self, query, memories):
        return ""


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

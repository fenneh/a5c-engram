from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ExtractionCandidate:
    type: str           # "fact" | "event" | "instruction" | "task"
    content: str
    fact_key: str | None = None
    confidence: float = 0.5
    evidence: str | None = None  # raw quote supporting this extraction


class LLMAdapter(Protocol):
    def extract(self, text: str) -> list[ExtractionCandidate]:
        """Return candidate memories from a chunk of conversation."""

    def hyde(self, query: str) -> str:
        """Generate a hypothetical answer to ``query`` for HyDE embedding."""

    def synthesise(self, query: str, memories: list[str]) -> str:
        """Synthesise a short answer from supporting memories. Returns plain text."""

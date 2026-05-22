from __future__ import annotations

import re

from a5c_engram.llm.base import ExtractionCandidate

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text: str) -> list[str]:
    stripped = re.sub(r"^\s*\[(?:user|assistant|system|tool)\]\s*", "", text, flags=re.MULTILINE)
    return [s.strip() for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]


class FakeLLM:
    """Deterministic stub LLM for tests. Recognises a handful of patterns so
    the pipeline can be exercised without network calls."""

    def extract(self, text: str) -> list[ExtractionCandidate]:
        out: list[ExtractionCandidate] = []
        lower = text.lower()
        # Mimic crude fact extraction.
        if "uses graphql" in lower:
            out.append(
                ExtractionCandidate(
                    type="fact",
                    content="The project uses GraphQL.",
                    fact_key="api_style",
                    confidence=0.9,
                    evidence=text[:120],
                )
            )
        if "uses rest" in lower:
            out.append(
                ExtractionCandidate(
                    type="fact",
                    content="The project uses REST.",
                    fact_key="api_style",
                    confidence=0.9,
                    evidence=text[:120],
                )
            )
        # Sentence-scoped matches so the stub doesn't capture a whole
        # multi-message blob just because one sentence triggers a pattern.
        for sentence in _sentences(text):
            sl = sentence.lower()
            if "deployed" in sl or "shipped" in sl:
                out.append(
                    ExtractionCandidate(
                        type="event",
                        content=sentence[:200],
                        confidence=0.6,
                        evidence=sentence[:200],
                    )
                )
            if "always " in sl or "never " in sl:
                out.append(
                    ExtractionCandidate(
                        type="instruction",
                        content=sentence[:200],
                        confidence=0.6,
                        evidence=sentence[:200],
                    )
                )
        return out

    def paraphrase(self, content: str) -> list[str]:
        # Deterministic stub: a couple of trivially transformed paraphrases
        # so the indexer has something non-empty to write. Real LLMs do far
        # better — see AnthropicLLM.paraphrase.
        lower = content.lower().rstrip(".!?")
        return [lower, f"about {lower}", f"info on {lower}"]

    def hyde(self, query: str) -> str:
        return f"A likely answer to '{query}' is that the relevant facts apply here."

    def synthesise(self, query: str, memories: list[str]) -> str:
        if not memories:
            return ""
        return memories[0]

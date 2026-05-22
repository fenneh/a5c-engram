from __future__ import annotations

from a5c_engram.llm.base import ExtractionCandidate


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
                    type="fact", content="The project uses GraphQL.",
                    fact_key="api_style", confidence=0.9, evidence=text[:120],
                )
            )
        if "uses rest" in lower:
            out.append(
                ExtractionCandidate(
                    type="fact", content="The project uses REST.",
                    fact_key="api_style", confidence=0.9, evidence=text[:120],
                )
            )
        if "deployed" in lower or "shipped" in lower:
            out.append(
                ExtractionCandidate(
                    type="event", content=text.strip()[:200],
                    confidence=0.6, evidence=text[:200],
                )
            )
        if "always " in lower or "never " in lower:
            out.append(
                ExtractionCandidate(
                    type="instruction", content=text.strip()[:200],
                    confidence=0.6, evidence=text[:200],
                )
            )
        return out

    def hyde(self, query: str) -> str:
        return f"A likely answer to '{query}' is that the relevant facts apply here."

    def synthesise(self, query: str, memories: list[str]) -> str:
        if not memories:
            return ""
        return memories[0]

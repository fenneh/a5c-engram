from __future__ import annotations

import re

from a5c_engram.llm.base import ExtractionCandidate

# Self-reference patterns — cheap, high-precision facts.
_SELF_REF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmy name is ([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)", re.IGNORECASE), "user_name"),
    (re.compile(r"\bI(?:'|\s+a)m\s+([a-zA-Z]+\s+[a-zA-Z]+)\b"), "user_identity"),
    (re.compile(r"\bI\s+(?:work|am working)\s+(?:at|for)\s+([A-Z][\w\-\s]{2,30})"), "user_employer"),
    (re.compile(r"\bI\s+(?:live|am based)\s+in\s+([A-Z][\w\-\s]{2,30})"), "user_location"),
    (re.compile(r"\bI\s+use\s+([a-zA-Z][\w\-]+(?:\s+[a-zA-Z][\w\-]+)?)\s+(?:for|as)\b"), "user_tooling"),
]

_ISO_DATE = re.compile(r"\b(20\d{2}-[01]\d-[0-3]\d)\b")
_RELATIVE_DATE = re.compile(
    r"\b(yesterday|today|tomorrow|next week|last week)\b", re.IGNORECASE
)

_NUMERIC_FACT = re.compile(
    r"\b([a-z][\w\-]{2,30})\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE
)

_INSTRUCTION_VERBS = re.compile(
    r"\b(always|never|don'?t|do not|must|should|prefer|avoid)\b", re.IGNORECASE
)

_TASK_VERBS = re.compile(
    r"\b(remind me|TODO|todo:|let'?s|need to|will|going to)\b", re.IGNORECASE
)


def deterministic_extract(text: str) -> list[ExtractionCandidate]:
    """Cheap, testable, no-LLM extraction. Half the corpus extracts here."""
    out: list[ExtractionCandidate] = []

    for pat, key in _SELF_REF_PATTERNS:
        m = pat.search(text)
        if m:
            out.append(
                ExtractionCandidate(
                    type="fact",
                    content=m.group(0).strip().rstrip(".,!?"),
                    fact_key=key,
                    confidence=0.85,
                    evidence=text[max(0, m.start() - 20): m.end() + 20],
                )
            )

    for m in _ISO_DATE.finditer(text):
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        snippet = text[start:end].strip()
        out.append(
            ExtractionCandidate(
                type="event",
                content=snippet,
                confidence=0.6,
                evidence=snippet,
            )
        )

    for m in _NUMERIC_FACT.finditer(text):
        key = m.group(1).lower().replace(" ", "_")
        val = m.group(2)
        out.append(
            ExtractionCandidate(
                type="fact",
                content=f"{m.group(1)} = {val}",
                fact_key=key,
                confidence=0.75,
                evidence=text[max(0, m.start() - 20): m.end() + 20],
            )
        )

    if _INSTRUCTION_VERBS.search(text) and len(text) < 400:
        out.append(
            ExtractionCandidate(
                type="instruction",
                content=text.strip()[:200],
                confidence=0.5,
                evidence=text[:200],
            )
        )

    if _TASK_VERBS.search(text) and len(text) < 240:
        out.append(
            ExtractionCandidate(
                type="task",
                content=text.strip()[:200],
                confidence=0.5,
                evidence=text[:200],
            )
        )

    return out


def has_relative_date(text: str) -> bool:
    return bool(_RELATIVE_DATE.search(text))

from __future__ import annotations

import re

from a5c_engram.llm.base import ExtractionCandidate

# Self-reference patterns — cheap, high-precision facts.
_SELF_REF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmy name is ([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)", re.IGNORECASE), "user_name"),
    (re.compile(r"\bI(?:'|\s+a)m\s+([a-zA-Z]+\s+[a-zA-Z]+)\b"), "user_identity"),
    (
        re.compile(r"\bI\s+(?:work|am working)\s+(?:at|for)\s+([A-Z][\w\-\s]{2,30})"),
        "user_employer",
    ),
    (re.compile(r"\bI\s+(?:live|am based)\s+in\s+([A-Z][\w\-\s]{2,30})"), "user_location"),
    (
        re.compile(r"\bI\s+use\s+([a-zA-Z][\w\-]+(?:\s+[a-zA-Z][\w\-]+)?)\s+(?:for|as)\b"),
        "user_tooling",
    ),
]

_ISO_DATE = re.compile(r"\b(20\d{2}-[01]\d-[0-3]\d)\b")
_RELATIVE_DATE = re.compile(r"\b(yesterday|today|tomorrow|next week|last week)\b", re.IGNORECASE)

_NUMERIC_FACT = re.compile(r"\b([a-z][\w\-]{2,30})\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)

_INSTRUCTION_VERBS = re.compile(
    r"\b(always|never|don'?t|do not|must|should|prefer|avoid)\b", re.IGNORECASE
)

_TASK_VERBS = re.compile(r"\b(remind me|TODO|todo:|let'?s|need to|will|going to)\b", re.IGNORECASE)


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
                    evidence=text[max(0, m.start() - 20) : m.end() + 20],
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
                evidence=text[max(0, m.start() - 20) : m.end() + 20],
            )
        )

    for sentence in _split_sentences(text):
        clean = sentence.strip()
        if not clean or len(clean) > 240:
            continue
        if _INSTRUCTION_VERBS.search(clean):
            out.append(
                ExtractionCandidate(
                    type="instruction",
                    content=clean[:200],
                    confidence=0.5,
                    evidence=clean[:200],
                )
            )
        if _TASK_VERBS.search(clean):
            out.append(
                ExtractionCandidate(
                    type="task",
                    content=clean[:200],
                    confidence=0.5,
                    evidence=clean[:200],
                )
            )

    return out


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter — good enough for instruction/task scoping
    without pulling in a tokenizer."""
    # Strip leading [role] tags inserted by Profile.ingest so they don't
    # leak into the extracted content.
    stripped = re.sub(r"^\s*\[(?:user|assistant|system|tool)\]\s*", "", text, flags=re.MULTILINE)
    return [s for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]


def has_relative_date(text: str) -> bool:
    return bool(_RELATIVE_DATE.search(text))


def parse_temporal_range(text: str, *, now: float | None = None) -> tuple[float, float] | None:
    """Return (start_ts, end_ts) for natural-language time phrases, or None.

    Deterministic so we never need an LLM for "what did I say yesterday?"
    queries. Uses local-day boundaries for yesterday/today/tomorrow and
    rolling windows for last-N-units expressions."""
    import time as _t

    now = now if now is not None else _t.time()
    SECOND = 1.0
    MINUTE = 60.0
    HOUR = 3600.0
    DAY = 86400.0
    WEEK = 7 * DAY

    lower = text.lower()

    # last/in the last N <unit>
    m = re.search(
        r"\b(?:in\s+the\s+)?(?:last|past)\s+(\d+)\s+(minute|minutes|hour|hours|day|days|week|weeks)\b",
        lower,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        unit_seconds = {
            "minute": MINUTE,
            "minutes": MINUTE,
            "hour": HOUR,
            "hours": HOUR,
            "day": DAY,
            "days": DAY,
            "week": WEEK,
            "weeks": WEEK,
        }[unit]
        return (now - n * unit_seconds, now + SECOND)

    # day-boundary local-time anchors
    local = _t.localtime(now)
    day_start = _t.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))

    if re.search(r"\byesterday\b", lower):
        return (day_start - DAY, day_start)
    if re.search(r"\btoday\b", lower):
        return (day_start, day_start + DAY)
    if re.search(r"\btomorrow\b", lower):
        return (day_start + DAY, day_start + 2 * DAY)
    if re.search(r"\b(last|past)\s+week\b", lower):
        return (day_start - 7 * DAY, day_start)
    if re.search(r"\bnext\s+week\b", lower):
        return (day_start + DAY, day_start + 8 * DAY)

    return None

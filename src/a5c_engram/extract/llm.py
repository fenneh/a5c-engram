from __future__ import annotations

from a5c_engram.llm.base import ExtractionCandidate, LLMAdapter

CHUNK_CHARS = 10_000
DETAIL_WINDOW_CHARS = 2_500
DETAIL_OVERLAP = 600


def llm_extract(text: str, llm: LLMAdapter) -> list[ExtractionCandidate]:
    """Dual-pass extraction. Full chunks for context, then overlapping detail
    windows for concrete values. Dedup by (type, normalised content)."""
    if not text or not text.strip():
        return []

    candidates: list[ExtractionCandidate] = []

    # Full pass — large chunks, coarse extraction.
    for chunk in _chunks(text, CHUNK_CHARS):
        candidates.extend(llm.extract(chunk))

    # Detail pass — only for long conversations.
    if len(text) > CHUNK_CHARS // 2:
        for window in _windows(text, DETAIL_WINDOW_CHARS, DETAIL_OVERLAP):
            candidates.extend(llm.extract(window))

    return _dedup(candidates)


def _chunks(s: str, size: int) -> list[str]:
    if len(s) <= size:
        return [s]
    return [s[i:i + size] for i in range(0, len(s), size)]


def _windows(s: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    out = []
    i = 0
    while i < len(s):
        out.append(s[i:i + size])
        if i + size >= len(s):
            break
        i += step
    return out


def _dedup(cands: list[ExtractionCandidate]) -> list[ExtractionCandidate]:
    seen: set[tuple[str, str]] = set()
    out = []
    for c in cands:
        key = (c.type, " ".join(c.content.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

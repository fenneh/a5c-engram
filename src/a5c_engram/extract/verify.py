from __future__ import annotations

from a5c_engram.llm.base import ExtractionCandidate


def verify_candidate(cand: ExtractionCandidate, source_text: str) -> tuple[bool, str]:
    """Eight cheap verification checks that run before a candidate is
    stored. Returns (ok, reason). Catches empty/over-long content,
    unknown types, candidates with no token overlap with the source
    text, and malformed fact_keys."""

    content = (cand.content or "").strip()
    src = source_text or ""

    # 1. non-empty
    if not content:
        return False, "empty content"
    # 2. minimum length
    if len(content) < 4:
        return False, "too short"
    # 3. maximum length — extractions should be atomic, not paragraphs
    if len(content) > 600:
        return False, "too long (not atomic)"
    # 4. type validity
    if cand.type not in {"fact", "event", "instruction", "task"}:
        return False, f"unknown type: {cand.type}"
    # 5. confidence sane
    if not (0.0 <= cand.confidence <= 1.0):
        return False, "confidence out of [0, 1]"
    # 6. evidence support — content must overlap with source (or evidence)
    if src:
        if not _shares_token(content, src) and not _shares_token(content, cand.evidence or ""):
            return False, "no factual support in source"
    # 7. fact_key shape (snake_case) if present
    if cand.fact_key is not None:
        if not cand.fact_key or any(c.isspace() or c.isupper() for c in cand.fact_key):
            return False, "fact_key must be snake_case lowercase"
    # 8. facts/instructions of meaningful length
    if cand.type in {"fact", "instruction"} and len(content.split()) < 2:
        return False, "fact/instruction too brief to be useful"

    return True, "ok"


def _shares_token(a: str, b: str) -> bool:
    """True if a and b share at least one non-stopword token of length >= 3."""
    if not a or not b:
        return False
    a_toks = {t.lower() for t in _split(a) if len(t) >= 3 and t.lower() not in _STOP}
    b_toks = {t.lower() for t in _split(b) if len(t) >= 3 and t.lower() not in _STOP}
    return bool(a_toks & b_toks)


def _split(s: str) -> list[str]:
    out = []
    cur = []
    for c in s:
        if c.isalnum():
            cur.append(c)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were",
    "you", "your", "our", "their", "his", "her", "has", "have", "had", "but",
    "not", "use", "uses", "used", "any", "all", "some",
}

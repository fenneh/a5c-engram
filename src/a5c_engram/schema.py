from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class MemoryType(StrEnum):
    FACT = "fact"
    EVENT = "event"
    INSTRUCTION = "instruction"
    TASK = "task"


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    ts: float = field(default_factory=time.time)
    session_id: str | None = None

    def content_id(self, profile: str) -> str:
        h = hashlib.sha256()
        h.update(profile.encode())
        h.update(b"|")
        h.update((self.session_id or "").encode())
        h.update(b"|")
        h.update(self.role.encode())
        h.update(b"|")
        h.update(self.content.encode())
        return h.hexdigest()[:32]


@dataclass
class Memory:
    id: str
    profile: str
    type: MemoryType
    content: str
    search_keywords: str = ""
    created_at: float = field(default_factory=time.time)
    fact_key: str | None = None
    version: int = 1
    superseded_by: str | None = None
    source_session_id: str | None = None
    source_message_id: str | None = None
    expires_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        profile: str,
        type: MemoryType | str,
        content: str,
        fact_key: str | None = None,
        search_keywords: str = "",
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        expires_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Memory:
        # Deterministic id from the key fields. Two writes of the same
        # (profile, type, session_id, fact_key, content) produce the same
        # id, so the storage layer's INSERT OR REPLACE collapses them and
        # remember() is safely re-runnable.
        h = hashlib.sha256()
        h.update(profile.encode())
        h.update(b"|")
        h.update(MemoryType(type).value.encode())
        h.update(b"|")
        h.update((source_session_id or "").encode())
        h.update(b"|")
        h.update((fact_key or "").encode())
        h.update(b"|")
        h.update(content.encode())
        mem_id = h.hexdigest()[:32]
        return cls(
            id=mem_id,
            profile=profile,
            type=MemoryType(type),
            content=content,
            search_keywords=search_keywords,
            fact_key=fact_key,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            expires_at=expires_at,
            extra=extra or {},
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Memory:
        d = dict(d)
        d["type"] = MemoryType(d["type"])
        return cls(**d)


@dataclass
class RecallHit:
    """One memory + its retrieval provenance.

    `score` is the RRF fusion score for fused hits or the per-channel
    RRF contribution for channel hits. RRF scores are rank-based, not
    similarity-based, so the absolute number isn't comparable across
    queries — use it for ordering, not thresholding.

    `channel_count` (fused hits only) is how many channels surfaced
    this memory. A memory hit by 3+ channels is a much stronger signal
    than one hit by a single channel at the same fused score. Per-channel
    hits leave this at 1.
    """

    memory: Memory
    channel: str
    rank: int
    score: float
    channel_count: int = 1


@dataclass
class RecallResult:
    """Result of Profile.recall — fused hits plus per-channel breakdown."""

    query: str
    hits: list[RecallHit]
    by_channel: dict[str, list[RecallHit]]
    synthesis: str | None = None

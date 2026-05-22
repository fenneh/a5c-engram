from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class MemoryType(str, Enum):
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
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        expires_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Memory:
        return cls(
            id=str(uuid.uuid4()),
            profile=profile,
            type=MemoryType(type),
            content=content,
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
    """One memory + its retrieval provenance."""

    memory: Memory
    channel: str
    rank: int
    score: float


@dataclass
class RecallResult:
    """Result of Profile.recall — fused hits plus per-channel breakdown."""

    query: str
    hits: list[RecallHit]
    by_channel: dict[str, list[RecallHit]]
    synthesis: str | None = None

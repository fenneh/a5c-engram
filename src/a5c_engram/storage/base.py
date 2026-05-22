from __future__ import annotations

from typing import Protocol

from a5c_engram.schema import Memory, MemoryType, Message


class StorageAdapter(Protocol):
    """Backend-agnostic memory store."""

    def init(self) -> None: ...

    def list_profiles(self) -> list[dict]:
        """Return [{name, counts: {fact, event, ...}, last_ingested_at}]."""

    def add_message(self, profile: str, msg: Message) -> str:
        """Insert raw message (idempotent on content_id). Return content_id."""

    def add_memory(self, mem: Memory) -> None: ...

    def get_memory(self, mem_id: str) -> Memory | None: ...

    def list_memories(
        self,
        profile: str,
        *,
        type: MemoryType | None = None,
        topic: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]: ...

    def forget(self, mem_id: str) -> bool: ...

    # ---- retrieval channels ----------------------------------------
    def search_fts(self, profile: str, query: str, k: int = 10) -> list[Memory]: ...
    def search_raw(self, profile: str, query: str, k: int = 10) -> list[Memory]: ...
    def search_factkey(self, profile: str, topic: str) -> Memory | None: ...
    def search_vector(
        self, profile: str, embedding: list[float], k: int = 10
    ) -> list[Memory]: ...

    # ---- supersession ----------------------------------------------
    def latest_for_factkey(self, profile: str, fact_key: str) -> Memory | None: ...
    def supersede(self, profile: str, fact_key: str, new_mem: Memory) -> None:
        """Insert new_mem and mark prior latest as superseded by it."""

    def supersession_chain(self, mem_id: str) -> list[Memory]:
        """Return ancestors (oldest → newest) ending at mem_id."""

    # ---- vector index ----------------------------------------------
    def upsert_embedding(self, mem_id: str, embedding: list[float]) -> None: ...

    # ---- ingest log ------------------------------------------------
    def list_sessions(self, profile: str) -> list[dict]:
        """Return [{session_id, message_count, first_ts, last_ts, memory_count}]."""

    def memories_for_session(self, profile: str, session_id: str) -> list[Memory]: ...

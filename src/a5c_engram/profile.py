from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from a5c_engram.embed.base import EmbedAdapter, FakeEmbedder
from a5c_engram.extract.deterministic import deterministic_extract
from a5c_engram.extract.llm import llm_extract
from a5c_engram.extract.verify import verify_candidate
from a5c_engram.llm.base import ExtractionCandidate, LLMAdapter
from a5c_engram.llm.fake import FakeLLM
from a5c_engram.retrieve.fuse import rrf_fuse
from a5c_engram.schema import Memory, MemoryType, Message, RecallResult
from a5c_engram.storage.base import StorageAdapter
from a5c_engram.storage.sqlite import SqliteStorage


def _default_db_path() -> str:
    p = os.environ.get("A5C_ENGRAM_DB")
    if p:
        return p
    base = Path.home() / ".a5c-engram"
    base.mkdir(exist_ok=True)
    return str(base / "engram.db")


class Profile:
    """A named, isolated memory profile.

    A Profile bundles a storage adapter, an embedder, and an LLM. Each
    Profile.open(name) returns a handle that the agent can ingest into and
    recall from."""

    def __init__(
        self,
        name: str,
        *,
        storage: StorageAdapter,
        embedder: EmbedAdapter,
        llm: LLMAdapter,
    ):
        self.name = name
        self.storage = storage
        self.embedder = embedder
        self.llm = llm

    # ------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        name: str,
        *,
        storage: StorageAdapter | None = None,
        embedder: EmbedAdapter | None = None,
        llm: LLMAdapter | None = None,
        db_path: str | None = None,
    ) -> Profile:
        if storage is None:
            storage = SqliteStorage(path=db_path or _default_db_path())
            storage.init()
        if embedder is None:
            embedder = FakeEmbedder()
        if llm is None:
            llm = FakeLLM()
        return cls(name, storage=storage, embedder=embedder, llm=llm)

    # ------------------------------------------------------------------
    def ingest(
        self,
        messages: list[dict[str, Any] | Message],
        *,
        session_id: str | None = None,
        use_llm: bool = True,
    ) -> list[Memory]:
        """Run extraction (deterministic + LLM) over messages and store the
        verified candidates. Returns the memories that were committed."""
        msgs = [_as_message(m, session_id) for m in messages]
        for m in msgs:
            self.storage.add_message(self.name, m)

        text = "\n".join(f"[{m.role}] {m.content}" for m in msgs)
        candidates = deterministic_extract(text)
        if use_llm and self.llm is not None:
            candidates.extend(llm_extract(text, self.llm))

        committed: list[Memory] = []
        for cand in candidates:
            ok, _reason = verify_candidate(cand, text)
            if not ok:
                continue
            mem = self._commit_candidate(cand, session_id=session_id)
            if mem is not None:
                committed.append(mem)
        return committed

    def remember(
        self,
        content: str,
        *,
        type: str | MemoryType = MemoryType.FACT,
        topic: str | None = None,
        session_id: str | None = None,
        expires_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Memory:
        """Directly write a memory (no extraction). Used when the agent itself
        decides what to remember."""
        cand = ExtractionCandidate(
            type=MemoryType(type).value,
            content=content,
            fact_key=topic,
            confidence=1.0,
            evidence=content,
        )
        mem = self._commit_candidate(cand, session_id=session_id, expires_at=expires_at, extra=extra)
        assert mem is not None
        return mem

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        use_hyde: bool = True,
        synthesise: bool = False,
    ) -> RecallResult:
        """Five-channel retrieval with RRF fusion."""
        by_channel: dict[str, list[Memory]] = {}

        # Channel 1: FTS
        by_channel["fts"] = self.storage.search_fts(self.name, query, k=k)

        # Channel 2: factkey — try snake_case-ifying salient nouns.
        factkey_hit = None
        for cand_key in _candidate_factkeys(query):
            factkey_hit = self.storage.search_factkey(self.name, cand_key)
            if factkey_hit:
                break
        by_channel["factkey"] = [factkey_hit] if factkey_hit else []

        # Channel 3: raw message FTS
        by_channel["raw"] = self.storage.search_raw(self.name, query, k=k)

        # Channel 4: direct vector
        if self.embedder is not None:
            qvec = self.embedder.embed(query)
            by_channel["vector"] = self.storage.search_vector(self.name, qvec, k=k)
        else:
            by_channel["vector"] = []

        # Channel 5: HyDE
        if use_hyde and self.llm is not None and self.embedder is not None:
            hypothetical = self.llm.hyde(query)
            hvec = self.embedder.embed(hypothetical)
            by_channel["hyde"] = self.storage.search_vector(self.name, hvec, k=k)
        else:
            by_channel["hyde"] = []

        fused, per_channel = rrf_fuse(by_channel, top_n=k)

        synthesis = None
        if synthesise and self.llm is not None and fused:
            synthesis = self.llm.synthesise(query, [h.memory.content for h in fused])

        return RecallResult(
            query=query,
            hits=fused,
            by_channel=per_channel,
            synthesis=synthesis,
        )

    def list(
        self,
        *,
        type: str | MemoryType | None = None,
        topic: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        t = MemoryType(type) if type is not None else None
        return self.storage.list_memories(
            self.name, type=t, topic=topic, limit=limit, offset=offset
        )

    def forget(self, memory_id: str) -> bool:
        return self.storage.forget(memory_id)

    # ------------------------------------------------------------------
    def _commit_candidate(
        self,
        cand: ExtractionCandidate,
        *,
        session_id: str | None,
        expires_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Memory | None:
        mem = Memory.new(
            profile=self.name,
            type=cand.type,
            content=cand.content,
            fact_key=cand.fact_key,
            source_session_id=session_id,
            expires_at=expires_at,
            extra={"confidence": cand.confidence, **(extra or {})},
        )
        if cand.fact_key:
            self.storage.supersede(self.name, cand.fact_key, mem)
        else:
            self.storage.add_memory(mem)

        # Tasks are excluded from vector index — they're ephemeral.
        if mem.type != MemoryType.TASK and self.embedder is not None:
            vec = self.embedder.embed(mem.content)
            self.storage.upsert_embedding(mem.id, vec)
        return mem


def _as_message(m: dict[str, Any] | Message, session_id: str | None) -> Message:
    if isinstance(m, Message):
        if m.session_id is None:
            m.session_id = session_id
        return m
    return Message(
        role=m.get("role", "user"),
        content=m["content"],
        session_id=m.get("session_id", session_id),
        ts=m.get("ts", __import__("time").time()),
    )


_TOKEN = re.compile(r"[a-zA-Z][\w]+")


def _candidate_factkeys(query: str) -> list[str]:
    """Produce a few snake_case candidates from the query for factkey lookup."""
    toks = [t.lower() for t in _TOKEN.findall(query) if len(t) >= 3]
    if not toks:
        return []
    out: list[str] = []
    # bigrams first (more specific), then unigrams.
    for i in range(len(toks) - 1):
        out.append(f"{toks[i]}_{toks[i + 1]}")
    out.extend(toks)
    return out

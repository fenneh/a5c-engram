from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from a5c_engram.embed import default_embedder
from a5c_engram.embed.base import EmbedAdapter
from a5c_engram.extract.deterministic import (
    deterministic_extract,
    parse_temporal_range,
)
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


# Direct-write content is capped so misuse (logging a 100kb blob to
# memory) gets rejected loudly rather than silently bloating storage and
# the FTS index. The verify_candidate path used by ingest() is already
# stricter (600-char atomic-extraction cap); this cap is the looser
# "you definitely didn't mean to do this" ceiling for direct remember()
# calls where the caller wants to bypass extraction.
MAX_DIRECT_CONTENT_CHARS = 10_000


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
        if embedder is None:
            embedder = default_embedder()
        if storage is None:
            storage = SqliteStorage(path=db_path or _default_db_path(), dim=embedder.dim)
            storage.init()
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

        # Dedup across deterministic + LLM by (type, normalised content).
        seen: set[tuple[str, str]] = set()
        committed: list[Memory] = []
        for cand in candidates:
            dedup_key = (cand.type, " ".join(cand.content.lower().split()))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
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
        """Directly write a memory, bypassing extraction. Used when the
        caller already knows the type and content shape.

        Idempotent: two `remember()` calls with the same `(type, content,
        session_id, topic)` write the same memory id, so the storage's
        INSERT OR REPLACE collapses them. When `topic` is set, repeated
        writes of the same content no-op rather than growing the
        supersession chain forever.

        Raises ValueError on empty / whitespace-only content, on content
        longer than MAX_DIRECT_CONTENT_CHARS, or on an unknown `type`.
        """
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        if len(content) > MAX_DIRECT_CONTENT_CHARS:
            raise ValueError(
                f"content too long ({len(content)} chars > "
                f"{MAX_DIRECT_CONTENT_CHARS}); use ingest() for chunked "
                "extraction or pre-truncate before calling remember()"
            )
        try:
            type_value = MemoryType(type).value
        except ValueError:
            valid = ", ".join(t.value for t in MemoryType)
            raise ValueError(
                f"unknown memory type {type!r}; must be one of: {valid}"
            ) from None
        cand = ExtractionCandidate(
            type=type_value,
            content=content,
            fact_key=topic,
            confidence=1.0,
            evidence=content,
        )
        mem = self._commit_candidate(
            cand, session_id=session_id, expires_at=expires_at, extra=extra
        )
        if mem is None:
            # _commit_candidate currently returns Memory in all branches.
            # Guard anyway so callers get a clear error instead of an
            # opaque assertion if a future refactor changes that.
            raise RuntimeError("memory commit returned None")
        return mem

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        use_hyde: bool = True,
        synthesise: bool = False,
    ) -> RecallResult:
        """Six-channel retrieval with RRF fusion. The temporal channel only
        activates when the query contains a natural-language time phrase
        (yesterday, last week, last 24 hours, etc.) and bypasses the LLM
        HyDE call when it does — we already know the answer is a time
        window."""
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

        # Channel 5: temporal — deterministic time-window parser.
        time_range = parse_temporal_range(query)
        if time_range is not None:
            start_ts, end_ts = time_range
            by_channel["temporal"] = self.storage.search_temporal(self.name, start_ts, end_ts, k=k)
        else:
            by_channel["temporal"] = []

        # Channel 6: HyDE — skip when temporal matched (we already know the
        # answer shape, no point hallucinating one).
        if use_hyde and time_range is None and self.llm is not None and self.embedder is not None:
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
        keywords = ""
        if self.llm is not None and cand.type != MemoryType.TASK.value:
            try:
                phrases = self.llm.paraphrase(cand.content)
            except Exception:
                phrases = []
            keywords = " | ".join(phrases)

        mem = Memory.new(
            profile=self.name,
            type=cand.type,
            content=cand.content,
            search_keywords=keywords,
            fact_key=cand.fact_key,
            source_session_id=session_id,
            expires_at=expires_at,
            extra={"confidence": cand.confidence, **(extra or {})},
        )
        if cand.fact_key:
            # Re-runnability: if the latest version under this fact_key already
            # has identical content, no-op. Otherwise we'd grow the chain
            # forever on repeated remember() calls with the same content.
            prior = self.storage.latest_for_factkey(self.name, cand.fact_key)
            if prior is not None and prior.content == cand.content:
                return prior
            self.storage.supersede(self.name, cand.fact_key, mem)
        else:
            self.storage.add_memory(mem)

        # Tasks are excluded from vector index — they're ephemeral.
        if mem.type != MemoryType.TASK and self.embedder is not None:
            # Embed content + paraphrases together so vector recall benefits
            # from the same query-shape augmentation as FTS does.
            embed_text = mem.content
            if mem.search_keywords:
                embed_text = f"{mem.content} {mem.search_keywords}"
            vec = self.embedder.embed(embed_text)
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

from __future__ import annotations

import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

from a5c_engram.schema import Memory, MemoryType, Message

try:
    import sqlite_vec
except ImportError:  # pragma: no cover
    sqlite_vec = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    fact_key TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    superseded_by TEXT,
    source_session_id TEXT,
    source_message_id TEXT,
    expires_at REAL,
    extra TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_profile ON memories(profile);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(profile, type);
CREATE INDEX IF NOT EXISTS idx_memories_factkey ON memories(profile, fact_key, version DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(profile, source_session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS messages (
    content_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    session_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_profile_session
    ON messages(profile, session_id, ts);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


def _f32_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class SqliteStorage:
    """Single-file SQLite store. FTS5 for keyword + raw search. sqlite-vec for
    embeddings. Supersession chains modelled with fact_key + version +
    superseded_by columns."""

    def __init__(self, path: str | Path = "a5c_engram.db", *, dim: int = 384):
        self.path = Path(path)
        self.dim = dim
        self._conn: sqlite3.Connection | None = None
        self._vec_available = False

    # ---- lifecycle -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if sqlite_vec is not None:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._vec_available = True
            except Exception:
                self._vec_available = False
        self._conn = conn
        return conn

    def init(self) -> None:
        conn = self._connect()
        conn.executescript(SCHEMA)
        if self._vec_available:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                f"  memory_id TEXT PRIMARY KEY,"
                f"  embedding float[{self.dim}]"
                f")"
            )
        conn.commit()

    # ---- profiles --------------------------------------------------
    def list_profiles(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT profile,
                   type,
                   COUNT(*) AS cnt,
                   MAX(created_at) AS last_at
            FROM memories
            GROUP BY profile, type
            """
        ).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            p = out.setdefault(
                r["profile"],
                {"name": r["profile"], "counts": {t.value: 0 for t in MemoryType},
                 "last_ingested_at": 0.0, "total": 0},
            )
            p["counts"][r["type"]] = r["cnt"]
            p["total"] += r["cnt"]
            p["last_ingested_at"] = max(p["last_ingested_at"], r["last_at"] or 0.0)
        return sorted(out.values(), key=lambda x: x["last_ingested_at"], reverse=True)

    # ---- messages --------------------------------------------------
    def add_message(self, profile: str, msg: Message) -> str:
        conn = self._connect()
        cid = msg.content_id(profile)
        conn.execute(
            "INSERT OR IGNORE INTO messages(content_id, profile, session_id, role, content, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, profile, msg.session_id, msg.role, msg.content, msg.ts),
        )
        conn.commit()
        return cid

    # ---- memories --------------------------------------------------
    def add_memory(self, mem: Memory) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, profile, type, content, created_at, fact_key, version,
             superseded_by, source_session_id, source_message_id, expires_at, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem.id, mem.profile, mem.type.value, mem.content, mem.created_at,
                mem.fact_key, mem.version, mem.superseded_by,
                mem.source_session_id, mem.source_message_id, mem.expires_at,
                json.dumps(mem.extra),
            ),
        )
        conn.commit()

    def get_memory(self, mem_id: str) -> Memory | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
        return _row_to_memory(row) if row else None

    def list_memories(
        self,
        profile: str,
        *,
        type: MemoryType | None = None,
        topic: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        conn = self._connect()
        q = "SELECT * FROM memories WHERE profile=?"
        args: list[Any] = [profile]
        if type is not None:
            q += " AND type=?"
            args.append(type.value)
        if topic is not None:
            q += " AND fact_key=?"
            args.append(topic)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        rows = conn.execute(q, args).fetchall()
        return [_row_to_memory(r) for r in rows]

    def forget(self, mem_id: str) -> bool:
        conn = self._connect()
        cur = conn.execute("DELETE FROM memories WHERE id=?", (mem_id,))
        if self._vec_available:
            conn.execute("DELETE FROM memories_vec WHERE memory_id=?", (mem_id,))
        conn.commit()
        return cur.rowcount > 0

    # ---- supersession ---------------------------------------------
    def latest_for_factkey(self, profile: str, fact_key: str) -> Memory | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memories "
            "WHERE profile=? AND fact_key=? AND superseded_by IS NULL "
            "ORDER BY version DESC LIMIT 1",
            (profile, fact_key),
        ).fetchone()
        return _row_to_memory(row) if row else None

    def supersede(self, profile: str, fact_key: str, new_mem: Memory) -> None:
        prior = self.latest_for_factkey(profile, fact_key)
        if prior is not None:
            new_mem.version = prior.version + 1
            self.add_memory(new_mem)
            conn = self._connect()
            conn.execute(
                "UPDATE memories SET superseded_by=? WHERE id=?",
                (new_mem.id, prior.id),
            )
            conn.commit()
        else:
            new_mem.version = 1
            self.add_memory(new_mem)

    def supersession_chain(self, mem_id: str) -> list[Memory]:
        conn = self._connect()
        target = self.get_memory(mem_id)
        if target is None or target.fact_key is None:
            return [target] if target else []
        rows = conn.execute(
            "SELECT * FROM memories WHERE profile=? AND fact_key=? ORDER BY version ASC",
            (target.profile, target.fact_key),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    # ---- retrieval channels ---------------------------------------
    def search_fts(self, profile: str, query: str, k: int = 10) -> list[Memory]:
        conn = self._connect()
        q = _fts_sanitize(query)
        if not q:
            return []
        rows = conn.execute(
            """
            SELECT m.* FROM memories_fts f
            JOIN memories m ON m.rowid = f.rowid
            WHERE memories_fts MATCH ? AND m.profile = ? AND m.superseded_by IS NULL
            ORDER BY bm25(memories_fts)
            LIMIT ?
            """,
            (q, profile, k),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def search_raw(self, profile: str, query: str, k: int = 10) -> list[Memory]:
        conn = self._connect()
        q = _fts_sanitize(query)
        if not q:
            return []
        rows = conn.execute(
            """
            SELECT msg.role, msg.content, msg.ts, msg.session_id, msg.content_id
            FROM messages_fts f
            JOIN messages msg ON msg.rowid = f.rowid
            WHERE messages_fts MATCH ? AND msg.profile = ?
            ORDER BY bm25(messages_fts)
            LIMIT ?
            """,
            (q, profile, k),
        ).fetchall()
        out: list[Memory] = []
        for r in rows:
            out.append(
                Memory(
                    id=f"raw:{r['content_id']}",
                    profile=profile,
                    type=MemoryType.EVENT,
                    content=f"[{r['role']}] {r['content']}",
                    created_at=r["ts"],
                    source_session_id=r["session_id"],
                    source_message_id=r["content_id"],
                    extra={"raw": True},
                )
            )
        return out

    def search_factkey(self, profile: str, topic: str) -> Memory | None:
        return self.latest_for_factkey(profile, topic)

    def search_vector(
        self, profile: str, embedding: list[float], k: int = 10
    ) -> list[Memory]:
        if not self._vec_available:
            return []
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT m.* FROM memories_vec v
            JOIN memories m ON m.id = v.memory_id
            WHERE v.embedding MATCH ? AND k = ? AND m.profile = ?
              AND m.superseded_by IS NULL
            ORDER BY v.distance
            """,
            (_f32_blob(embedding), k, profile),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def upsert_embedding(self, mem_id: str, embedding: list[float]) -> None:
        if not self._vec_available:
            return
        conn = self._connect()
        conn.execute("DELETE FROM memories_vec WHERE memory_id=?", (mem_id,))
        conn.execute(
            "INSERT INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
            (mem_id, _f32_blob(embedding)),
        )
        conn.commit()

    # ---- ingest log -----------------------------------------------
    def list_sessions(self, profile: str) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT session_id,
                   COUNT(*) AS message_count,
                   MIN(ts) AS first_ts,
                   MAX(ts) AS last_ts
            FROM messages
            WHERE profile = ?
            GROUP BY session_id
            ORDER BY last_ts DESC
            """,
            (profile,),
        ).fetchall()
        out = []
        for r in rows:
            mem_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE profile=? AND source_session_id=?",
                (profile, r["session_id"]),
            ).fetchone()[0]
            out.append({
                "session_id": r["session_id"],
                "message_count": r["message_count"],
                "first_ts": r["first_ts"],
                "last_ts": r["last_ts"],
                "memory_count": mem_count,
            })
        return out

    def memories_for_session(self, profile: str, session_id: str) -> list[Memory]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM memories WHERE profile=? AND source_session_id=? "
            "ORDER BY created_at ASC",
            (profile, session_id),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]


def _row_to_memory(row: sqlite3.Row | None) -> Memory:
    assert row is not None
    return Memory(
        id=row["id"],
        profile=row["profile"],
        type=MemoryType(row["type"]),
        content=row["content"],
        created_at=row["created_at"],
        fact_key=row["fact_key"],
        version=row["version"],
        superseded_by=row["superseded_by"],
        source_session_id=row["source_session_id"],
        source_message_id=row["source_message_id"],
        expires_at=row["expires_at"],
        extra=json.loads(row["extra"]) if row["extra"] else {},
    )


_FTS_RESERVED = {"AND", "OR", "NOT", "NEAR"}


def _fts_sanitize(q: str) -> str:
    """Strip FTS5 operators from user input — we want plain bag-of-words match,
    not query DSL. We keep only [A-Za-z0-9_] within tokens and quote each
    token as a literal phrase to neutralise any remaining FTS metacharacters
    or reserved words."""
    out: list[str] = []
    cur: list[str] = []
    for ch in q:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    # Drop FTS5 reserved words even though the quoting below would handle
    # them — keeps the query smaller and avoids hitting bm25 with noise.
    toks = [t for t in out if t.upper() not in _FTS_RESERVED]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks)


# Local imports to keep module standalone
_ = time  # silence pyflakes if unused

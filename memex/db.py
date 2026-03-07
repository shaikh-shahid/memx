import json
import sqlite3
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None


@dataclass
class Memory:
    id: str
    fact: str
    scope: str
    source: str
    created_at: int
    expires_at: Optional[int]
    tags: list[str]
    source_group_id: Optional[str] = None
    memory_kind: str = "fact"
    source_label: Optional[str] = None


def get_connection(db_path: Path) -> sqlite3.Connection:
    if sqlite_vec is None:
        raise RuntimeError("sqlite-vec is not installed. Install with: pip install sqlite-vec")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except sqlite3.Error as e:
        raise RuntimeError(
            "Failed to load sqlite-vec extension. Ensure sqlite-vec is installed and compatible with your SQLite build."
        ) from e

    return conn


def init_db(conn: sqlite3.Connection, embedding_dim: int = 768) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            fact        TEXT NOT NULL,
            scope       TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            expires_at  INTEGER,
            tags        TEXT DEFAULT '[]',
            source_group_id TEXT,
            memory_kind TEXT NOT NULL DEFAULT 'fact',
            source_label TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vss USING vec0(
            memory_id   TEXT,
            embedding   float[{embedding_dim}]
        );
    """
    )
    _ensure_column(conn, "memories", "source_group_id", "TEXT")
    _ensure_column(conn, "memories", "memory_kind", "TEXT NOT NULL DEFAULT 'fact'")
    _ensure_column(conn, "memories", "source_label", "TEXT")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_sql: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")


def _active_memory_id(conn: sqlite3.Connection, fact: str, scope: str) -> Optional[str]:
    now = int(time.time())
    row = conn.execute(
        """
        SELECT id
        FROM memories
        WHERE scope = ?
          AND fact = ?
          AND (expires_at IS NULL OR expires_at > ?)
        LIMIT 1
        """,
        (scope, fact, now),
    ).fetchone()
    return str(row["id"]) if row else None


def has_active_memory(conn: sqlite3.Connection, fact: str, scope: str) -> bool:
    return _active_memory_id(conn, fact, scope) is not None


def store_memory(
    conn: sqlite3.Connection,
    fact: str,
    embedding: list[float],
    scope: str,
    source: str,
    tags: Optional[list[str]] = None,
    expires_at: Optional[int] = None,
    *,
    dedupe: bool = True,
    commit: bool = True,
    source_group_id: Optional[str] = None,
    memory_kind: str = "fact",
    source_label: Optional[str] = None,
) -> str:
    if not embedding:
        raise ValueError("Embedding is empty.")
    if not fact.strip():
        raise ValueError("Fact is empty.")

    if dedupe:
        existing_id = _active_memory_id(conn, fact.strip(), scope)
        if existing_id:
            return existing_id

    tag_list = tags or []
    memory_id = str(uuid.uuid4())
    now = int(time.time())

    conn.execute(
        """
        INSERT INTO memories (
            id, fact, scope, source, created_at, expires_at, tags, source_group_id, memory_kind, source_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            fact.strip(),
            scope,
            source,
            now,
            expires_at,
            json.dumps(tag_list),
            source_group_id,
            memory_kind,
            source_label,
        ),
    )

    embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
    conn.execute(
        "INSERT INTO memory_vss (memory_id, embedding) VALUES (?, ?)",
        (memory_id, embedding_blob),
    )
    if commit:
        conn.commit()
    return memory_id


def get_all_memories(
    conn: sqlite3.Connection,
    scope: Optional[str] = None,
    include_expired: bool = False,
    source_label: Optional[str] = None,
) -> list[Memory]:
    now = int(time.time())
    query = "SELECT * FROM memories WHERE 1=1"
    params: list[object] = []

    if scope:
        query += " AND scope = ?"
        params.append(scope)
    if source_label:
        query += " AND source_label = ?"
        params.append(source_label)

    if not include_expired:
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(now)

    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [_row_to_memory(r) for r in rows]


def delete_memory(conn: sqlite3.Connection, memory_id: str) -> bool:
    cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.execute("DELETE FROM memory_vss WHERE memory_id = ?", (memory_id,))
    conn.commit()
    return cursor.rowcount > 0


def delete_expired(conn: sqlite3.Connection) -> int:
    now = int(time.time())
    cursor = conn.execute(
        "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
    )
    conn.execute(
        """
        DELETE FROM memory_vss
        WHERE memory_id NOT IN (SELECT id FROM memories)
        """
    )
    conn.commit()
    return cursor.rowcount


def delete_all_memories(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("DELETE FROM memories")
    conn.execute("DELETE FROM memory_vss")
    conn.commit()
    return cursor.rowcount


def get_memory_by_id(conn: sqlite3.Connection, memory_id: str) -> Optional[Memory]:
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return _row_to_memory(row) if row else None


def _row_to_memory(row: sqlite3.Row) -> Memory:
    tags: list[str]
    try:
        loaded = json.loads(row["tags"]) if row["tags"] else []
        tags = loaded if isinstance(loaded, list) else []
    except json.JSONDecodeError:
        tags = []

    return Memory(
        id=row["id"],
        fact=row["fact"],
        scope=row["scope"],
        source=row["source"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        tags=tags,
        source_group_id=row["source_group_id"] if "source_group_id" in row.keys() else None,
        memory_kind=row["memory_kind"] if "memory_kind" in row.keys() else "fact",
        source_label=row["source_label"] if "source_label" in row.keys() else None,
    )

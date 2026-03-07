import json
import sqlite3
import struct
import time
from typing import Optional

from memex.db import Memory


def search_memories(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    scope: Optional[str] = None,
    limit: int = 5,
) -> list[tuple[Memory, float]]:
    if limit <= 0:
        return []

    now = int(time.time())
    embedding_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)

    query = """
        SELECT
            m.*,
            v.distance
        FROM memory_vss v
        JOIN memories m ON m.id = v.memory_id
        WHERE v.embedding MATCH ?
          AND k = ?
          AND (m.expires_at IS NULL OR m.expires_at > ?)
    """
    params: list[object] = [embedding_blob, limit * 3, now]

    if scope:
        query += " AND m.scope = ?"
        params.append(scope)

    query += " ORDER BY v.distance ASC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()

    results: list[tuple[Memory, float]] = []
    for row in rows:
        loaded_tags = json.loads(row["tags"]) if row["tags"] else []
        tags = loaded_tags if isinstance(loaded_tags, list) else []
        memory = Memory(
            id=row["id"],
            fact=row["fact"],
            scope=row["scope"],
            source=row["source"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            tags=tags,
        )
        distance = float(row["distance"])
        score = round(1.0 / (1.0 + distance), 4)
        results.append((memory, score))

    return results

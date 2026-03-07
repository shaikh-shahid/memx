# memex — Full Build Prompt

> Pass this entire file to Codex or Cursor as your build specification.
> Do not skip any section. Every detail here is intentional.

---

## What You Are Building

A lightweight, local-first memory layer called **memex**. It runs silently on the developer's machine, stores atomic facts extracted from conversations and notes, and exposes those facts to any MCP-compatible AI tool (Cursor, Claude Desktop, etc.). There is no web UI, no cloud sync, no Docker, no API keys. Everything runs locally via Ollama.

The system has two entry points:
- `memex` — a CLI for human interaction
- `memex-server` — an MCP server for AI tool integration

Both share the exact same core engine.

---

## Hard Constraints

- **Python 3.11+ only.** Use `tomllib` from stdlib (no `tomli` package). Use `match` statements where appropriate.
- **No heavy dependencies.** No PyTorch, no LangChain, no Transformers, no vector DB daemons.
- **Ollama is the only AI backend.** All LLM and embedding calls go to `http://localhost:11434` via plain HTTP. No OpenAI, no Anthropic API.
- **SQLite is the only database.** One file at `~/.memex/memory.db`. Use `sqlite-vec` for vector search.
- **MCP transport is stdio only.** No HTTP server, no SSE, no WebSocket.
- **Every Ollama call must have a timeout and graceful error.** If Ollama is not running, memex must fail with a clear human-readable message, not a stack trace.
- **All user-facing errors must be clean.** Use `typer.echo` with `err=True` and `raise typer.Exit(1)`. Never let raw exceptions surface to the user.

---

## Project Structure

Create exactly this directory layout. Do not add extra files:

```
memex/
├── memex/
│   ├── __init__.py          # version = "0.1.0", nothing else
│   ├── config.py            # load/save TOML config
│   ├── db.py                # schema, migrations, CRUD
│   ├── embeddings.py        # Ollama nomic-embed-text wrapper
│   ├── extraction.py        # Ollama llama3.2:3b fact extraction
│   └── search.py            # cosine search via sqlite-vec
├── cli.py                   # Typer CLI → `memex` command
├── server.py                # MCP server → `memex-server` command
├── config.toml              # default config file (shipped with package)
├── pyproject.toml
└── README.md
```

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "memex"
version = "0.1.0"
description = "Local memory layer for AI tools via MCP"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12.0",
    "mcp>=1.0.0",
    "sqlite-vec>=0.1.6",
    "httpx>=0.27.0",
    "pyperclip>=1.8.2",
]

[project.scripts]
memex = "cli:app"
memex-server = "server:main"

[tool.hatch.build.targets.wheel]
packages = ["memex"]
```

---

## config.toml (default, shipped with package)

```toml
[ollama]
base_url = "http://localhost:11434"
embed_model = "nomic-embed-text"
extract_model = "llama3.2:3b"
timeout_seconds = 30

[storage]
db_path = "~/.memex/memory.db"

[memory]
extract_threshold_chars = 300
default_scope = "self"
project_ttl_days = 30
```

---

## memex/config.py

Load config from two places in order (second overrides first):
1. The `config.toml` shipped with the package (defaults)
2. `~/.memex/config.toml` (user overrides, created on first run if missing)

```python
import tomllib
import shutil
from pathlib import Path
from dataclasses import dataclass

PACKAGE_CONFIG = Path(__file__).parent.parent / "config.toml"
USER_CONFIG_DIR = Path.home() / ".memex"
USER_CONFIG = USER_CONFIG_DIR / "config.toml"


@dataclass
class OllamaConfig:
    base_url: str
    embed_model: str
    extract_model: str
    timeout_seconds: int


@dataclass
class StorageConfig:
    db_path: Path


@dataclass
class MemoryConfig:
    extract_threshold_chars: int
    default_scope: str
    project_ttl_days: int


@dataclass
class Config:
    ollama: OllamaConfig
    storage: StorageConfig
    memory: MemoryConfig


def load_config() -> Config:
    # Load defaults
    with open(PACKAGE_CONFIG, "rb") as f:
        data = tomllib.load(f)

    # Ensure user dir exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Copy default config to user dir if not present
    if not USER_CONFIG.exists():
        shutil.copy(PACKAGE_CONFIG, USER_CONFIG)

    # Load user overrides and merge
    if USER_CONFIG.exists():
        with open(USER_CONFIG, "rb") as f:
            user_data = tomllib.load(f)
        # Deep merge user_data into data
        for section, values in user_data.items():
            if section in data and isinstance(values, dict):
                data[section].update(values)

    return Config(
        ollama=OllamaConfig(**data["ollama"]),
        storage=StorageConfig(db_path=Path(data["storage"]["db_path"]).expanduser()),
        memory=MemoryConfig(**data["memory"]),
    )
```

---

## memex/db.py

### Schema

Two tables only. Do not add more.

```sql
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    fact        TEXT NOT NULL,
    scope       TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER,
    tags        TEXT DEFAULT '[]'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_vss USING vec0(
    memory_id   TEXT,
    embedding   float[768]
);
```

### Full Implementation

```python
import sqlite3
import sqlite_vec
import uuid
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Memory:
    id: str
    fact: str
    scope: str
    source: str
    created_at: int
    expires_at: Optional[int]
    tags: list[str]


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            fact        TEXT NOT NULL,
            scope       TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  INTEGER NOT NULL,
            expires_at  INTEGER,
            tags        TEXT DEFAULT '[]'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vss USING vec0(
            memory_id   TEXT,
            embedding   float[768]
        );
    """)
    conn.commit()


def store_memory(
    conn: sqlite3.Connection,
    fact: str,
    embedding: list[float],
    scope: str,
    source: str,
    tags: list[str] = [],
    expires_at: Optional[int] = None,
) -> str:
    memory_id = str(uuid.uuid4())
    now = int(time.time())

    conn.execute(
        """
        INSERT INTO memories (id, fact, scope, source, created_at, expires_at, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, fact, scope, source, now, expires_at, json.dumps(tags)),
    )

    import struct
    embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
    conn.execute(
        "INSERT INTO memory_vss (memory_id, embedding) VALUES (?, ?)",
        (memory_id, embedding_blob),
    )
    conn.commit()
    return memory_id


def get_all_memories(
    conn: sqlite3.Connection,
    scope: Optional[str] = None,
    include_expired: bool = False,
) -> list[Memory]:
    now = int(time.time())
    query = "SELECT * FROM memories WHERE 1=1"
    params = []

    if scope:
        query += " AND scope = ?"
        params.append(scope)

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
    # Clean up orphaned vectors
    conn.execute(
        """
        DELETE FROM memory_vss
        WHERE memory_id NOT IN (SELECT id FROM memories)
        """
    )
    conn.commit()
    return cursor.rowcount


def get_memory_by_id(conn: sqlite3.Connection, memory_id: str) -> Optional[Memory]:
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return _row_to_memory(row) if row else None


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        fact=row["fact"],
        scope=row["scope"],
        source=row["source"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        tags=json.loads(row["tags"]),
    )
```

---

## memex/embeddings.py

```python
import httpx
import json
from typing import Any


class OllamaEmbeddingError(Exception):
    pass


def embed_text(text: str, model: str, base_url: str, timeout: int) -> list[float]:
    """
    Call Ollama embeddings API. Returns float[768] for nomic-embed-text.
    Raises OllamaEmbeddingError with a clean message if anything fails.
    """
    url = f"{base_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if "embedding" not in data:
            raise OllamaEmbeddingError(
                f"Ollama returned unexpected response: {data}"
            )

        return data["embedding"]

    except httpx.ConnectError:
        raise OllamaEmbeddingError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is Ollama running? Try: ollama serve"
        )
    except httpx.TimeoutException:
        raise OllamaEmbeddingError(
            f"Ollama timed out after {timeout}s. "
            "Try increasing timeout_seconds in ~/.memex/config.toml"
        )
    except httpx.HTTPStatusError as e:
        raise OllamaEmbeddingError(
            f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
        )
```

---

## memex/extraction.py

```python
import httpx
import json
import re

EXTRACTION_PROMPT = """You are a memory extraction assistant. Your job is to extract atomic, self-contained facts from the text below.

Rules:
- Each fact must make complete sense on its own, without any surrounding context
- Write facts as statements, not questions
- Be specific. "Uses Supabase for auth" is good. "Uses a database" is too vague.
- Do not summarize. Extract distinct facts.
- Ignore small talk, greetings, filler content
- Maximum 15 facts per extraction
- If the text contains no meaningful facts worth remembering, return an empty array

Return ONLY a JSON array of strings. No explanation, no markdown, no preamble.

Example output:
["Prefers TypeScript over JavaScript", "Project is deployed on Railway", "Using Supabase for auth, not Clerk"]

Text to extract from:
{text}"""


class OllamaExtractionError(Exception):
    pass


def extract_facts(text: str, model: str, base_url: str, timeout: int) -> list[str]:
    """
    Use Ollama to extract atomic facts from raw text.
    Returns a list of fact strings.
    Raises OllamaExtractionError with a clean message if anything fails.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    prompt = EXTRACTION_PROMPT.format(text=text.strip())

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temperature — we want deterministic extraction
            "top_p": 0.9,
        },
    }

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "").strip()

        # Parse JSON from response — strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("```").strip()
        facts = json.loads(cleaned)

        if not isinstance(facts, list):
            raise OllamaExtractionError(
                f"Expected a JSON array, got: {type(facts).__name__}"
            )

        # Filter: only non-empty strings
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    except json.JSONDecodeError as e:
        raise OllamaExtractionError(
            f"Could not parse extraction response as JSON: {e}\nRaw: {raw_text[:200]}"
        )
    except httpx.ConnectError:
        raise OllamaExtractionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is Ollama running? Try: ollama serve"
        )
    except httpx.TimeoutException:
        raise OllamaExtractionError(
            f"Ollama timed out after {timeout}s during extraction."
        )
    except httpx.HTTPStatusError as e:
        raise OllamaExtractionError(
            f"Ollama HTTP error {e.response.status_code}: {e.response.text}"
        )
```

---

## memex/search.py

```python
import sqlite3
import struct
from typing import Optional
from memex.db import Memory, _row_to_memory


def search_memories(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    scope: Optional[str] = None,
    limit: int = 5,
) -> list[tuple[Memory, float]]:
    """
    Cosine similarity search using sqlite-vec.
    Returns list of (Memory, score) tuples, ordered by relevance descending.
    Filters out expired memories.
    """
    import time
    now = int(time.time())

    embedding_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)

    # Search in vector table, join with memories for filtering
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
    params: list = [embedding_blob, limit * 3, now]  # Fetch 3x to allow scope filter

    if scope:
        query += " AND m.scope = ?"
        params.append(scope)

    query += " ORDER BY v.distance ASC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        memory = Memory(
            id=row["id"],
            fact=row["fact"],
            scope=row["scope"],
            source=row["source"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            tags=__import__("json").loads(row["tags"]),
        )
        # sqlite-vec returns L2 distance; convert to a 0-1 similarity score
        distance = row["distance"]
        score = 1.0 / (1.0 + distance)
        results.append((memory, round(score, 4)))

    return results
```

---

## cli.py

This is the full Typer CLI. Implement every command exactly as specified.

```python
import typer
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

app = typer.Typer(
    name="memex",
    help="Local memory layer for AI tools.",
    no_args_is_help=True,
    add_completion=False,
)


def _get_deps():
    """Lazy load heavy deps — keeps CLI startup fast."""
    from memex.config import load_config
    from memex.db import get_connection, init_db
    return load_config, get_connection, init_db


def _validate_scope(scope: str) -> str:
    """Scope must be 'self' or start with 'project:'."""
    if scope == "self" or scope.startswith("project:"):
        return scope
    typer.echo(
        f"[error] Invalid scope '{scope}'. Must be 'self' or 'project:<name>'.",
        err=True,
    )
    raise typer.Exit(1)


def _compute_expires_at(scope: str, ttl_days: int) -> Optional[int]:
    if scope == "self":
        return None
    return int(time.time()) + (ttl_days * 86400)


@app.command()
def add(
    fact: str = typer.Argument(..., help="The fact to store"),
    scope: str = typer.Option("self", "--scope", "-s", help="'self' or 'project:<name>'"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Add a single fact directly to memory."""
    from memex.config import load_config
    from memex.db import get_connection, init_db, store_memory
    from memex.embeddings import embed_text, OllamaEmbeddingError

    _validate_scope(scope)
    cfg = load_config()

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    expires_at = _compute_expires_at(scope, cfg.memory.project_ttl_days)

    try:
        embedding = embed_text(
            fact,
            model=cfg.ollama.embed_model,
            base_url=cfg.ollama.base_url,
            timeout=cfg.ollama.timeout_seconds,
        )
    except OllamaEmbeddingError as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    conn = get_connection(cfg.storage.db_path)
    init_db(conn)
    memory_id = store_memory(
        conn,
        fact=fact,
        embedding=embedding,
        scope=scope,
        source="cli",
        tags=tag_list,
        expires_at=expires_at,
    )
    typer.echo(f"✓ Stored [{memory_id[:8]}] → {fact}")


@app.command(name="from")
def from_input(
    file: Optional[str] = typer.Argument(None, help="File path or '-' for stdin"),
    scope: str = typer.Option("self", "--scope", "-s", help="'self' or 'project:<name>'"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show extracted facts without storing"),
):
    """Extract facts from a file or stdin and store them."""
    from memex.config import load_config
    from memex.db import get_connection, init_db, store_memory
    from memex.embeddings import embed_text, OllamaEmbeddingError
    from memex.extraction import extract_facts, OllamaExtractionError

    _validate_scope(scope)
    cfg = load_config()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    # Read input
    if file is None or file == "-":
        if sys.stdin.isatty():
            typer.echo("[error] No input. Pipe text or provide a file path.", err=True)
            raise typer.Exit(1)
        text = sys.stdin.read()
    else:
        path = Path(file)
        if not path.exists():
            typer.echo(f"[error] File not found: {file}", err=True)
            raise typer.Exit(1)
        text = path.read_text(encoding="utf-8")

    if not text.strip():
        typer.echo("[error] Input is empty.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracting facts from {len(text)} chars of text...")

    try:
        facts = extract_facts(
            text,
            model=cfg.ollama.extract_model,
            base_url=cfg.ollama.base_url,
            timeout=cfg.ollama.timeout_seconds,
        )
    except OllamaExtractionError as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    if not facts:
        typer.echo("No memorable facts found in this text.")
        return

    typer.echo(f"\nExtracted {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        typer.echo(f"  {i}. {fact}")

    if dry_run:
        typer.echo("\n[dry-run] Nothing stored.")
        return

    typer.echo("\nStoring...")
    conn = get_connection(cfg.storage.db_path)
    init_db(conn)
    expires_at = _compute_expires_at(scope, cfg.memory.project_ttl_days)

    stored = 0
    for fact in facts:
        try:
            embedding = embed_text(
                fact,
                model=cfg.ollama.embed_model,
                base_url=cfg.ollama.base_url,
                timeout=cfg.ollama.timeout_seconds,
            )
            store_memory(
                conn,
                fact=fact,
                embedding=embedding,
                scope=scope,
                source="extract",
                tags=tag_list,
                expires_at=expires_at,
            )
            stored += 1
        except OllamaEmbeddingError as e:
            typer.echo(f"  [skip] Embedding failed for fact: {e}", err=True)

    typer.echo(f"\n✓ Stored {stored}/{len(facts)} facts to scope '{scope}'")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Filter by scope"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
):
    """Search memories semantically."""
    from memex.config import load_config
    from memex.db import get_connection, init_db
    from memex.embeddings import embed_text, OllamaEmbeddingError
    from memex.search import search_memories

    if scope:
        _validate_scope(scope)

    cfg = load_config()

    try:
        query_embedding = embed_text(
            query,
            model=cfg.ollama.embed_model,
            base_url=cfg.ollama.base_url,
            timeout=cfg.ollama.timeout_seconds,
        )
    except OllamaEmbeddingError as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    conn = get_connection(cfg.storage.db_path)
    init_db(conn)
    results = search_memories(conn, query_embedding, scope=scope, limit=limit)

    if not results:
        typer.echo("No memories found.")
        return

    typer.echo(f"\nTop {len(results)} results for: \"{query}\"\n")
    for memory, score in results:
        typer.echo(f"  [{memory.id[:8]}] ({memory.scope}) score={score}")
        typer.echo(f"  {memory.fact}\n")


@app.command(name="list")
def list_memories(
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Filter by scope"),
    show_expired: bool = typer.Option(False, "--expired", help="Include expired memories"),
):
    """List all stored memories."""
    from memex.config import load_config
    from memex.db import get_connection, init_db, get_all_memories

    if scope:
        _validate_scope(scope)

    cfg = load_config()
    conn = get_connection(cfg.storage.db_path)
    init_db(conn)
    memories = get_all_memories(conn, scope=scope, include_expired=show_expired)

    if not memories:
        typer.echo("No memories found.")
        return

    # Group by scope
    grouped: dict[str, list] = {}
    for m in memories:
        grouped.setdefault(m.scope, []).append(m)

    for scope_name, items in grouped.items():
        typer.echo(f"\n── {scope_name} ({len(items)}) ──")
        for m in items:
            expires = (
                datetime.fromtimestamp(m.expires_at, tz=timezone.utc).strftime("%Y-%m-%d")
                if m.expires_at
                else "never"
            )
            typer.echo(f"  [{m.id[:8]}] {m.fact}  (expires: {expires})")


@app.command()
def forget(
    memory_id: str = typer.Argument(..., help="Memory ID or first 8 chars of ID"),
):
    """Delete a specific memory by ID."""
    from memex.config import load_config
    from memex.db import get_connection, init_db, delete_memory, get_all_memories

    cfg = load_config()
    conn = get_connection(cfg.storage.db_path)
    init_db(conn)

    # Support partial ID (first 8 chars)
    if len(memory_id) == 8:
        all_memories = get_all_memories(conn, include_expired=True)
        matches = [m for m in all_memories if m.id.startswith(memory_id)]
        if len(matches) == 0:
            typer.echo(f"[error] No memory found with ID starting with '{memory_id}'", err=True)
            raise typer.Exit(1)
        if len(matches) > 1:
            typer.echo(f"[error] Ambiguous ID prefix '{memory_id}' matches {len(matches)} memories. Use full ID.", err=True)
            raise typer.Exit(1)
        memory_id = matches[0].id

    deleted = delete_memory(conn, memory_id)
    if deleted:
        typer.echo(f"✓ Deleted memory {memory_id[:8]}")
    else:
        typer.echo(f"[error] Memory not found: {memory_id}", err=True)
        raise typer.Exit(1)


@app.command()
def dump(
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Scope to dump"),
    no_copy: bool = typer.Option(False, "--no-copy", help="Print only, don't copy to clipboard"),
):
    """Dump memory context block to clipboard (for pasting into ChatGPT etc.)."""
    from memex.config import load_config
    from memex.db import get_connection, init_db, get_all_memories
    from datetime import datetime, timezone

    if scope:
        _validate_scope(scope)

    cfg = load_config()
    conn = get_connection(cfg.storage.db_path)
    init_db(conn)
    memories = get_all_memories(conn, scope=scope)

    if not memories:
        typer.echo("No memories to dump.")
        return

    # Group by scope
    grouped: dict[str, list] = {}
    for m in memories:
        grouped.setdefault(m.scope, []).append(m)

    lines = ["--- MEMEX CONTEXT ---"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    for scope_name, items in grouped.items():
        lines.append(f"[{scope_name}]")
        for m in items:
            lines.append(f"- {m.fact}")
        lines.append("")

    lines.append("--- END CONTEXT ---")
    output = "\n".join(lines)

    typer.echo(output)

    if not no_copy:
        try:
            import pyperclip
            pyperclip.copy(output)
            typer.echo("\n✓ Copied to clipboard.")
        except Exception:
            typer.echo("\n[warn] Could not copy to clipboard. Install xclip/xsel on Linux.", err=True)


@app.command()
def info():
    """Show memory statistics and config."""
    from memex.config import load_config
    from memex.db import get_connection, init_db, get_all_memories, delete_expired

    cfg = load_config()
    conn = get_connection(cfg.storage.db_path)
    init_db(conn)

    # Clean expired first
    expired_count = delete_expired(conn)

    memories = get_all_memories(conn, include_expired=False)
    scopes = {}
    for m in memories:
        scopes[m.scope] = scopes.get(m.scope, 0) + 1

    typer.echo(f"\nmemex info")
    typer.echo(f"  DB path:      {cfg.storage.db_path}")
    typer.echo(f"  Total facts:  {len(memories)}")
    typer.echo(f"  Expired (cleaned): {expired_count}")
    typer.echo(f"  Ollama:       {cfg.ollama.base_url}")
    typer.echo(f"  Embed model:  {cfg.ollama.embed_model}")
    typer.echo(f"  Extract model:{cfg.ollama.extract_model}")
    typer.echo(f"\n  Scopes:")
    for scope_name, count in sorted(scopes.items()):
        typer.echo(f"    {scope_name}: {count} facts")


if __name__ == "__main__":
    app()
```

---

## server.py

This is the MCP server. It exposes exactly 4 tools. Use the official `mcp` Python SDK. Transport is stdio only.

```python
import asyncio
import sys
import time
from typing import Optional
import mcp.server.stdio
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp import types

# Load config at startup
from memex.config import load_config
from memex.db import get_connection, init_db, store_memory, get_all_memories, delete_memory
from memex.embeddings import embed_text, OllamaEmbeddingError
from memex.extraction import extract_facts, OllamaExtractionError
from memex.search import search_memories

server = Server("memex")
_cfg = None
_conn = None


def get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def get_conn():
    global _conn
    if _conn is None:
        cfg = get_cfg()
        _conn = get_connection(cfg.storage.db_path)
        init_db(_conn)
    return _conn


def _compute_expires_at(scope: str) -> Optional[int]:
    cfg = get_cfg()
    if scope == "self":
        return None
    return int(time.time()) + (cfg.memory.project_ttl_days * 86400)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="mem_store",
            description=(
                "Store information in persistent local memory. "
                "If the text is long (a conversation, notes, research), it will be automatically "
                "extracted into atomic facts before storing. Short direct facts are stored as-is. "
                "Use scope='self' for personal preferences and permanent info. "
                "Use scope='project:<name>' for project-specific context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text or fact to store. Can be a single sentence or a long conversation dump.",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Memory scope. 'self' for permanent personal memory, 'project:<name>' for project context.",
                        "default": "self",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="mem_search",
            description=(
                "Search memory semantically. Returns the most relevant stored facts for a given query. "
                "Use this at the start of a session or when you need context about preferences, past decisions, or project state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for relevant memories.",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional. Filter results to a specific scope.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return. Default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="mem_list",
            description="List all stored memories, optionally filtered by scope.",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Optional scope filter ('self' or 'project:<name>').",
                    },
                },
            },
        ),
        types.Tool(
            name="mem_forget",
            description="Delete a specific memory by its ID. Use mem_list to find the ID first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The full memory ID (UUID) to delete.",
                    },
                },
                "required": ["id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    cfg = get_cfg()
    conn = get_conn()

    match name:
        case "mem_store":
            text = arguments["text"]
            scope = arguments.get("scope", cfg.memory.default_scope)

            # Validate scope
            if scope != "self" and not scope.startswith("project:"):
                return [types.TextContent(
                    type="text",
                    text=f"Error: Invalid scope '{scope}'. Must be 'self' or 'project:<name>'."
                )]

            expires_at = _compute_expires_at(scope)

            # Decide: extract or store directly
            if len(text) > cfg.memory.extract_threshold_chars:
                try:
                    facts = await asyncio.to_thread(
                        extract_facts,
                        text,
                        model=cfg.ollama.extract_model,
                        base_url=cfg.ollama.base_url,
                        timeout=cfg.ollama.timeout_seconds,
                    )
                except OllamaExtractionError as e:
                    return [types.TextContent(type="text", text=f"Extraction error: {e}")]
            else:
                facts = [text.strip()]

            if not facts:
                return [types.TextContent(type="text", text="No memorable facts found in this text.")]

            stored_ids = []
            failed = 0
            for fact in facts:
                try:
                    embedding = await asyncio.to_thread(
                        embed_text,
                        fact,
                        model=cfg.ollama.embed_model,
                        base_url=cfg.ollama.base_url,
                        timeout=cfg.ollama.timeout_seconds,
                    )
                    memory_id = store_memory(
                        conn,
                        fact=fact,
                        embedding=embedding,
                        scope=scope,
                        source="mcp",
                        expires_at=expires_at,
                    )
                    stored_ids.append((memory_id[:8], fact))
                except OllamaEmbeddingError:
                    failed += 1

            lines = [f"Stored {len(stored_ids)} fact(s) to scope '{scope}':"]
            for short_id, fact in stored_ids:
                lines.append(f"  [{short_id}] {fact}")
            if failed:
                lines.append(f"  ({failed} facts failed to embed and were skipped)")

            return [types.TextContent(type="text", text="\n".join(lines))]

        case "mem_search":
            query = arguments["query"]
            scope = arguments.get("scope")
            limit = int(arguments.get("limit", 5))

            try:
                query_embedding = await asyncio.to_thread(
                    embed_text,
                    query,
                    model=cfg.ollama.embed_model,
                    base_url=cfg.ollama.base_url,
                    timeout=cfg.ollama.timeout_seconds,
                )
            except OllamaEmbeddingError as e:
                return [types.TextContent(type="text", text=f"Embedding error: {e}")]

            results = search_memories(conn, query_embedding, scope=scope, limit=limit)

            if not results:
                return [types.TextContent(type="text", text="No relevant memories found.")]

            lines = [f"Found {len(results)} relevant memories:\n"]
            for memory, score in results:
                lines.append(f"[{memory.scope}] {memory.fact}  (relevance: {score})")

            return [types.TextContent(type="text", text="\n".join(lines))]

        case "mem_list":
            scope = arguments.get("scope")
            memories = get_all_memories(conn, scope=scope)

            if not memories:
                return [types.TextContent(type="text", text="No memories stored.")]

            grouped: dict[str, list] = {}
            for m in memories:
                grouped.setdefault(m.scope, []).append(m)

            lines = []
            for scope_name, items in grouped.items():
                lines.append(f"\n[{scope_name}] — {len(items)} facts")
                for m in items:
                    lines.append(f"  [{m.id[:8]}] {m.fact}")

            return [types.TextContent(type="text", text="\n".join(lines))]

        case "mem_forget":
            memory_id = arguments["id"]
            deleted = delete_memory(conn, memory_id)
            if deleted:
                return [types.TextContent(type="text", text=f"Deleted memory {memory_id[:8]}.")]
            else:
                return [types.TextContent(type="text", text=f"Memory not found: {memory_id}")]

        case _:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


def main():
    async def run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memex",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
```

---

## MCP Client Configuration

### Claude Desktop
File: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
File: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "memex": {
      "command": "memex-server"
    }
  }
}
```

### Cursor (global, all projects)
File: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "memex": {
      "command": "memex-server"
    }
  }
}
```

### Windsurf
File: `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "memex": {
      "command": "memex-server"
    }
  }
}
```

---

## README.md

Write a README with these exact sections:

1. **What it is** — one paragraph, no fluff
2. **Prerequisites** — Python 3.11+, Ollama running with `nomic-embed-text` and `llama3.2:3b` pulled
3. **Install** — `pip install -e .` from repo root
4. **Ollama setup** — `ollama pull nomic-embed-text && ollama pull llama3.2:3b`
5. **MCP setup** — the three JSON config blocks above (Claude Desktop, Cursor, Windsurf)
6. **CLI reference** — table of all commands with one-line descriptions
7. **How scopes work** — `self` vs `project:<name>`, TTL behaviour
8. **Config** — explain `~/.memex/config.toml` and every key
9. **Typical workflow** — 3 real scenarios: end-of-session save, new session start in Cursor, dumping for ChatGPT

---

## Behaviour Contracts (Implement Exactly)

### Scope validation
- Valid scopes: `"self"`, `"project:anything-here"` (alphanumeric, hyphens, underscores)
- Invalid scopes must produce a clean error immediately, never reach storage

### Extraction threshold
- If `len(text) > config.memory.extract_threshold_chars` (default 300): run extraction
- If text is short: store directly, no extraction call
- This applies in BOTH CLI and MCP server

### Expiry
- `scope == "self"` → `expires_at = NULL` (never expires)
- Any `project:*` scope → `expires_at = now + (project_ttl_days * 86400)`
- `memex info` and `memex list` always clean expired memories before displaying

### Ollama errors
- If Ollama is not reachable: print `Cannot connect to Ollama at <url>. Is Ollama running? Try: ollama serve`
- If model not found (404): print `Model '<model>' not found in Ollama. Run: ollama pull <model>`
- Never show a raw Python traceback to the user in CLI mode
- In MCP mode: return the error as a TextContent response, never crash the server

### MCP server stability
- The MCP server must never crash due to a bad tool call. Wrap every tool handler in try/except.
- Ollama not running must return a helpful TextContent error, not kill the server process.
- The server process must stay alive even after individual tool call failures.

### sqlite-vec
- Load the extension immediately on every `get_connection()` call
- If `sqlite-vec` is not installed: print a clear install instruction and exit 1
- Embedding dimension is always 768 (nomic-embed-text). Do not hardcode this elsewhere — read it from config if possible, or derive from the first embedding response.

### First run
- On first `memex` command: create `~/.memex/` directory, copy `config.toml` to `~/.memex/config.toml`, create `memory.db` with schema. All silently.
- Print nothing on first run except the result of the command itself.

---

## Testing (Manual Verification Steps)

After build, verify these work in order:

```bash
# 1. Basic add and list
memex add "I prefer TypeScript over Python" --scope self
memex add "This project uses FastAPI" --scope project:myapp
memex list

# 2. Semantic search
memex search "what language do I prefer"
memex search "what framework is the project using" --scope project:myapp

# 3. Extract from file
echo "We decided to use Supabase for auth because Clerk was too expensive. The project is deployed on Railway. We use pnpm not npm." > /tmp/test.txt
memex from /tmp/test.txt --scope project:myapp --dry-run
memex from /tmp/test.txt --scope project:myapp

# 4. Dump to clipboard
memex dump --scope project:myapp

# 5. Forget
memex list  # note an ID
memex forget <first-8-chars>
memex list  # confirm deleted

# 6. Info
memex info

# 7. MCP server starts without crashing
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' | memex-server
```

---

## What NOT to Build

Do not add any of the following, even if they seem like good ideas:

- Web UI or dashboard of any kind
- HTTP server or SSE transport for MCP
- Graph visualization
- Cloud sync or S3 support
- Inter-agent event system
- Automatic background sync or file watching
- Database migrations beyond the initial schema
- OpenAI or any non-Ollama embedding backend
- Docker or containerization
- Logging to files (stderr only, and only on errors)
- User authentication or multi-user support

If you find yourself about to add any of the above: stop. Do not add it.

---

*End of build specification.*

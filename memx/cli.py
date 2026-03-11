import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="memx",
    help="Local memory layer for AI tools.",
    no_args_is_help=True,
    add_completion=False,
)

SCOPE_PATTERN = re.compile(r"^(self|project:[A-Za-z0-9_-]+)$")
SOURCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9:_-]{1,64}$")


def _fail(message: str) -> None:
    typer.echo(f"[error] {message}", err=True)
    raise typer.Exit(1)


def _validate_scope(scope: str) -> str:
    if SCOPE_PATTERN.fullmatch(scope):
        return scope
    _fail(
        f"Invalid scope '{scope}'. Must be 'self' or 'project:<name>' where name uses only letters, numbers, underscores, and hyphens."
    )


def _compute_expires_at(scope: str, ttl_days: int) -> Optional[int]:
    if scope == "self":
        return None
    return int(time.time()) + (ttl_days * 86400)


def _validate_source_label(source_label: Optional[str]) -> Optional[str]:
    if source_label is None:
        return None
    value = source_label.strip()
    if not value:
        return None
    if not SOURCE_LABEL_PATTERN.fullmatch(value):
        _fail("Invalid source label. Must match ^[A-Za-z0-9:_-]{1,64}$.")
    return value


def _load_deps():
    from memex.config import load_config
    from memex.db import delete_expired, get_connection, init_db

    return load_config, get_connection, init_db, delete_expired


def _load_config_or_exit():
    load_config, _, _, _ = _load_deps()
    try:
        return load_config()
    except Exception as e:  # pragma: no cover - defensive user-facing guard
        _fail(f"Could not load config: {e}")


def _open_conn_or_exit(cfg):
    _, get_connection, init_db, _ = _load_deps()
    try:
        conn = get_connection(cfg.storage.db_path)
        init_db(conn, cfg.memory.embedding_dim)
        return conn
    except RuntimeError as e:
        _fail(str(e))
    except Exception as e:  # pragma: no cover - defensive user-facing guard
        _fail(f"Could not open database: {e}")


def _cleanup_expired_or_exit(conn):
    _, _, _, delete_expired = _load_deps()
    try:
        return delete_expired(conn)
    except Exception as e:  # pragma: no cover - defensive user-facing guard
        _fail(f"Could not clean expired memories: {e}")


def _parse_tags(tags: Optional[str]) -> list[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


@app.command()
def add(
    fact: str = typer.Argument(..., help="The fact to store"),
    scope: str = typer.Option("self", "--scope", "-s", help="'self' or 'project:<name>'"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    source_label: Optional[str] = typer.Option(None, "--source-label", help="Optional source grouping label"),
):
    """Add a single fact directly to memory."""
    from memex.db import has_active_memory, store_memory
    from memex.embeddings import OllamaEmbeddingError, embed_text

    _validate_scope(scope)
    source_label = _validate_source_label(source_label)
    cfg = _load_config_or_exit()
    tag_list = _parse_tags(tags)
    expires_at = _compute_expires_at(scope, cfg.memory.project_ttl_days)
    conn = _open_conn_or_exit(cfg)
    if has_active_memory(conn, fact.strip(), scope):
        typer.echo("Skipped duplicate fact (already exists in active memory).")
        return

    try:
        embedding = embed_text(
            fact,
            model=cfg.ollama.embed_model,
            base_url=cfg.ollama.base_url,
            timeout=cfg.ollama.timeout_seconds,
        )
    except OllamaEmbeddingError as e:
        _fail(str(e))

    try:
        memory_id = store_memory(
            conn,
            fact=fact,
            embedding=embedding,
            scope=scope,
            source="cli",
            tags=tag_list,
            expires_at=expires_at,
            source_label=source_label,
        )
    except Exception as e:
        _fail(f"Could not store memory: {e}")

    typer.echo(f"✓ Stored [{memory_id[:8]}] -> {fact}")


@app.command(name="from")
def from_input(
    file: Optional[str] = typer.Argument(None, help="File path or '-' for stdin"),
    scope: str = typer.Option("self", "--scope", "-s", help="'self' or 'project:<name>'"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    source_label: Optional[str] = typer.Option(None, "--source-label", help="Optional source grouping label"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show extracted facts without storing"),
):
    """Extract facts from a file or stdin and store them."""
    from memex.db import has_active_memory, store_memory
    from memex.embeddings import OllamaEmbeddingError, embed_text
    from memex.extraction import OllamaExtractionError, dedupe_and_filter_facts, extract_facts, summarize_text

    _validate_scope(scope)
    source_label = _validate_source_label(source_label)
    cfg = _load_config_or_exit()
    tag_list = _parse_tags(tags)

    if file is None or file == "-":
        if sys.stdin.isatty():
            _fail("No input. Pipe text or provide a file path.")
        text = sys.stdin.read()
    else:
        path = Path(file)
        if not path.exists():
            _fail(f"File not found: {file}")
        text = path.read_text(encoding="utf-8")

    if not text.strip():
        _fail("Input is empty.")

    if len(text) > cfg.memory.extract_threshold_chars:
        typer.echo(f"Extracting facts from {len(text)} chars of text...")
        try:
            facts = extract_facts(
                text,
                model=cfg.ollama.extract_model,
                base_url=cfg.ollama.base_url,
                timeout=cfg.ollama.timeout_seconds,
            )
            summary = summarize_text(
                text,
                model=cfg.ollama.extract_model,
                base_url=cfg.ollama.base_url,
                timeout=cfg.ollama.timeout_seconds,
            )
        except OllamaExtractionError as e:
            _fail(str(e))
    else:
        facts = [text.strip()]
        summary = ""

    facts = dedupe_and_filter_facts(facts)
    if not facts:
        typer.echo("No memorable facts found in this text.")
        return

    typer.echo(f"\nPrepared {len(facts)} facts:")
    for i, fact in enumerate(facts, 1):
        typer.echo(f"  {i}. {fact}")

    if dry_run:
        typer.echo("\n[dry-run] Nothing stored.")
        return

    typer.echo("\nStoring...")
    conn = _open_conn_or_exit(cfg)
    expires_at = _compute_expires_at(scope, cfg.memory.project_ttl_days)

    stored = 0
    skipped_duplicates = 0
    source_group_id = str(uuid.uuid4()) if len(text) > cfg.memory.extract_threshold_chars else None
    if summary.strip() and not has_active_memory(conn, summary.strip(), scope):
        try:
            summary_embedding = embed_text(
                summary.strip(),
                model=cfg.ollama.embed_model,
                base_url=cfg.ollama.base_url,
                timeout=cfg.ollama.timeout_seconds,
            )
            store_memory(
                conn,
                fact=summary.strip(),
                embedding=summary_embedding,
                scope=scope,
                source="extract",
                tags=tag_list,
                expires_at=expires_at,
                source_group_id=source_group_id,
                memory_kind="source_summary",
                source_label=source_label,
            )
            stored += 1
        except OllamaEmbeddingError as e:
            typer.echo(f"[skip] Embedding failed for summary: {e}", err=True)
        except Exception as e:
            typer.echo(f"[skip] Could not store summary: {e}", err=True)
    elif summary.strip():
        skipped_duplicates += 1

    for fact in facts:
        normalized_fact = fact.strip()
        if not normalized_fact:
            continue
        if has_active_memory(conn, normalized_fact, scope):
            skipped_duplicates += 1
            continue
        try:
            embedding = embed_text(
                normalized_fact,
                model=cfg.ollama.embed_model,
                base_url=cfg.ollama.base_url,
                timeout=cfg.ollama.timeout_seconds,
            )
            store_memory(
                conn,
                fact=normalized_fact,
                embedding=embedding,
                scope=scope,
                source="extract",
                tags=tag_list,
                expires_at=expires_at,
                source_group_id=source_group_id,
                memory_kind="fact",
                source_label=source_label,
            )
            stored += 1
        except OllamaEmbeddingError as e:
            typer.echo(f"[skip] Embedding failed for fact: {e}", err=True)
        except Exception as e:
            typer.echo(f"[skip] Could not store fact: {e}", err=True)

    typer.echo(f"\n✓ Stored {stored}/{len(facts)} facts to scope '{scope}'")
    if skipped_duplicates:
        typer.echo(f"[info] Skipped {skipped_duplicates} duplicate facts.")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for"),
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Filter by scope"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
):
    """Search memories semantically."""
    from memex.embeddings import OllamaEmbeddingError, embed_text
    from memex.search import search_memories

    if scope:
        _validate_scope(scope)
    if limit <= 0:
        _fail("Limit must be greater than 0.")

    cfg = _load_config_or_exit()

    try:
        query_embedding = embed_text(
            query,
            model=cfg.ollama.embed_model,
            base_url=cfg.ollama.base_url,
            timeout=cfg.ollama.timeout_seconds,
        )
    except OllamaEmbeddingError as e:
        _fail(str(e))

    conn = _open_conn_or_exit(cfg)

    try:
        results = search_memories(conn, query_embedding, scope=scope, limit=limit)
    except Exception as e:
        _fail(f"Search failed: {e}")

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
    source_label: Optional[str] = typer.Option(None, "--source-label", help="Filter by source label"),
    show_expired: bool = typer.Option(False, "--expired", help="Include expired memories"),
):
    """List all stored memories."""
    from memex.db import get_all_memories

    if scope:
        _validate_scope(scope)
    source_label = _validate_source_label(source_label)

    cfg = _load_config_or_exit()
    conn = _open_conn_or_exit(cfg)
    _cleanup_expired_or_exit(conn)

    try:
        memories = get_all_memories(
            conn,
            scope=scope,
            include_expired=show_expired,
            source_label=source_label,
        )
    except Exception as e:
        _fail(f"Could not list memories: {e}")

    if not memories:
        typer.echo("No memories found.")
        return

    grouped: dict[str, list] = {}
    for m in memories:
        grouped.setdefault(m.scope, []).append(m)

    for scope_name, items in grouped.items():
        typer.echo(f"\n-- {scope_name} ({len(items)}) --")
        for m in items:
            expires = (
                datetime.fromtimestamp(m.expires_at, tz=timezone.utc).strftime("%Y-%m-%d")
                if m.expires_at
                else "never"
            )
            prefix = "[summary] " if m.memory_kind == "source_summary" else ""
            typer.echo(f"  [{m.id[:8]}] {prefix}{m.fact}  (expires: {expires})")


@app.command()
def forget(
    memory_id: str = typer.Argument(..., help="Memory ID or first 8 chars of ID"),
):
    """Delete a specific memory by ID."""
    from memex.db import delete_memory, get_all_memories

    cfg = _load_config_or_exit()
    conn = _open_conn_or_exit(cfg)

    if len(memory_id) == 8:
        try:
            all_memories = get_all_memories(conn, include_expired=True)
        except Exception as e:
            _fail(f"Could not load memories for ID lookup: {e}")

        matches = [m for m in all_memories if m.id.startswith(memory_id)]
        if len(matches) == 0:
            _fail(f"No memory found with ID starting with '{memory_id}'")
        if len(matches) > 1:
            _fail(f"Ambiguous ID prefix '{memory_id}' matches {len(matches)} memories. Use full ID.")
        memory_id = matches[0].id

    try:
        deleted = delete_memory(conn, memory_id)
    except Exception as e:
        _fail(f"Could not delete memory: {e}")

    if deleted:
        typer.echo(f"✓ Deleted memory {memory_id[:8]}")
    else:
        _fail(f"Memory not found: {memory_id}")


@app.command()
def clear(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt and delete all memories.",
    ),
):
    """Delete all memories after confirmation."""
    from memex.db import delete_all_memories

    cfg = _load_config_or_exit()
    conn = _open_conn_or_exit(cfg)

    if not yes:
        confirmed = typer.confirm(
            "This will permanently delete all memories in all scopes. Continue?",
            default=False,
        )
        if not confirmed:
            typer.echo("Cancelled. No memories deleted.")
            return

    try:
        deleted_count = delete_all_memories(conn)
    except Exception as e:
        _fail(f"Could not clear memories: {e}")

    typer.echo(f"✓ Cleared {deleted_count} memories.")


@app.command()
def dedupe(
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Limit to one scope"),
    source_label: Optional[str] = typer.Option(None, "--source-label", help="Filter by source label"),
    apply: bool = typer.Option(False, "--apply", help="Actually delete duplicates"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation when using --apply"),
):
    """Find and optionally delete near-duplicate memories."""
    import difflib

    from memex.db import delete_memory, get_all_memories
    from memex.extraction import normalize_fact

    if scope:
        _validate_scope(scope)
    source_label = _validate_source_label(source_label)

    cfg = _load_config_or_exit()
    conn = _open_conn_or_exit(cfg)
    _cleanup_expired_or_exit(conn)

    try:
        memories = get_all_memories(conn, scope=scope, source_label=source_label)
    except Exception as e:
        _fail(f"Could not load memories for dedupe: {e}")

    kept_norm: dict[tuple[str, str], list[str]] = {}
    duplicate_ids: list[str] = []
    duplicate_lines: list[str] = []

    for m in memories:
        key = (m.scope, m.memory_kind)
        candidate = normalize_fact(m.fact)
        seen = kept_norm.setdefault(key, [])
        is_duplicate = candidate in seen or any(
            difflib.SequenceMatcher(a=candidate, b=existing).ratio() >= 0.93 for existing in seen
        )
        if is_duplicate:
            duplicate_ids.append(m.id)
            duplicate_lines.append(f"  [{m.id[:8]}] ({m.scope}) {m.fact}")
        else:
            seen.append(candidate)

    if not duplicate_ids:
        typer.echo("No duplicates found.")
        return

    typer.echo(f"Found {len(duplicate_ids)} duplicate memories:")
    for line in duplicate_lines[:100]:
        typer.echo(line)
    if len(duplicate_lines) > 100:
        typer.echo(f"  ... and {len(duplicate_lines) - 100} more")

    if not apply:
        typer.echo("\nPreview only. Re-run with --apply to delete these duplicates.")
        return

    if not yes:
        confirmed = typer.confirm(
            f"Delete {len(duplicate_ids)} duplicate memories permanently?",
            default=False,
        )
        if not confirmed:
            typer.echo("Cancelled. No memories deleted.")
            return

    deleted = 0
    for memory_id in duplicate_ids:
        try:
            if delete_memory(conn, memory_id):
                deleted += 1
        except Exception as e:
            typer.echo(f"[skip] Could not delete {memory_id[:8]}: {e}", err=True)

    typer.echo(f"✓ Deleted {deleted}/{len(duplicate_ids)} duplicates.")


@app.command()
def dump(
    scope: Optional[str] = typer.Option(None, "--scope", "-s", help="Scope to dump"),
    source_label: Optional[str] = typer.Option(None, "--source-label", help="Filter by source label"),
    no_copy: bool = typer.Option(False, "--no-copy", help="Print only, don't copy to clipboard"),
):
    """Dump memory context block to clipboard (for pasting into ChatGPT etc.)."""
    from memex.db import get_all_memories

    if scope:
        _validate_scope(scope)
    source_label = _validate_source_label(source_label)

    cfg = _load_config_or_exit()
    conn = _open_conn_or_exit(cfg)

    try:
        memories = get_all_memories(conn, scope=scope, source_label=source_label)
    except Exception as e:
        _fail(f"Could not dump memories: {e}")

    if not memories:
        typer.echo("No memories to dump.")
        return

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
    from memex.db import get_all_memories

    cfg = _load_config_or_exit()
    conn = _open_conn_or_exit(cfg)

    expired_count = _cleanup_expired_or_exit(conn)

    try:
        memories = get_all_memories(conn, include_expired=False)
    except Exception as e:
        _fail(f"Could not read memory stats: {e}")

    scopes: dict[str, int] = {}
    for m in memories:
        scopes[m.scope] = scopes.get(m.scope, 0) + 1

    typer.echo("\nmemx info")
    typer.echo(f"  DB path:      {cfg.storage.db_path}")
    typer.echo(f"  Total facts:  {len(memories)}")
    typer.echo(f"  Expired (cleaned): {expired_count}")
    typer.echo(f"  Ollama:       {cfg.ollama.base_url}")
    typer.echo(f"  Embed model:  {cfg.ollama.embed_model}")
    typer.echo(f"  Extract model:{cfg.ollama.extract_model}")
    typer.echo("\n  Scopes:")
    for scope_name, count in sorted(scopes.items()):
        typer.echo(f"    {scope_name}: {count} facts")


if __name__ == "__main__":
    app()

import asyncio
import argparse
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import mcp.server.stdio
from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions

from memx.config import load_config
from memx.db import (
    delete_expired,
    delete_memory,
    get_all_memories,
    get_connection,
    has_active_memory,
    init_db,
    store_memory,
)
from memx.embeddings import OllamaEmbeddingError, embed_text
from memx.extraction import OllamaExtractionError, dedupe_and_filter_facts, extract_facts, summarize_text
from memx.search import search_memories

server = Server("memx")
LOGGER = logging.getLogger("memx.server")
_cfg = None
_conn = None
_db_lock = None
SCOPE_PATTERN = re.compile(r"^(self|project:[A-Za-z0-9_-]+)$")
SOURCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9:_-]{1,64}$")
MAX_TOOL_TEXT_CHARS = 100_000
MAX_FACT_CHARS = 2_000
_last_cleanup_at = 0
STARTUP_BANNER = r"""
 /$$      /$$ /$$$$$$$$ /$$      /$$| $$   /$$
| $$$    /$$$| $$_____/| $$$    /$$$| $$  / $$
| $$$$  /$$$$| $$      | $$$$  /$$$$|  $$/ $$/
| $$ $$/$$ $$| $$$$$   | $$ $$/$$ $$| \ $$$$/ 
| $$  $$$| $$| $$__/   | $$  $$$| $$|  >$$  $$ 
| $$\  $ | $$| $$      | $$\  $ | $$| /$$/\  $$
| $$ \/  | $$| $$$$$$$$| $$ \/  | $$|  $$  \ $$
|__/     |__/|________/|__/     |__/|__/  |__/                                                        
"""


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
        init_db(_conn, cfg.memory.embedding_dim)
    return _conn


def get_db_lock() -> asyncio.Lock:
    global _db_lock
    if _db_lock is None:
        _db_lock = asyncio.Lock()
    return _db_lock


def _compute_expires_at(scope: str) -> Optional[int]:
    cfg = get_cfg()
    if scope == "self":
        return None
    return int(time.time()) + (cfg.memory.project_ttl_days * 86400)


def _scope_valid(scope: str) -> bool:
    return SCOPE_PATTERN.fullmatch(scope) is not None


def _maybe_cleanup_expired(conn) -> None:
    global _last_cleanup_at
    now = int(time.time())
    # Keep MCP calls fast; cleanup every 10 minutes.
    if now - _last_cleanup_at < 600:
        return
    delete_expired(conn)
    _last_cleanup_at = now


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
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags for later filtering/export.",
                    },
                    "source_label": {
                        "type": "string",
                        "description": "Optional label to group related memories (example: research:redis-ha).",
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
            name="mem_context",
            description=(
                "Return structured context for agent reasoning. "
                "It groups relevant summary memories with supporting atomic facts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to assemble context for.",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Optional. Filter context to a specific scope.",
                    },
                    "source_label": {
                        "type": "string",
                        "description": "Optional source label filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Approximate max memories to use. Default 12.",
                        "default": 12,
                    },
                    "include_facts_per_group": {
                        "type": "integer",
                        "description": "How many facts to include under each summary. Default 3.",
                        "default": 3,
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
                    "source_label": {
                        "type": "string",
                        "description": "Optional source label filter.",
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
    try:
        async with get_db_lock():
            cfg = get_cfg()
            conn = get_conn()
            args = arguments or {}

            match name:
                case "mem_store":
                    text = args.get("text")
                    if not isinstance(text, str) or not text.strip():
                        return [types.TextContent(type="text", text="Error: 'text' must be a non-empty string.")]
                    if len(text) > MAX_TOOL_TEXT_CHARS:
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Error: 'text' exceeds {MAX_TOOL_TEXT_CHARS} characters. Provide a smaller chunk.",
                            )
                        ]

                    scope = args.get("scope", cfg.memory.default_scope)
                    if not isinstance(scope, str) or not _scope_valid(scope):
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Error: Invalid scope '{scope}'. Must be 'self' or 'project:<name>'.",
                            )
                        ]
                    raw_tags = args.get("tags", [])
                    tags: list[str] = []
                    if raw_tags is not None:
                        if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
                            return [types.TextContent(type="text", text="Error: 'tags' must be an array of strings.")]
                        tags = [t.strip() for t in raw_tags if t.strip()][:20]
                    source_label = args.get("source_label")
                    if source_label is not None:
                        if not isinstance(source_label, str) or not SOURCE_LABEL_PATTERN.fullmatch(source_label):
                            return [
                                types.TextContent(
                                    type="text",
                                    text="Error: 'source_label' must match ^[A-Za-z0-9:_-]{1,64}$.",
                                )
                            ]

                    expires_at = _compute_expires_at(scope)
                    _maybe_cleanup_expired(conn)

                    if len(text) > cfg.memory.extract_threshold_chars:
                        try:
                            facts = await asyncio.to_thread(
                                extract_facts,
                                text,
                                model=cfg.ollama.extract_model,
                                base_url=cfg.ollama.base_url,
                                timeout=cfg.ollama.timeout_seconds,
                            )
                            summary = await asyncio.to_thread(
                                summarize_text,
                                text,
                                model=cfg.ollama.extract_model,
                                base_url=cfg.ollama.base_url,
                                timeout=cfg.ollama.timeout_seconds,
                            )
                        except OllamaExtractionError as e:
                            return [types.TextContent(type="text", text=f"Extraction error: {e}")]
                    else:
                        facts = [text.strip()]
                        summary = ""

                    facts = dedupe_and_filter_facts(facts)
                    if not facts and not summary:
                        return [types.TextContent(type="text", text="No memorable facts found in this text.")]

                    stored_ids: list[tuple[str, str]] = []
                    failed = 0
                    skipped_duplicates = 0
                    prepared: list[tuple[str, list[float]]] = []
                    source_group_id = str(uuid.uuid4()) if len(text) > cfg.memory.extract_threshold_chars else None

                    for fact in facts:
                        trimmed = fact.strip()
                        if not trimmed:
                            continue
                        if len(trimmed) > MAX_FACT_CHARS:
                            trimmed = trimmed[:MAX_FACT_CHARS]
                        if has_active_memory(conn, trimmed, scope):
                            skipped_duplicates += 1
                            continue
                        try:
                            embedding = await asyncio.to_thread(
                                embed_text,
                                trimmed,
                                model=cfg.ollama.embed_model,
                                base_url=cfg.ollama.base_url,
                                timeout=cfg.ollama.timeout_seconds,
                            )
                            prepared.append((trimmed, embedding))
                        except OllamaEmbeddingError:
                            failed += 1

                    summary_payload: Optional[tuple[str, list[float]]] = None
                    if summary:
                        summary_text = summary.strip()
                        if summary_text and not has_active_memory(conn, summary_text, scope):
                            try:
                                summary_embedding = await asyncio.to_thread(
                                    embed_text,
                                    summary_text,
                                    model=cfg.ollama.embed_model,
                                    base_url=cfg.ollama.base_url,
                                    timeout=cfg.ollama.timeout_seconds,
                                )
                                summary_payload = (summary_text, summary_embedding)
                            except OllamaEmbeddingError:
                                failed += 1
                        elif summary_text:
                            skipped_duplicates += 1

                    if not prepared and not summary_payload:
                        return [types.TextContent(type="text", text="No storable facts produced from input text.")]

                    try:
                        conn.execute("BEGIN")
                        if summary_payload:
                            summary_id = store_memory(
                                conn,
                                fact=summary_payload[0],
                                embedding=summary_payload[1],
                                scope=scope,
                                source="mcp",
                                tags=tags,
                                expires_at=expires_at,
                                source_group_id=source_group_id,
                                memory_kind="source_summary",
                                source_label=source_label,
                                dedupe=True,
                                commit=False,
                            )
                            stored_ids.append((summary_id[:8], f"[summary] {summary_payload[0]}"))
                        for fact, embedding in prepared:
                            memory_id = store_memory(
                                conn,
                                fact=fact,
                                embedding=embedding,
                                scope=scope,
                                source="mcp",
                                tags=tags,
                                expires_at=expires_at,
                                source_group_id=source_group_id,
                                memory_kind="fact",
                                source_label=source_label,
                                dedupe=True,
                                commit=False,
                            )
                            stored_ids.append((memory_id[:8], fact))
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise

                    lines = [f"Stored {len(stored_ids)} fact(s) to scope '{scope}':"]
                    for short_id, fact in stored_ids:
                        lines.append(f"  [{short_id}] {fact}")
                    if skipped_duplicates:
                        lines.append(f"  ({skipped_duplicates} duplicate facts skipped)")
                    if failed:
                        lines.append(f"  ({failed} facts failed to embed and were skipped)")

                    return [types.TextContent(type="text", text="\n".join(lines))]

                case "mem_search":
                    query = args.get("query")
                    if not isinstance(query, str) or not query.strip():
                        return [types.TextContent(type="text", text="Error: 'query' must be a non-empty string.")]

                    scope = args.get("scope")
                    if scope is not None and (not isinstance(scope, str) or not _scope_valid(scope)):
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Error: Invalid scope '{scope}'. Must be 'self' or 'project:<name>'.",
                            )
                        ]

                    try:
                        limit = int(args.get("limit", 5))
                    except (TypeError, ValueError):
                        limit = 5
                    if limit <= 0:
                        limit = 5

                    _maybe_cleanup_expired(conn)
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
                        lines.append(f"[{memory.scope}] [{memory.id[:8]}] {memory.fact}  (relevance: {score})")

                    return [types.TextContent(type="text", text="\n".join(lines))]

                case "mem_context":
                    query = args.get("query")
                    if not isinstance(query, str) or not query.strip():
                        return [types.TextContent(type="text", text="Error: 'query' must be a non-empty string.")]

                    scope = args.get("scope")
                    if scope is not None and (not isinstance(scope, str) or not _scope_valid(scope)):
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Error: Invalid scope '{scope}'. Must be 'self' or 'project:<name>'.",
                            )
                        ]
                    source_label = args.get("source_label")
                    if source_label is not None:
                        if not isinstance(source_label, str) or not SOURCE_LABEL_PATTERN.fullmatch(source_label):
                            return [
                                types.TextContent(
                                    type="text",
                                    text="Error: 'source_label' must match ^[A-Za-z0-9:_-]{1,64}$.",
                                )
                            ]

                    try:
                        limit = int(args.get("limit", 12))
                    except (TypeError, ValueError):
                        limit = 12
                    if limit <= 0:
                        limit = 12

                    try:
                        facts_per_group = int(args.get("include_facts_per_group", 3))
                    except (TypeError, ValueError):
                        facts_per_group = 3
                    if facts_per_group <= 0:
                        facts_per_group = 3

                    _maybe_cleanup_expired(conn)
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

                    # Pull more than requested so we can build grouped context.
                    results = search_memories(conn, query_embedding, scope=scope, limit=limit * 3)
                    if source_label is not None:
                        results = [(m, s) for (m, s) in results if m.source_label == source_label]

                    if not results:
                        return [types.TextContent(type="text", text="No relevant memories found.")]

                    summaries: list[tuple[object, float]] = []
                    grouped_facts: dict[str, list[tuple[object, float]]] = {}
                    ungrouped_facts: list[tuple[object, float]] = []

                    for memory, score in results:
                        if memory.memory_kind == "source_summary":
                            summaries.append((memory, score))
                            continue
                        if memory.source_group_id:
                            grouped_facts.setdefault(memory.source_group_id, []).append((memory, score))
                        else:
                            ungrouped_facts.append((memory, score))

                    summaries.sort(key=lambda x: x[1], reverse=True)
                    ungrouped_facts.sort(key=lambda x: x[1], reverse=True)
                    for facts in grouped_facts.values():
                        facts.sort(key=lambda x: x[1], reverse=True)

                    lines = [f"Context for: {query}", "", "Context Summary:"]
                    if summaries:
                        for summary, score in summaries[: max(1, min(3, limit // 3 or 1))]:
                            lines.append(f"- ({summary.scope}, r={score}) {summary.fact}")
                    else:
                        lines.append("- No summary memories found; using atomic facts only.")

                    lines.append("")
                    lines.append("Key Facts:")
                    used_facts = 0

                    for summary, _ in summaries:
                        if used_facts >= limit:
                            break
                        group_id = summary.source_group_id
                        if not group_id:
                            continue
                        facts = grouped_facts.get(group_id, [])
                        if not facts:
                            continue
                        lines.append(f"- Supporting facts for [{summary.id[:8]}]:")
                        for fact_mem, fact_score in facts[:facts_per_group]:
                            lines.append(f"  - ({fact_score}) {fact_mem.fact}")
                            used_facts += 1
                            if used_facts >= limit:
                                break

                    if used_facts < limit and ungrouped_facts:
                        lines.append("- Additional relevant facts:")
                        for fact_mem, fact_score in ungrouped_facts[: max(0, limit - used_facts)]:
                            lines.append(f"  - ({fact_score}) [{fact_mem.scope}] {fact_mem.fact}")

                    return [types.TextContent(type="text", text="\n".join(lines))]

                case "mem_list":
                    scope = args.get("scope")
                    if scope is not None and (not isinstance(scope, str) or not _scope_valid(scope)):
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Error: Invalid scope '{scope}'. Must be 'self' or 'project:<name>'.",
                            )
                        ]
                    source_label = args.get("source_label")
                    if source_label is not None:
                        if not isinstance(source_label, str) or not SOURCE_LABEL_PATTERN.fullmatch(source_label):
                            return [
                                types.TextContent(
                                    type="text",
                                    text="Error: 'source_label' must match ^[A-Za-z0-9:_-]{1,64}$.",
                                )
                            ]

                    _maybe_cleanup_expired(conn)
                    memories = get_all_memories(conn, scope=scope, source_label=source_label)
                    if not memories:
                        return [types.TextContent(type="text", text="No memories stored.")]

                    grouped: dict[str, list] = {}
                    for m in memories:
                        grouped.setdefault(m.scope, []).append(m)

                    lines: list[str] = []
                    for scope_name, items in grouped.items():
                        lines.append(f"\n[{scope_name}] - {len(items)} facts")
                        for m in items:
                            prefix = "[summary] " if m.memory_kind == "source_summary" else ""
                            lines.append(f"  [{m.id[:8]}] {prefix}{m.fact}")

                    return [types.TextContent(type="text", text="\n".join(lines))]

                case "mem_forget":
                    memory_id = args.get("id")
                    if not isinstance(memory_id, str) or not memory_id.strip():
                        return [types.TextContent(type="text", text="Error: 'id' must be a non-empty string.")]

                    deleted = delete_memory(conn, memory_id)
                    if deleted:
                        return [types.TextContent(type="text", text=f"Deleted memory {memory_id[:8]}.")]
                    return [types.TextContent(type="text", text=f"Memory not found: {memory_id}")]

                case _:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception:  # Server-stability guard
        LOGGER.exception("Unhandled MCP tool error in %s", name)
        return [types.TextContent(type="text", text="Error: Internal memx server error.")]


def _default_log_file() -> Path:
    return Path.home() / ".memx" / "server.log"


def _configure_logging(level: str, log_file: Optional[str]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_path = Path(log_file).expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        except OSError:
            # Fall back to stderr-only logging if file cannot be opened.
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _spawn_background(log_file: str, log_level: str) -> int:
    log_path = Path(log_file).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Failed to prepare log directory for background mode: {e}", file=sys.stderr)
        return 1

    cmd = [sys.argv[0], "--log-level", log_level, "--log-file", str(log_path)]

    try:
        with open(log_path, "a", encoding="utf-8") as log_fh:
            kwargs = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": log_fh,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                process = subprocess.Popen(cmd, **kwargs)
            else:
                process = subprocess.Popen(cmd, start_new_session=True, **kwargs)
    except OSError as e:
        print(f"Failed to start memx-server in background: {e}", file=sys.stderr)
        return 1

    print(f"memx-server started in background (pid={process.pid})")
    print(f"Logs: {log_path}")
    return 0


async def _run_server() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        LOGGER.info("\n%s", STARTUP_BANNER.rstrip())
        LOGGER.info("memx-server started...")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="memx",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="memx-server", add_help=True)
    parser.add_argument("--bg", action="store_true", help="Run memx-server in background.")
    parser.add_argument(
        "--log-file",
        default=str(_default_log_file()),
        help="Path to server log file. Default: ~/.memx/server.log",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    if args.bg:
        return _spawn_background(args.log_file, args.log_level)

    _configure_logging(args.log_level, args.log_file)
    try:
        asyncio.run(_run_server())
        return 0
    except KeyboardInterrupt:
        LOGGER.info("memx-server stopped")
        return 0
    except Exception:
        LOGGER.exception("memx-server crashed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

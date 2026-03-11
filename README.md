## 1. What it is
memx is a (No BS) local-first AI memory layer that stores atomic facts from notes and conversations in SQLite, embeds them with Ollama, and exposes memory through both a CLI (`memx`) and an MCP stdio server (`memx-server`) for tools like Cursor and Claude Desktop (and any agent that supports local MCP).

## 2. Prerequisites
- Python 3.11+
- Ollama installed
- Ollama models pulled: `nomic-embed-text` and `llama3.2:3b`

## 3. Local install (recommended for development)
From repo root:

```bash
cd memx
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Verify commands are installed:

```bash
which memx
which memx-server
memx --help
```

If your shell cannot find them, call absolute paths:

```bash
./venv/bin/memx --help
./venv/bin/memx-server
```

## 4. Ollama setup
Pull default models:

```bash
ollama pull nomic-embed-text && ollama pull llama3.2:3b
```

Start Ollama:

```bash
ollama serve
```

Optional model tuning:
- Edit `~/.memx/config.toml`
- Set `[ollama].embed_model` and `[ollama].extract_model`
- If embed model dimension is not 768, set `[memory].embedding_dim` to match

## 5. First-run validation
Run these from a terminal with your venv activated:

```bash
memx add "I prefer TypeScript over Python" --scope self
memx list
memx search "what language do I prefer"
```

MCP initialize smoke test:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.0.1"}}}' | memx-server
```

Run server manually with logging/background options:

```bash
memx-server --log-level INFO --log-file ~/.memx/server.log
memx-server --bg --log-level INFO --log-file ~/.memx/server.log
```

## 6. MCP app setup
Use the `memx-server` command from your environment. If your app cannot find it, use your virtualenv binary path.

### Claude Desktop
File: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)  
File: `%APPDATA%\\Claude\\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "memx": {
      "command": "memx-server"
    }
  }
}
```

### Claude Code
Run this command to add MemX MCP server to Claude code.
```bash
claude mcp add memx <path>/memx/venv/bin/memx-server
```
Replace `path` with your local path.

### Cursor (global)
File: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "memx": {
      "command": "memx-server"
    }
  }
}
```

### Windsurf
File: `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "memx": {
      "command": "memx-server"
    }
  }
}
```

After editing config files, fully restart the app.

## 7. CLI command reference
| Command | Description |
|---|---|
| `memx add <fact> [--scope] [--tags] [--source-label]` | Store one fact directly. |
| `memx from [file|-] [--scope] [--tags] [--source-label] [--dry-run]` | Store from file/stdin; long input extracts atomic facts + summary. |
| `memx search <query> [--scope] [--limit]` | Semantic search over memories. |
| `memx list [--scope] [--source-label] [--expired]` | List memories by scope/label. |
| `memx forget <id-or-prefix>` | Delete one memory by UUID or unique short prefix. |
| `memx clear [--yes]` | Delete all memories with confirmation. |
| `memx dedupe [--scope] [--source-label] [--apply] [--yes]` | Preview/remove near-duplicate memories. |
| `memx dump [--scope] [--source-label] [--no-copy]` | Print context block and copy to clipboard. |
| `memx info` | Show stats/config and clean expired records first. |

## 8. MCP tool reference
| Tool | Description |
|---|---|
| `mem_store` | Store text to memory; long text is extracted into facts (+summary). Supports `scope`, `tags`, `source_label`. |
| `mem_search` | Flat semantic retrieval with optional `scope`. |
| `mem_context` | Structured retrieval for agents: summaries + supporting facts grouped by source. |
| `mem_list` | List memories with optional `scope` and `source_label`. |
| `mem_forget` | Delete memory by full ID. |

## 9. Scope and memory behavior
- Valid scopes: `self` or `project:<name>` where `<name>` uses `[A-Za-z0-9_-]+`
- `self`: never expires
- `project:*`: expires using `project_ttl_days`
- Long-form saves produce:
  - one `[summary]` memory
  - multiple atomic `fact` memories
  - linked by `source_group_id`
- Active duplicates in same scope are skipped

## 10. Config
Runtime config: `~/.memx/config.toml`

First run creates:
- `~/.memx/config.toml`
- `~/.memx/memory.db`

Keys:
- `[ollama].base_url`
- `[ollama].embed_model`
- `[ollama].extract_model`
- `[ollama].timeout_seconds`
- `[storage].db_path`
- `[memory].extract_threshold_chars`
- `[memory].default_scope`
- `[memory].project_ttl_days`
- `[memory].embedding_dim`

## 11. Typical workflow
### Save from Claude, retrieve in Cursor
1. In Claude, call `mem_store` with `scope` and optional `source_label`.
2. In Cursor, call `mem_context` with same scope/label.
3. Use grouped summary+facts for coding context.

### End of session
```bash
cat notes.txt | memx from - --scope project:myapp --source-label research:myapp
memx list --scope project:myapp --source-label research:myapp
```

### Clean old duplicates
```bash
memx dedupe --scope project:myapp
memx dedupe --scope project:myapp --apply
```

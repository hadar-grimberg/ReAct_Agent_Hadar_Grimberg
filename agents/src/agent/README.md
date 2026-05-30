# Bitext Customer-Support ReAct Agent

A LangGraph-style ReAct agent that answers structured and unstructured questions
about the [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
It routes queries, calls dataset tools, maintains a per-session user profile, and
exposes the same tools externally via an MCP server.

---

## Requirements

```
pip install langgraph langchain-core langchain-mcp-adapters mcp pandas pydantic python-dotenv datasets
```

Set your Nebius API key before running anything:

```bash
export NEBIUS_API_KEY=v1.your-key-here
```

---

## Download the dataset

```bash
python download_dataset.py                       # saves bitext_full.csv
python download_dataset.py --output my_data.csv  # custom output path
```

---

## Running the agent

### Interactive mode

```bash
python main.py
python main.py --dataset path/to/data.csv
python main.py --session alice               # persist profile under session "alice"
```

### Single-query mode

```bash
python main.py -q "What categories exist in the dataset?"
python main.py -q "Show me 3 examples from SHIPPING" --session alice
python main.py --dataset my_data.csv -q "How many refund requests?" --session alice
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `bitext_full.csv` | Path to the CSV dataset |
| `--query` / `-q` | — | Run a single query and exit |
| `--session` | `default_session` | Session ID for profile persistence |
| `--no-colour` | — | Disable ANSI colour output |

### Query types

- **Structured** — data-driven questions answered by querying the dataset:
  _"How many refund requests did we get?"_, _"Show me 3 examples from SHIPPING"_
- **Unstructured** — open-ended questions requiring narrative:
  _"Summarise the FEEDBACK category."_, _"How do agents handle cancellations?"_
- **Out-of-scope** — politely declined:
  _"Who won the 2024 Champions League?"_
- **Profile** — answered from your stored session profile:
  _"What do you remember about me?"_

---

## MCP server

The MCP server exposes three dataset tools to any external MCP client over stdio.
It is separate from the agent — use it when you want to connect a third-party tool
(e.g. Claude Desktop, a custom client, or another agent) directly to the dataset.

### Starting the server

```bash
# Uses bitext_full.csv in the same directory by default
python mcp_server.py

# Point at a different dataset file
BITEXT_DATASET_PATH=/path/to/data.csv python mcp_server.py
```

The server prints startup confirmation to stderr and then waits for MCP messages on
stdin/stdout:

```
📊 [MCP STARTUP] Loading dataset from bitext_full.csv...
✅ [MCP STARTUP] Dataset loaded and ready.
```

### Connecting a client and calling a tool

The server uses the **stdio transport**, so any MCP client launches it as a
subprocess and communicates over stdin/stdout. The example below uses the
official `mcp` Python SDK:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
    # Pass a custom dataset path if needed:
    # env={"BITEXT_DATASET_PATH": "/path/to/data.csv"}
)

async def main():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── List available tools ──────────────────────────────────────
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])
            # → ['list_categories', 'count_rows', 'filter_and_sample']

            # ── Call list_categories (no arguments) ───────────────────────
            result = await session.call_tool("list_categories", {})
            print(result.content[0].text)

            # ── Call count_rows with filters ──────────────────────────────
            result = await session.call_tool(
                "count_rows",
                {"category": "REFUND", "intent": ""},
            )
            print(result.content[0].text)

            # ── Call filter_and_sample ────────────────────────────────────
            result = await session.call_tool(
                "filter_and_sample",
                {"category": "SHIPPING", "intent": "track_order", "n": "2"},
            )
            print(result.content[0].text)

asyncio.run(main())
```

### Available MCP tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `list_categories` | _(none)_ | Return all distinct top-level category names |
| `count_rows` | `category`, `intent` (both optional) | Count rows matching the given filters |
| `filter_and_sample` | `category`, `intent` (optional), `n` (default `"3"`) | Return N sample rows matching the filters |

All tools return a JSON string. On success the payload is the `data` object; on
failure it is a `"Tool Execution Failure: ..."` string.

---

## Project structure

```
.
├── main.py              # CLI entry point
├── agent.py             # LangGraph ReAct agent + graph definition
├── router.py            # Query classifier (structured / unstructured / out-of-scope)
├── tools.py             # Dataset tool implementations + LangGraph tool registry
├── llm.py               # Nebius API chat wrapper returning LangChain messages
├── mcp_server.py        # FastMCP server exposing tools over stdio
├── download_dataset.py  # HuggingFace dataset downloader
├── bitext_full.csv      # Dataset (generated by download_dataset.py)
└── profiles.json        # Per-session user profiles (auto-created)
```
#!/usr/bin/env python3
"""
mcp_server.py — Model Context Protocol (MCP) Server for the Bitext Dataset.
Exposes tools to external clients via FastMCP over stdio.
"""
import os
import sys
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir))

from tools import load_dataset, call_tool

mcp = FastMCP(
    "Bitext Customer Support Analytics Server",
    dependencies=["pandas", "pydantic", "mcp"]
)

DEFAULT_CSV_PATH = current_dir / "bitext_full.csv"


def _startup_load_data():
    """Load the CSV dataset into memory on server start."""
    env_path = os.environ.get("BITEXT_DATASET_PATH")
    dataset_path = Path(env_path) if env_path else DEFAULT_CSV_PATH

    if not dataset_path.exists():
        print(f"⚠️ [MCP ERROR] CSV data not found at: {dataset_path.absolute()}", file=sys.stderr)
        sys.exit(1)

    print(f"📊 [MCP STARTUP] Loading dataset from {dataset_path.name}...", file=sys.stderr)
    load_dataset(str(dataset_path))
    print("✅ [MCP STARTUP] Dataset loaded and ready.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(name="list_categories")
def mcp_list_categories() -> str:
    """
    Retrieve all distinct top-level domain categories from the dataset.
    Use to identify valid filter tags like 'ACCOUNT', 'SHIPPING', or 'REFUND'.
    """
    raw_response = call_tool("list_categories", {})
    return _parse_tool_payload(raw_response)


@mcp.tool(name="count_rows")
def mcp_count_rows(category: str = "", intent: str = "") -> str:
    """
    Count matching records, optionally filtered by category and/or intent.
    Arguments:
        category: Filter rows by top-level section (e.g. 'REFUND', 'SHIPPING').
        intent: Filter rows by granular intent (e.g. 'track_refund', 'cancel_order').
    """
    params = {"category": category.strip().upper(), "intent": intent.strip().lower()}
    raw_response = call_tool("count_rows", params)
    return _parse_tool_payload(raw_response)


@mcp.tool(name="filter_and_sample")
def mcp_filter_and_sample(category: str = "", intent: str = "", n: str = "3") -> str:
    """
    Return sample rows filtered by category and/or intent.
    Args:
        category: Optional upper-case category filter (e.g., 'REFUND').
        intent: Optional lower-case intent filter (e.g., 'track_order').
        n: Number of rows to return (default "3").
    """
    params = {
        "category": category.strip().upper(),
        "intent": intent.strip().lower(),
        "n": str(n)
    }
    raw_response = call_tool("filter_and_sample", params)
    return _parse_tool_payload(raw_response)


def _parse_tool_payload(json_string_result: str) -> str:
    """Decode tool JSON output for LLM consumption."""
    try:
        parsed = json.loads(json_string_result)
        if parsed.get("success", False):
            return json.dumps(parsed.get("data", {}), indent=2, ensure_ascii=False)
        return f"Tool Execution Failure: {parsed.get('error', 'Unknown exception.')}"
    except Exception as err:
        return f"Malformed JSON output: {err}"


if __name__ == "__main__":
    # Load dataset before handing control to FastMCP's event loop
    _startup_load_data()
    mcp.run(transport="stdio")
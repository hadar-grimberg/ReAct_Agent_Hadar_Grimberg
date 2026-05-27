"""
tools.py — Dataset tools for the Bitext customer-support ReAct agent.

Each tool has:
  • A clear name and docstring (the LLM's description)
  • A typed InputSchema (dataclass acting as Pydantic model)
  • A typed return value

Design principle: "A few well-designed tools beat many poorly described ones."
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Shared dataset handle (loaded once, passed to tool functions)
# ---------------------------------------------------------------------------

_df: Optional[pd.DataFrame] = None


def load_dataset(path: str) -> pd.DataFrame:
    global _df
    _df = pd.read_csv(path)
    # Normalise column names
    _df.columns = [c.strip().lower() for c in _df.columns]
    # Ensure expected columns exist
    required = {"instruction", "category", "intent", "response"}
    missing = required - set(_df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Found: {list(_df.columns)}")
    return _df


def get_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
    return _df


# ---------------------------------------------------------------------------
# Input schemas (dataclasses replace Pydantic here; same validation contract)
# ---------------------------------------------------------------------------

@dataclass
class ListCategoriesInput:
    """No parameters needed — lists all unique top-level categories."""
    pass


@dataclass
class ListIntentsInput:
    """
    Parameters
    ----------
    category : str, optional
        If provided, return only intents belonging to this category.
        Pass an empty string or omit to list ALL intents across the dataset.
    """
    category: str = ""


@dataclass
class CountRowsInput:
    """
    Parameters
    ----------
    category : str, optional
        Filter to a specific category before counting.
    intent : str, optional
        Filter to a specific intent before counting.
    Note: both filters are applied together (AND logic).
    """
    category: str = ""
    intent: str = ""


@dataclass
class FilterAndSampleInput:
    """
    Parameters
    ----------
    category : str, optional
        Category to filter on (e.g. "SHIPPING").
    intent : str, optional
        Intent to filter on (e.g. "get_refund").
    n : int
        Number of example rows to return (default 3, max 10).
    """
    n: int = 3
    category: str = ""
    intent: str = ""


@dataclass
class IntentDistributionInput:
    """
    Parameters
    ----------
    category : str, optional
        If given, show the distribution of intents *within* that category.
        If omitted, show the distribution of intents across the whole dataset.
    """
    category: str = ""


@dataclass
class CategoryDistributionInput:
    """No parameters — returns the count of rows per top-level category."""
    pass


@dataclass
class SummarizeCategoryInput:
    """
    Parameters
    ----------
    category : str
        The category to summarise (e.g. "FEEDBACK").
        Returns representative customer queries and agent response patterns.
    max_samples : int
        How many sample rows to pull for summarisation context (default 15).
    """
    category: str
    max_samples: int = 15


@dataclass
class SearchKeywordInput:
    """
    Parameters
    ----------
    keyword : str
        Keyword or phrase to search for inside customer query text.
    column : str
        Which column to search: 'instruction' (customer message) or
        'response' (agent reply). Default is 'instruction'.
    """
    keyword: str
    column: str = "instruction"


# ---------------------------------------------------------------------------
# Tool return types
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def list_categories(_: ListCategoriesInput) -> ToolResult:
    """
    LIST_CATEGORIES — Return all unique top-level category names in the dataset.

    Use this tool first when the user asks what categories exist, or before
    filtering by category so you know valid values.
    """
    try:
        df = get_df()
        cats = sorted(df["category"].dropna().unique().tolist())
        return ToolResult(success=True, data={"categories": cats, "count": len(cats)})
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def list_intents(inp: ListIntentsInput) -> ToolResult:
    """
    LIST_INTENTS — Return unique intent names, optionally filtered by category.

    Use when the user asks what intents exist, or before filtering by intent
    to confirm the exact intent name to pass to other tools.
    """
    try:
        df = get_df()
        if inp.category:
            mask = df["category"].str.upper() == inp.category.upper()
            df = df[mask]
            if df.empty:
                return ToolResult(success=False, data=None,
                                  error=f"Category '{inp.category}' not found.")
        intents = sorted(df["intent"].dropna().unique().tolist())
        return ToolResult(success=True, data={
            "intents": intents,
            "count": len(intents),
            "category_filter": inp.category or "ALL"
        })
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def count_rows(inp: CountRowsInput) -> ToolResult:
    """
    COUNT_ROWS — Count how many rows match the given filters.

    Use to answer questions like:
      • "How many refund requests did we get?" → count_rows(intent='get_refund')
      • "How many rows are in SHIPPING?" → count_rows(category='SHIPPING')
      • "Total dataset size?" → count_rows()

    Filters are combined with AND logic. Empty string = no filter.
    """
    try:
        df = get_df()
        total = len(df)
        if inp.category:
            df = df[df["category"].str.upper() == inp.category.upper()]
        if inp.intent:
            df = df[df["intent"].str.lower() == inp.intent.lower()]
        return ToolResult(success=True, data={
            "count": len(df),
            "total_rows": total,
            "filters_applied": {
                "category": inp.category or None,
                "intent": inp.intent or None
            }
        })
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def filter_and_sample(inp: FilterAndSampleInput) -> ToolResult:
    """
    FILTER_AND_SAMPLE — Return N example rows filtered by category and/or intent.

    Use when the user asks to "show me examples from X" or wants to see
    what actual customer messages look like for a given category or intent.

    Returns the customer instruction, intent label, and agent response for
    each sampled row so the caller can understand the data patterns.
    """
    try:
        df = get_df()
        if inp.category:
            df = df[df["category"].str.upper() == inp.category.upper()]
        if inp.intent:
            df = df[df["intent"].str.lower() == inp.intent.lower()]
        if df.empty:
            return ToolResult(success=False, data=None,
                              error="No rows match the given filters.")
        n = max(1, min(inp.n, 10))
        sample = df.sample(min(n, len(df)), random_state=42)
        rows = sample[["category", "intent", "instruction", "response"]].to_dict(orient="records")
        return ToolResult(success=True, data={"examples": rows, "total_matching": len(df)})
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def intent_distribution(inp: IntentDistributionInput) -> ToolResult:
    """
    INTENT_DISTRIBUTION — Show the count and percentage of each intent.

    Use when the user asks about "distribution of intents" or wants to
    understand which intents are most/least common, optionally within one
    category. Results are sorted by frequency descending.
    """
    try:
        df = get_df()
        if inp.category:
            df = df[df["category"].str.upper() == inp.category.upper()]
            if df.empty:
                return ToolResult(success=False, data=None,
                                  error=f"Category '{inp.category}' not found.")
        counts = df["intent"].value_counts()
        total = counts.sum()
        distribution = [
            {"intent": intent, "count": int(cnt), "pct": round(cnt / total * 100, 1)}
            for intent, cnt in counts.items()
        ]
        return ToolResult(success=True, data={
            "distribution": distribution,
            "total": int(total),
            "category_filter": inp.category or "ALL"
        })
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def category_distribution(_: CategoryDistributionInput) -> ToolResult:
    """
    CATEGORY_DISTRIBUTION — Show the count and percentage of each top-level category.

    Use when the user asks for an overview of the dataset composition,
    category breakdown, or wants to know which categories have the most data.
    """
    try:
        df = get_df()
        counts = df["category"].value_counts()
        total = counts.sum()
        distribution = [
            {"category": cat, "count": int(cnt), "pct": round(cnt / total * 100, 1)}
            for cat, cnt in counts.items()
        ]
        return ToolResult(success=True, data={
            "distribution": distribution,
            "total": int(total)
        })
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def summarize_category(inp: SummarizeCategoryInput) -> ToolResult:
    """
    SUMMARIZE_CATEGORY — Extract representative data for summarising a category.

    Use for open-ended questions like "Summarise the FEEDBACK category" or
    "What kinds of issues appear in SHIPPING?". Returns sample customer messages
    and agent responses so the LLM can synthesise a meaningful summary.

    This tool is meant for UNSTRUCTURED queries that need narrative answers.
    """
    try:
        df = get_df()
        cat_df = df[df["category"].str.upper() == inp.category.upper()]
        if cat_df.empty:
            return ToolResult(success=False, data=None,
                              error=f"Category '{inp.category}' not found.")
        # Stats
        n_rows = len(cat_df)
        intents = cat_df["intent"].value_counts().to_dict()
        # Sample diverse examples (a few per intent)
        samples = []
        for intent in cat_df["intent"].unique():
            sub = cat_df[cat_df["intent"] == intent].head(3)
            for _, row in sub.iterrows():
                samples.append({
                    "intent": row["intent"],
                    "customer_message": row["instruction"],
                    "agent_response": row["response"]
                })
        # Limit total samples
        samples = samples[: inp.max_samples]
        return ToolResult(success=True, data={
            "category": inp.category.upper(),
            "total_rows": n_rows,
            "intents_breakdown": intents,
            "sample_conversations": samples
        })
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


def search_keyword(inp: SearchKeywordInput) -> ToolResult:
    """
    SEARCH_KEYWORD — Find rows where a column contains a keyword or phrase.

    Use when the user asks about a specific topic (e.g. "cancellation", "refund",
    "password") and you need to locate relevant rows before counting or sampling.
    The search is case-insensitive. Column must be 'instruction' or 'response'.
    """
    try:
        df = get_df()
        col = inp.column.lower()
        if col not in ("instruction", "response"):
            return ToolResult(success=False, data=None,
                              error="column must be 'instruction' or 'response'.")
        mask = df[col].str.contains(inp.keyword, case=False, na=False)
        matched = df[mask]
        sample = matched[["category", "intent", "instruction"]].head(5).to_dict(orient="records")
        return ToolResult(success=True, data={
            "keyword": inp.keyword,
            "column_searched": col,
            "matches": len(matched),
            "sample_matches": sample
        })
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e))


# ---------------------------------------------------------------------------
# Tool registry — maps tool name → (callable, input_class, description)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "list_categories": {
        "fn": list_categories,
        "schema": ListCategoriesInput,
        "description": (
            "Return all unique top-level category names in the dataset. "
            "No parameters required. Use first when you need to know what categories exist."
        ),
        "params": {}
    },
    "list_intents": {
        "fn": list_intents,
        "schema": ListIntentsInput,
        "description": (
            "Return unique intent names, optionally filtered to one category. "
            "Param: category (str, optional). Use before filtering by intent to confirm exact names."
        ),
        "params": {"category": "str (optional) — filter intents to this category"}
    },
    "count_rows": {
        "fn": count_rows,
        "schema": CountRowsInput,
        "description": (
            "Count dataset rows, optionally filtered by category and/or intent. "
            "Params: category (str, optional), intent (str, optional). "
            "Use to answer 'how many X' questions."
        ),
        "params": {
            "category": "str (optional) — top-level category filter",
            "intent": "str (optional) — intent filter"
        }
    },
    "filter_and_sample": {
        "fn": filter_and_sample,
        "schema": FilterAndSampleInput,
        "description": (
            "Return N example rows filtered by category and/or intent. "
            "Params: n (int, default 3, max 10), category (str, optional), intent (str, optional). "
            "Use when the user asks to 'show examples' or 'give me samples from X'."
        ),
        "params": {
            "n": "int — number of examples to return (default 3)",
            "category": "str (optional) — category to filter",
            "intent": "str (optional) — intent to filter"
        }
    },
    "intent_distribution": {
        "fn": intent_distribution,
        "schema": IntentDistributionInput,
        "description": (
            "Show count and percentage of each intent, optionally within one category. "
            "Param: category (str, optional). "
            "Use for 'distribution of intents' or 'what intents appear most often' questions."
        ),
        "params": {"category": "str (optional) — restrict distribution to this category"}
    },
    "category_distribution": {
        "fn": category_distribution,
        "schema": CategoryDistributionInput,
        "description": (
            "Show the count and percentage of rows per top-level category across the entire dataset. "
            "No parameters required. Use for dataset overview or 'how is the data split by category' questions."
        ),
        "params": {}
    },
    "summarize_category": {
        "fn": summarize_category,
        "schema": SummarizeCategoryInput,
        "description": (
            "Extract sample conversations and intent breakdown for a given category so the LLM "
            "can write a narrative summary. Params: category (str, required), max_samples (int, default 15). "
            "Use ONLY for unstructured / open-ended questions like 'summarise FEEDBACK' or "
            "'how do agents handle CANCELLATION requests?'."
        ),
        "params": {
            "category": "str (required) — category to summarise",
            "max_samples": "int (optional) — max sample conversations to retrieve"
        }
    },
    "search_keyword": {
        "fn": search_keyword,
        "schema": SearchKeywordInput,
        "description": (
            "Case-insensitive keyword search across customer instructions or agent responses. "
            "Params: keyword (str, required), column ('instruction' or 'response', default 'instruction'). "
            "Use when you need to find rows related to a concept not captured by category/intent filters."
        ),
        "params": {
            "keyword": "str (required) — word or phrase to search for",
            "column": "'instruction' or 'response' (default 'instruction')"
        }
    },
}


def call_tool(name: str, params: dict) -> str:
    """Dispatch a tool call by name with a dict of params. Returns JSON string."""
    if name not in TOOL_REGISTRY:
        return ToolResult(
            success=False, data=None,
            error=f"Unknown tool '{name}'. Available: {list(TOOL_REGISTRY.keys())}"
        ).to_json()
    entry = TOOL_REGISTRY[name]
    try:
        inp = entry["schema"](**params)
        result = entry["fn"](inp)
        return result.to_json()
    except TypeError as e:
        return ToolResult(success=False, data=None,
                          error=f"Invalid params for '{name}': {e}").to_json()

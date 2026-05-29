from __future__ import annotations

import json
from typing import Any, Optional, Dict, List
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import tool


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
# Pydantic Input Schemas (Used by LangGraph for function validation)
# ---------------------------------------------------------------------------

class ListCategoriesInput(BaseModel):
    """Input for listing top-level categories."""
    pass


class ListIntentsInput(BaseModel):
    category: str = Field(
        default="", 
        description="If provided, return only intents belonging to this category. Pass empty or omit to list ALL intents."
    )


class CountRowsInput(BaseModel):
    category: str = Field(default="", description="Filter to a specific top-level category before counting.")
    intent: str = Field(default="", description="Filter to a specific intent before counting.")


class FilterAndSampleInput(BaseModel):
    n: int = Field(default=3, description="Number of example rows to return (default 3, max 10).")
    category: str = Field(default="", description="Category to filter on (e.g. 'SHIPPING').")
    intent: str = Field(default="", description="Intent to filter on (e.g. 'get_refund').")


class IntentDistributionInput(BaseModel):
    category: str = Field(
        default="", 
        description="If given, show intent distribution within that category. If omitted, across the whole dataset."
    )


class CategoryDistributionInput(BaseModel):
    """Input for showing top-level category composition."""
    pass


class SummarizeCategoryInput(BaseModel):
    category: str = Field(..., description="The category to summarise (e.g. 'FEEDBACK'). Case-insensitive.")
    max_samples: int = Field(default=15, description="How many sample rows to pull for summarisation context.")


class SearchKeywordInput(BaseModel):
    keyword: str = Field(..., description="Keyword or phrase to search for inside data text fields.")
    column: str = Field(
        default="instruction", 
        description="Which column to search: 'instruction' (customer message) or 'response' (agent reply)."
    )


class SummarizeByIntentInput(BaseModel):
    intent_keyword: str = Field(
        ...,
        description=(
            "A keyword or concept to match against intent names (e.g. 'complaint', 'refund', 'cancel'). "
            "All intents whose name contains this substring (case-insensitive) will be included."
        )
    )
    max_samples: int = Field(
        default=15,
        description="Maximum number of sample conversations to return across all matched intents."
    )


# ---------------------------------------------------------------------------
# Tool return type (Strongly typed dict structure for LangGraph logs)
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: str = ""

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Tool implementations (Natively decorated for LangGraph execution)
# ---------------------------------------------------------------------------

@tool(args_schema=ListCategoriesInput)
def list_categories() -> Dict[str, Any]:
    """Return all unique top-level category names in the dataset.

    Use this tool first when the user asks what categories exist, or before
    filtering by category so you know valid values.
    """
    try:
        df = get_df()
        cats = sorted(df["category"].dropna().unique().tolist())
        return ToolResult(success=True, data={"categories": cats, "count": len(cats)}).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=ListIntentsInput)
def list_intents(category: str = "") -> Dict[str, Any]:
    """Return unique intent names, optionally filtered by category.

    Use when the user asks what intents exist, or before filtering by intent
    to confirm the exact intent name to pass to other tools.
    """
    try:
        df = get_df()
        if category:
            mask = df["category"].str.upper() == category.upper()
            df = df[mask]
            if df.empty:
                return ToolResult(success=False, data=None, error=f"Category '{category}' not found.").model_dump()
        intents = sorted(df["intent"].dropna().unique().tolist())
        return ToolResult(success=True, data={
            "intents": intents,
            "count": len(intents),
            "category_filter": category or "ALL"
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=CountRowsInput)
def count_rows(category: str = "", intent: str = "") -> Dict[str, Any]:
    """Count how many rows match the given filters.

    Use to answer questions like:
      - 'How many refund requests did we get?' -> count_rows(intent='get_refund')
      - 'How many rows are in SHIPPING?' -> count_rows(category='SHIPPING')
      - 'Total dataset size?' -> count_rows()
    """
    try:
        df = get_df()
        total = len(df)
        if category:
            df = df[df["category"].str.upper() == category.upper()]
        if intent:
            df = df[df["intent"].str.lower() == intent.lower()]
        return ToolResult(success=True, data={
            "count": len(df),
            "total_rows": total,
            "filters_applied": {
                "category": category or None,
                "intent": intent or None
            }
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=FilterAndSampleInput)
def filter_and_sample(n: int = 3, category: str = "", intent: str = "") -> Dict[str, Any]:
    """Return N example rows filtered by category and/or intent.

    Use when the user asks to 'show me examples from X' or wants to see
    what actual customer messages look like for a given category or intent.
    """
    try:
        df = get_df()
        if category:
            df = df[df["category"].str.upper() == category.upper()]
        if intent:
            df = df[df["intent"].str.lower() == intent.lower()]
        if df.empty:
            return ToolResult(success=False, data=None, error="No rows match the given filters.").model_dump()
        
        # Enforce dynamic types safely
        try:
            n = int(n)
        except (ValueError, TypeError):
            n = 3
        n = max(1, min(n, 10))
        
        sample = df.sample(min(n, len(df)), random_state=42)
        rows = sample[["category", "intent", "instruction", "response"]].to_dict(orient="records")
        return ToolResult(success=True, data={"examples": rows, "total_matching": len(df)}).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=IntentDistributionInput)
def intent_distribution(category: str = "") -> Dict[str, Any]:
    """Show the count and percentage of each intent in the dataset.

    Use when the user asks about 'distribution of intents' or wants to
    understand which intents are most/least common within one category.
    """
    try:
        df = get_df()
        if category:
            df = df[df["category"].str.upper() == category.upper()]
            if df.empty:
                return ToolResult(success=False, data=None, error=f"Category '{category}' not found.").model_dump()
        counts = df["intent"].value_counts()
        total = counts.sum()
        distribution = [
            {"intent": intent, "count": int(cnt), "pct": round(cnt / total * 100, 1)}
            for intent, cnt in counts.items()
        ]
        return ToolResult(success=True, data={
            "distribution": distribution,
            "total": int(total),
            "category_filter": category or "ALL"
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=CategoryDistributionInput)
def category_distribution() -> Dict[str, Any]:
    """Show the count and percentage of each top-level category across the entire dataset.

    Use when the user asks for an overview of the dataset composition or category breakdown.
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
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=SummarizeByIntentInput)
def summarize_by_intent(intent_keyword: str, max_samples: int = 15) -> Dict[str, Any]:
    """Find all intents whose name contains a keyword, then return representative
    conversations across those intents for qualitative summarisation.

    Use this when the user asks about a concept expressed as an intent type rather
    than a category — e.g.:
      • "How do agents handle complaint intents?"    → intent_keyword='complaint'
      • "Summarise refund-related intents."          → intent_keyword='refund'
      • "How are cancellation requests dealt with?"  → intent_keyword='cancel'

    Do NOT use summarize_category for these — use this tool instead.
    """
    try:
        df = get_df()
        keyword_lower = intent_keyword.lower()

        matched_intents = [
            i for i in df["intent"].dropna().unique()
            if keyword_lower in i.lower()
        ]

        if not matched_intents:
            # Broaden: search the instruction text as a fallback
            mask = df["intent"].str.lower().str.contains(keyword_lower, na=False)
            if mask.sum() == 0:
                all_intents = sorted(df["intent"].dropna().unique().tolist())
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        f"No intents found containing '{intent_keyword}'. "
                        f"Available intents: {all_intents}"
                    )
                ).model_dump()

        subset = df[df["intent"].isin(matched_intents)]
        total_rows = len(subset)

        intent_counts = subset["intent"].value_counts().to_dict()

        samples: List[Dict] = []
        per_intent = max(1, max_samples // max(len(matched_intents), 1))
        for intent in matched_intents:
            rows = subset[subset["intent"] == intent].head(per_intent)
            for _, row in rows.iterrows():
                samples.append({
                    "intent": row["intent"],
                    "category": row["category"],
                    "customer_message": row["instruction"],
                    "agent_response": row["response"],
                })
        samples = samples[:max_samples]

        return ToolResult(success=True, data={
            "intent_keyword": intent_keyword,
            "matched_intents": matched_intents,
            "intent_counts": intent_counts,
            "total_rows": total_rows,
            "sample_conversations": samples,
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=SummarizeCategoryInput)
def summarize_category(category: str, max_samples: int = 15) -> Dict[str, Any]:
    """Extract representative conversations for summarizing an entire category.

    Use ONLY for unstructured, open-ended questions like 'summarise FEEDBACK' or
    'how do agents handle CANCELLATION requests?'.
    """
    try:
        df = get_df()
        cat_df = df[df["category"].str.upper() == category.upper()]
        if cat_df.empty:
            return ToolResult(success=False, data=None, error=f"Category '{category}' not found.").model_dump()
        
        n_rows = len(cat_df)
        intents = cat_df["intent"].value_counts().to_dict()
        
        samples = []
        for intent in cat_df["intent"].unique():
            sub = cat_df[cat_df["intent"] == intent].head(3)
            for _, row in sub.iterrows():
                samples.append({
                    "intent": row["intent"],
                    "customer_message": row["instruction"],
                    "agent_response": row["response"]
                })
        samples = samples[:max_samples]
        return ToolResult(success=True, data={
            "category": category.upper(),
            "total_rows": n_rows,
            "intents_breakdown": intents,
            "sample_conversations": samples
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


@tool(args_schema=SearchKeywordInput)
def search_keyword(keyword: str, column: str = "instruction") -> Dict[str, Any]:
    """Find dataset rows where a text column contains a specific keyword or phrase.

    Use when searching about an abstract topic ('password', 'cancellation')
    and you need to locate relevant rows before executing counters.
    """
    try:
        df = get_df()
        col = column.lower()
        if col not in ("instruction", "response"):
            return ToolResult(success=False, data=None, error="column must be 'instruction' or 'response'.").model_dump()
        
        mask = df[col].str.contains(keyword, case=False, na=False)
        matched = df[mask]
        sample = matched[["category", "intent", "instruction"]].head(5).to_dict(orient="records")
        return ToolResult(success=True, data={
            "keyword": keyword,
            "column_searched": col,
            "matches": len(matched),
            "sample_matches": sample
        }).model_dump()
    except Exception as e:
        return ToolResult(success=False, data=None, error=str(e)).model_dump()


# ---------------------------------------------------------------------------
# LangGraph Compliant Tool Registry & Utility Dispatcher
# ---------------------------------------------------------------------------

# This simple array can be passed directly to LangGraph's native ToolNode builder!
LANGGRAPH_TOOLS: List[Any] = [
    list_categories,
    list_intents,
    count_rows,
    filter_and_sample,
    intent_distribution,
    category_distribution,
    summarize_category,
    summarize_by_intent,
    search_keyword
]

# Quick-lookup mapping dictionary to support backward compatibility with your custom call_tool functions
TOOL_MAP: Dict[str, Any] = {t.name: t for t in LANGGRAPH_TOOLS}


def call_tool(name: str, params: dict) -> str:
    """Dispatch a tool call by string name using LangGraph's underlying execution layer.

    Returns an OpenAI/LangGraph spec-compliant stringified JSON payload.
    """
    if name not in TOOL_MAP:
        return ToolResult(
            success=False, 
            data=None, 
            error=f"Unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}"
        ).to_json()
    
    selected_tool = TOOL_MAP[name]
    try:
        # Run the tool's validation check and invoke directly
        raw_output = selected_tool.invoke(input=params)
        return json.dumps(raw_output, ensure_ascii=False, indent=2)
    except Exception as e:
        return ToolResult(success=False, data=None, error=f"Execution failed: {str(e)}").to_json()
"""
router.py — Query classification node for the LangGraph-style ReAct agent.

Classifies every incoming user query into one of three buckets:
  • structured   — concrete, data-driven questions answerable by querying the dataset
  • unstructured — open-ended questions requiring summarisation or narrative
  • out_of_scope — questions unrelated to the Bitext customer-support dataset
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from llm import chat


class QueryType(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class RouterDecision:
    query_type: QueryType
    reasoning: str
    confidence: str  # "high" | "medium" | "low"


ROUTER_SYSTEM = """You are a query router for an AI agent that analyses the Bitext
customer-support dataset. The dataset contains customer service conversations with
columns: category (e.g. ACCOUNT, ORDER, REFUND, SHIPPING, PAYMENT, FEEDBACK,
CANCELLATION), intent (e.g. get_refund, cancel_order), customer instruction text,
and agent response text.

Your job is to classify the user's query into EXACTLY ONE of these three types:

STRUCTURED — The query has a concrete, data-driven answer obtainable by querying
  the dataset. Examples:
    • "What categories exist in the dataset?"
    • "How many refund requests did we get?"
    • "Show me 3 examples from the SHIPPING intent."
    • "What is the distribution of intents in the ACCOUNT category?"
    • "How many rows are in the ORDER category?"

UNSTRUCTURED — The query is open-ended and requires summarisation, narrative, or
  qualitative analysis of the dataset. Examples:
    • "Summarise the FEEDBACK category."
    • "How do customer service reps typically respond to cancellation requests?"
    • "What patterns do you notice in shipping complaints?"
    • "Describe how agents handle refund queries."

OUT_OF_SCOPE — The query is NOT about the Bitext dataset. It could be answered from
  general world knowledge, or it is a creative/utility request. Examples:
    • "Who won the 2024 Champions League?"
    • "Write me a poem about customer service."
    • "What is the capital of France?"
    • "Can you help me write an email to my boss?"

Respond in this EXACT format (no extra text):
TYPE: <STRUCTURED|UNSTRUCTURED|OUT_OF_SCOPE>
CONFIDENCE: <high|medium|low>
REASONING: <one sentence explaining the classification>
"""


def classify_query(user_query: str) -> RouterDecision:
    """
    Call the LLM router to classify the query type.
    Falls back to a keyword-based heuristic if the LLM response is malformed.
    """
    messages = [{"role": "user", "content": user_query}]
    response = chat(messages, system=ROUTER_SYSTEM, max_tokens=200, temperature=0.0)
    response = response["content"]

    # Parse structured response
    type_match = re.search(r"TYPE:\s*(STRUCTURED|UNSTRUCTURED|OUT_OF_SCOPE)", response, re.I)
    conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", response, re.I)
    reason_match = re.search(r"REASONING:\s*(.+)", response, re.I)

    if type_match:
        raw_type = type_match.group(1).upper()
        qtype = {
            "STRUCTURED": QueryType.STRUCTURED,
            "UNSTRUCTURED": QueryType.UNSTRUCTURED,
            "OUT_OF_SCOPE": QueryType.OUT_OF_SCOPE,
        }[raw_type]
    else:
        # Fallback heuristic
        qtype = _heuristic_classify(user_query)

    confidence = conf_match.group(1).lower() if conf_match else "medium"
    reasoning = reason_match.group(1).strip() if reason_match else response[:120]

    return RouterDecision(query_type=qtype, reasoning=reasoning, confidence=confidence)


def _heuristic_classify(query: str) -> QueryType:
    """Simple keyword heuristic as a safety net."""
    q = query.lower()
    dataset_keywords = {
        "category", "intent", "dataset", "bitext", "distribution", "refund",
        "shipping", "order", "account", "payment", "feedback", "cancellation",
        "rows", "examples", "sample", "how many", "show me", "list",
    }
    summarise_keywords = {"summarise", "summarize", "describe", "explain", "patterns",
                          "typically", "how do", "what kind"}

    if any(k in q for k in summarise_keywords):
        return QueryType.UNSTRUCTURED
    if any(k in q for k in dataset_keywords):
        return QueryType.STRUCTURED
    return QueryType.OUT_OF_SCOPE

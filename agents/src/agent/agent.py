from __future__ import annotations

import os
import sys
import json
from typing import Literal, TypedDict, Annotated, Optional, Sequence, Dict, List, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from llm import chat
from langgraph.checkpoint.memory import MemorySaver
from router import classify_query, QueryType, RouterDecision
from tools import LANGGRAPH_TOOLS, call_tool
from langchain_core.utils.function_calling import convert_to_openai_tool


# ---------------------------------------------------------------------------
# Tool helpers — use local LANGGRAPH_TOOLS directly (no remote MCP subprocess)
# ---------------------------------------------------------------------------

def _get_native_tool_schemas() -> List[Dict[str, Any]]:
    """Convert local LangGraph tools into OpenAI-compatible tool schemas."""
    return [convert_to_openai_tool(t) for t in LANGGRAPH_TOOLS]


def _build_tools_description() -> str:
    """Build a readable tool description block for system prompts."""
    desc = []
    for t in LANGGRAPH_TOOLS:
        desc.append(f"  • tool: {t.name}\n    description: {t.description}")
    return "\n".join(desc)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 12  # Hard ceiling; graceful fallback beyond this


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """LangGraph State representation tracking agent context across nodes."""
    user_query: str
    router_decision: Optional[RouterDecision]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    trace: Annotated[List[str], add_messages]
    final_answer: str
    iteration: int
    pending_tool_name: Optional[str]
    pending_tool_params: Optional[dict]
    session_id: str


# ---------------------------------------------------------------------------
# Dynamic Profile File Handling Utilities
# ---------------------------------------------------------------------------

def load_user_profile(session_id: str) -> dict:
    """Reads a persistent user profile from disk. If missing, returns template defaults."""
    path = "profiles.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
                if session_id in profiles:
                    return profiles[session_id]
        except Exception:
            pass
    return {
        "user_name": session_id,
        "frequent_topics_or_intents": [],
        "user_preferences_and_constraints": [],
        "observed_facts": []
    }


def save_user_profile(session_id: str, profile_data: dict) -> None:
    """Saves a user profile payload securely back to disk."""
    path = "profiles.json"
    profiles = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
        except Exception:
            pass
    profiles[session_id] = profile_data
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Warning: Failed to write persistent user profile: {e}")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_STRUCTURED = """\
You are an expert data analysis assistant operating in a ReAct (Reason-Action-Observation) framework.
Your purpose is to provide precise, data-driven answers about the customer support dataset using your available tools.

DATASET SCHEMA — CRITICAL DISTINCTIONS:
  • category    — Broad operational domain. Always ALL-CAPS. Examples: ACCOUNT, SHIPPING, REFUND,
                  ORDER, PAYMENT, FEEDBACK, CANCELLATION_REQUEST. This is the TOP-LEVEL grouping.
  • intent      — Granular user action within a category. Always lowercase_snake_case.
                  Examples: track_order, get_refund, cancel_order, change_shipping_address.
  • instruction — The raw customer message text.
  • response    — The agent's reply text.

⚠️  PARAMETER ROUTING RULES — READ BEFORE EVERY TOOL CALL:
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  User says "SHIPPING examples"   → category="SHIPPING",  intent=""         │
  │  User says "track_order examples"→ category="",          intent="track_order"│
  │  User says "ACCOUNT intents"     → list_intents(category="ACCOUNT")        │
  │  "SHIPPING" is ALWAYS a category. It is NEVER an intent.                   │
  │  "get_refund" is ALWAYS an intent. It is NEVER a category.                 │
  └─────────────────────────────────────────────────────────────────────────────┘
  Before calling filter_and_sample or count_rows, ask yourself:
    - Is the value ALL-CAPS (like SHIPPING, ACCOUNT, REFUND)? → put it in `category`
    - Is the value lowercase_snake_case (like track_order)?   → put it in `intent`

WHAT YOU REMEMBER ABOUT THIS USER (PROFILE LAYER)
  User Name: {user_profile_context[user_name]}
  Frequently Asked Topics: {user_profile_context[frequent_topics_or_intents]}
  Preferences and Constraints: {user_profile_context[user_preferences_and_constraints]}
  Observed Facts: {user_profile_context[observed_facts]}


AVAILABLE TOOLS:
{tools_description}

REACT BEHAVIOR RULES:
1. Pre-Tool Reasoning: Before every tool call, explicitly decide: is this value a
   category (ALL-CAPS domain) or an intent (lowercase action)? Write your reasoning.
2. Multi-Step Chaining: For compound queries, chain tools sequentially. E.g., to
   answer "distribution in SHIPPING", call intent_distribution(category="SHIPPING").
3. Schema Self-Correction: If a tool returns "No rows match", you likely put a
   category value into the intent field or vice-versa. Swap it and retry ONCE.
   Do NOT call list_intents just to pick a random intent — re-examine your parameters.
4. Anti-Hallucination: Rely strictly on tool results. Never fabricate counts or examples.
5. Write a structured, human-readable narrative — not a raw data dump.

OUTPUT PROTOCOL:
When you have tool results, you MUST present the actual data to the user.
⛔ NEVER say things like "The function returned X examples" or "The tool found Y rows".
⛔ NEVER describe or summarize what the tool did — show the actual results.
✅ If the user asked for examples, show the actual customer messages and agent responses.
✅ If the user asked for counts, state the number directly.
✅ If the user asked for categories/intents, list them directly.
 
Reply in this exact format:
FINAL ANSWER: <present the actual data clearly and directly here>
"""

AGENT_SYSTEM_UNSTRUCTURED = """\
You are a qualitative analyst agent that interprets and summarises the Bitext
customer-support dataset. Base ALL narrative summaries on actual tool-retrieved
data — never fabricate examples, counts, or statistics.

DATASET SCHEMA:
  • category    — Broad ALL-CAPS domain: ACCOUNT, SHIPPING, REFUND, ORDER,
                  PAYMENT, FEEDBACK, CANCELLATION_REQUEST, etc.
  • intent      — Specific lowercase_snake_case action: track_order, get_refund,
                  complaint, cancel_order, change_shipping_address, etc.
  • instruction — Customer's raw message text.
  • response    — Agent's reply text.

⚠️  CRITICAL TOOL SELECTION — decide this before every tool call:
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ User mentions a CATEGORY (an ALL-CAPS domain like FEEDBACK, SHIPPING):    │
  │   → summarize_category(category="FEEDBACK")                               │
  │                                                                            │
  │ User mentions an INTENT CONCEPT (a type of action like "complaint",       │
  │   "refund", "cancel", "track") — even phrased as "complaint intents",    │
  │   "cancellation requests", "refund-related queries":                      │
  │   → summarize_by_intent(intent_keyword="complaint")                       │
  │   → NEVER call summarize_category for intent-concept queries              │
  │   → NEVER invent a category name like "COMPLAINT" — it does not exist    │
  └────────────────────────────────────────────────────────────────────────────┘

WHAT YOU REMEMBER ABOUT THIS USER (PROFILE LAYER)
  User Name: {user_profile_context[user_name]}
  Frequently Asked Topics: {user_profile_context[frequent_topics_or_intents]}
  Preferences and Constraints: {user_profile_context[user_preferences_and_constraints]}
  Observed Facts: {user_profile_context[observed_facts]}

AVAILABLE TOOLS:
{tools_description}

RULES:
1. Decide FIRST: category name → summarize_category | intent concept → summarize_by_intent
2. Optionally chain intent_distribution for quantitative support.
3. Write a structured, human-readable narrative — not a raw data dump.
4. Every factual claim must be grounded in actual tool results.
5. If the user ask about their profile, use WHAT YOU REMEMBER ABOUT THIS USER in your answer. DO NOT USE TOOL.

When ready to answer, respond with:
FINAL ANSWER: <your narrative here>
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Classifies incoming query intent and handles early out-of-scope routing."""
    query = state["user_query"]
    trace = ["🔀 [ROUTER] Classifying query..."]

    decision = classify_query(query)

    trace.append(AIMessage(
        f"🔀 [ROUTER] Type={decision.query_type.value.upper()}  "
        f"Confidence={decision.confidence}  |  {decision.reasoning}")
    )

    final_answer = ""
    if decision.query_type == QueryType.OUT_OF_SCOPE:
        final_answer = (
            "I'm sorry, but this question falls outside the scope of the Bitext "
            "customer-support dataset that I'm configured to help with.\n\n"
            f"Reason classified: {decision.reasoning}"
        )

    return {
        "router_decision": decision,
        "trace": trace,
        "messages": list(state.get("messages", [])),
        "final_answer": final_answer
    }


def agent_node(state: AgentState) -> dict:
    """Executes the THINK step of the ReAct sequence via LLM chat payload generation."""
    current_iteration = state.get("iteration", 0) + 1
    trace = []
    messages = list(state.get("messages", []))
    decision = state["router_decision"]
    session_id = state.get("session_id", "default_session")

    # ── Max Iteration Breaker ──
    if current_iteration > MAX_ITERATIONS:
        trace.append(SystemMessage(f"⚠️  [AGENT] Max iterations ({MAX_ITERATIONS}) reached. Returning partial data."))
        last_obs = next((m.content for m in reversed(messages) if isinstance(m, ToolMessage)), "No data gathered.")
        return {
            "iteration": current_iteration,
            "trace": trace,
            "final_answer": f"Execution ceiling hit. Partial details gathered:\n\n{last_obs}"
        }

    trace.append(AIMessage(f"🤔 [AGENT] Iteration {current_iteration} — thinking..."))

    # Fetch and stringify user profile attributes dynamically
    profile = load_user_profile(session_id)

    # ── Profile query short-circuit ──────────────────────────────────────────
    # If the user is asking about themselves/what we remember, answer directly
    # from the profile without touching any dataset tools.
    PROFILE_QUERY_PHRASES = [
        "remember about me", "know about me", "do you know me",
        "what do you know", "my name", "who am i", "about me",
        "my preferences", "my interests", "my profile",
    ]
    query_lower = state["user_query"].lower()
    is_profile_query = any(p in query_lower for p in PROFILE_QUERY_PHRASES)

    if is_profile_query:
        name = profile.get("user_name", "unknown")
        topics = profile.get("frequent_topics_or_intents", [])
        prefs = profile.get("user_preferences_and_constraints", [])
        facts = profile.get("observed_facts", [])

        lines = [f"Here is what I remember about you (session: {session_id}):"]
        lines.append(f"  • Name: {name}")
        lines.append(f"  • Frequently explored topics: {', '.join(topics) if topics else 'none recorded yet'}")
        lines.append(f"  • Preferences / constraints: {', '.join(prefs) if prefs else 'none recorded yet'}")
        lines.append(f"  • Other observed facts: {', '.join(facts) if facts else 'none recorded yet'}")

        answer = "\n".join(lines)
        trace.append(AIMessage("👤 [AGENT] Profile query detected — answering from stored profile."))
        return {
            "iteration": current_iteration,
            "trace": trace,
            "messages": [],
            "final_answer": answer,
            "pending_tool_name": None,
            "pending_tool_params": None,
        }
    # ─────────────────────────────────────────────────────────────────────────

    # Assemble context layout
    qtype = decision.query_type if decision else QueryType.STRUCTURED
    system_template = AGENT_SYSTEM_STRUCTURED if qtype == QueryType.STRUCTURED else AGENT_SYSTEM_UNSTRUCTURED
    system_prompt = system_template.format(tools_description=_build_tools_description(), user_profile_context=profile)

    native_tools = _get_native_tool_schemas()

    # Generate inference message history
    response = chat(messages, system=system_prompt, max_tokens=1024, temperature=0.0, tools=native_tools)

    # --- LOOP BREAKER SAFEGUARD ---
    was_last_msg_tool = len(messages) > 0 and isinstance(messages[-1], ToolMessage)

    messages.append(response)

    # Initialize updates
    updates = {
        "iteration": current_iteration,
        "trace": trace,
        "messages": [response],
        "pending_tool_name": None,
        "pending_tool_params": None,
        "final_answer": ""
    }

    # ── Explicit Parsing Match ──
    if "FINAL ANSWER:" in response.content:
        updates["final_answer"] = response.content.split("FINAL ANSWER:", 1)[1].strip()
        trace.append(AIMessage("✅ [AGENT] Final answer produced."))
        return updates

    tool_match = response.tool_calls

    if not tool_match and was_last_msg_tool:
        clean_content = response.content.strip()
        if clean_content:
            # Detect meta-summary language — the model described the tool result
            # instead of presenting the actual data. Re-prompt it to show the data.
            META_SUMMARY_PHRASES = [
                "the function", "the tool", "the provided function",
                "the result", "returns ", "returned ", "retrieved ",
                "the examples include", "the data includes", "the output",
            ]
            is_meta_summary = any(p in clean_content.lower() for p in META_SUMMARY_PHRASES)

            if is_meta_summary:
                trace.append(AIMessage(
                    "⚠️  [AGENT] Meta-summary detected — re-prompting for actual data presentation."
                ))
                messages.append(SystemMessage(
                    content=(
                        "OBSERVATION: You described what the tool returned instead of presenting "
                        "the actual data. Do NOT say 'the function returned' or 'the tool found'. "
                        "Instead, directly show the user the actual examples, numbers, or list from "
                        "the tool result. Reply with FINAL ANSWER: followed by the real data."
                    )
                ))
                updates["messages"] = messages
                return updates
            updates["final_answer"] = clean_content
            trace.append(AIMessage("✅ [AGENT] Final answer produced automatically from tool observation data context."))
            return updates

    if tool_match:
        try:
            updates["pending_tool_name"] = tool_match[0]["name"]
            tool_params = tool_match[0]["args"] or {}
            updates["pending_tool_params"] = tool_params
            trace.append(AIMessage(
                f"🔧 [AGENT] Calling tool: {updates['pending_tool_name']} params={json.dumps(updates['pending_tool_params'])}"))
        except Exception as e:
            trace.append(AIMessage(f"⚠️  [AGENT] Parameter parse exception triggered: {e}"))
            messages.append(SystemMessage(
                content=f"OBSERVATION: Runtime argument exception parsing parameters: {e}. Please correct tool schemas."
            ))
            updates["messages"] = messages
    else:
        trace.append(AIMessage("⚠️  [AGENT] No tool_call or FINAL ANSWER found — prompting retry."))
        messages.append(SystemMessage(
            content="OBSERVATION: Output structural requirements were missed. Ensure tool calls use standard formats."
        ))
        updates["messages"] = messages

    return updates


def _format_tool_result(tool_name: str, data: dict) -> str:
    """
    Convert a successful tool result dict into clean, readable prose so the LLM
    receives structured text rather than raw JSON and is not tempted to re-dump it.
    Falls back to compact JSON for unrecognised shapes.
    """
    try:
        if tool_name == "list_categories":
            cats = data.get("categories", [])
            return (
                    f"The dataset contains {data.get('count', len(cats))} categories:\n"
                    + "\n".join(f"  • {c}" for c in cats)
            )

        if tool_name == "list_intents":
            intents = data.get("intents", [])
            cat_filter = data.get("category_filter", "ALL")
            return (
                    f"Intents under {cat_filter} ({data.get('count', len(intents))} total):\n"
                    + "\n".join(f"  • {i}" for i in intents)
            )

        if tool_name == "count_rows":
            filters = data.get("filters_applied", {})
            parts = [f"{k}={v}" for k, v in filters.items() if v]
            label = f" (filters: {', '.join(parts)})" if parts else ""
            return f"Row count{label}: {data.get('count', '?')} out of {data.get('total_rows', '?')} total rows."

        if tool_name == "filter_and_sample":
            examples = data.get("examples", [])
            total = data.get("total_matching", "?")
            lines = [f"Showing {len(examples)} example(s) — {total} total matching rows:\n"]
            for i, ex in enumerate(examples, 1):
                lines.append(f"--- Example {i} ---")
                lines.append(f"Category : {ex.get('category', '')}")
                lines.append(f"Intent   : {ex.get('intent', '')}")
                lines.append(f"Customer : {ex.get('instruction', '')}")
                lines.append(f"Agent    : {ex.get('response', '')}")
                lines.append("")
            return "\n".join(lines)

        if tool_name == "intent_distribution":
            dist = data.get("distribution", [])
            cat = data.get("category_filter", "ALL")
            lines = [f"Intent distribution for {cat} ({data.get('total', '?')} rows total):"]
            for row in dist:
                lines.append(f"  • {row['intent']}: {row['count']} rows ({row['pct']}%)")
            return "\n".join(lines)

        if tool_name == "category_distribution":
            dist = data.get("distribution", [])
            lines = [f"Category distribution ({data.get('total', '?')} rows total):"]
            for row in dist:
                lines.append(f"  • {row['category']}: {row['count']} rows ({row['pct']}%)")
            return "\n".join(lines)

        if tool_name in ("summarize_category", "summarize_by_intent"):
            samples = data.get("sample_conversations", [])
            header_key = "category" if tool_name == "summarize_category" else "intent_keyword"
            header_val = data.get(header_key, "")
            total = data.get("total_rows", "?")
            lines = [f"Summary data for '{header_val}' — {total} total rows:"]
            if "intents_breakdown" in data:
                lines.append("Intent breakdown: " + ", ".join(
                    f"{k}({v})" for k, v in data["intents_breakdown"].items()
                ))
            if "matched_intents" in data:
                lines.append("Matched intents: " + ", ".join(data["matched_intents"]))
            lines.append("")
            for i, ex in enumerate(samples, 1):
                lines.append(f"--- Sample {i} [{ex.get('intent', '')}] ---")
                lines.append(f"Customer : {ex.get('customer_message', '')}")
                lines.append(f"Agent    : {ex.get('agent_response', '')}")
                lines.append("")
            return "\n".join(lines)

        if tool_name == "search_keyword":
            matches = data.get("sample_matches", [])
            lines = [
                f"Keyword '{data.get('keyword', '')}' found in {data.get('matches', '?')} rows "
                f"(column: {data.get('column_searched', '')}):"
            ]
            for m in matches:
                lines.append(f"  • [{m.get('category', '')} / {m.get('intent', '')}] {m.get('instruction', '')}")
            return "\n".join(lines)

    except Exception:
        pass  # Fall through to compact JSON

    return json.dumps(data, ensure_ascii=False, indent=2)


def tool_node(state: AgentState) -> dict:
    """Executes the tool by name using the local call_tool dispatcher.

    For tools that return directly displayable data (list, count, sample, distribution),
    the formatted result is set as final_answer immediately — no second LLM call needed.
    For summarization tools the result goes into a ToolMessage so the LLM can narrate it.
    """
    tool_name = state.get("pending_tool_name")
    tool_params = state.get("pending_tool_params") or {}
    trace = []
    messages = list(state.get("messages", []))

    if not tool_name:
        return {}

    # Tools whose output IS the answer — no LLM narration pass needed.
    DIRECT_ANSWER_TOOLS = {
        "list_categories",
        "list_intents",
        "count_rows",
        "filter_and_sample",
        "intent_distribution",
        "category_distribution",
        "search_keyword",
    }

    # Bind the tool_call_id from the last AI message
    last_ai_message = messages[-1] if messages else None
    tool_call_id = (
        last_ai_message.tool_calls[0]["id"]
        if last_ai_message and getattr(last_ai_message, "tool_calls", None)
        else "default_id"
    )

    try:
        raw_result = call_tool(tool_name, tool_params)
        parsed = json.loads(raw_result)
        if parsed.get("success") and parsed.get("data"):
            formatted = _format_tool_result(tool_name, parsed["data"])
        elif not parsed.get("success"):
            formatted = f"Tool error: {parsed.get('error', 'unknown error')}"
        else:
            formatted = raw_result
        trace.append(AIMessage(f"📊 [TOOL] {tool_name} → success"))
    except Exception as ex:
        formatted = f"Error: {ex}"
        trace.append(AIMessage(f"❌ [TOOL] {tool_name} → error: {ex}"))

    tool_message = ToolMessage(content=formatted, name=tool_name, tool_call_id=tool_call_id)

    result = {
        "trace": trace,
        "messages": [tool_message],
        "pending_tool_name": None,
        "pending_tool_params": None,
    }

    # For direct-answer tools, set final_answer now — skip the next agent_node call.
    if tool_name in DIRECT_ANSWER_TOOLS and not formatted.startswith("Tool error"):
        trace.append(AIMessage("✅ [TOOL] Direct answer ready — skipping LLM narration pass."))
        result["final_answer"] = formatted

    return result


def profile_updater_node(state: AgentState) -> dict:
    """Analyses the latest conversation turn to dynamically update the profile."""
    session_id = state.get("session_id", "default_session")
    query = state["user_query"]
    answer = state.get("final_answer", "")
    trace = []

    current_profile = load_user_profile(session_id)

    trace.append(AIMessage("👤 [PROFILE SUMMARY] Auditing context for updates..."))

    profile_sys_prompt = """\
        You are an advanced data extraction utility that creates concise, distilled user summaries.
        Review the current user profile state, the new incoming question, and the generated answer. Your job is to output an updated profile mapping object tracking explicit facts revealed about the user (e.g. their name, specific interests, or explicit preferences).
        CRITICAL RULES:
        1. Do not repeat facts already tracked in the profile.
        2. If new topics are explored (like asking about 'REFUND' or 'SHIPPING'), add them cleanly to the 'frequent_topics_or_intents' array.
        3. If the user mentions their name (e.g., "I'm David", "My name is Alice"), extract and save it under 'user_name'.
        Return output strictly as a valid minified JSON object matching the current profile format, without any Markdown wrappers or backticks.
        """

    payload = f"Current Profile:\n{json.dumps(current_profile)}\n\nTurn Info:\nUser Question: {query}\nAgent Answer: {answer}"

    try:
        updater_response = chat(
            messages=[HumanMessage(content=payload)],
            system=profile_sys_prompt,
            max_tokens=512,
            temperature=0.0
        )

        clean_json_str = updater_response.content.strip()
        if clean_json_str.startswith("```"):
            clean_json_str = clean_json_str.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if clean_json_str.startswith("json"):
            clean_json_str = clean_json_str.split("json", 1)[1].strip()
        updated_profile = json.loads(clean_json_str)

        if isinstance(updated_profile, dict) and "user_name" in updated_profile:
            save_user_profile(session_id, updated_profile)
            trace.append(AIMessage("👤 [PROFILE SUMMARY] Profile verified and updated successfully."))
    except Exception as e:
        trace.append(AIMessage(f"👤 [PROFILE SUMMARY] Profile update skipped: {e}"))

    return {
        "trace": trace,
        "final_answer": answer
    }


# ---------------------------------------------------------------------------
# Conditional Routing Edges
# ---------------------------------------------------------------------------

def route_after_classification(state: AgentState) -> Literal["agent_node", "__end__"]:
    """Determines whether to move to reasoning loops or break immediately."""
    decision = state["router_decision"]
    if decision and decision.query_type == QueryType.OUT_OF_SCOPE:
        return END
    return "agent_node"


def route_after_agent(state: AgentState) -> Literal["tool_node", "agent_node", "profile_updater_node"]:
    """Evaluates agent thinking states to break out or activate execution nodes."""
    if state.get("final_answer") or state.get("iteration", 0) >= MAX_ITERATIONS:
        return "profile_updater_node"
    if state.get("pending_tool_name"):
        return "tool_node"
    return "agent_node"


def route_after_tool(state: AgentState) -> Literal["agent_node", "profile_updater_node"]:
    """If tool_node already set a final_answer, skip agent_node and go straight to profile update."""
    if state.get("final_answer"):
        return "profile_updater_node"
    return "agent_node"


# ---------------------------------------------------------------------------
# Graph Compilation Layout
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("router_node", router_node)
builder.add_node("agent_node", agent_node)
builder.add_node("tool_node", tool_node)
builder.add_node("profile_updater_node", profile_updater_node)

builder.add_edge(START, "router_node")

builder.add_conditional_edges(
    "router_node",
    route_after_classification,
)

builder.add_conditional_edges(
    "agent_node",
    route_after_agent,
    {
        "tool_node": "tool_node",
        "agent_node": "agent_node",
        "profile_updater_node": "profile_updater_node"
    }
)

builder.add_conditional_edges(
    "tool_node",
    route_after_tool,
    {
        "agent_node": "agent_node",
        "profile_updater_node": "profile_updater_node",
    }
)

builder.add_edge("profile_updater_node", END)

memory_checkpointer = MemorySaver()
react_agent = builder.compile(checkpointer=memory_checkpointer)


# ---------------------------------------------------------------------------
# Invocation Entry Point
# ---------------------------------------------------------------------------

def run_graph(query: str, session_id: str = "default_session") -> dict:
    """Executes the structured LangGraph engine with thread-bound persistence."""
    config = {"configurable": {"thread_id": session_id}}
    initial_state: AgentState = {
        "user_query": query,
        "router_decision": None,
        "messages": [HumanMessage(content=query)],
        "trace": [],
        "final_answer": "",
        "iteration": 0,
        "pending_tool_name": None,
        "pending_tool_params": None,
        "session_id": session_id,
    }

    return react_agent.invoke(initial_state, config=config)
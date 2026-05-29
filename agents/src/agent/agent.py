from __future__ import annotations

import json
from typing import Literal, TypedDict, Annotated, Optional, Sequence, Dict, List, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from llm import chat
from router import classify_query, QueryType, RouterDecision
from tools import LANGGRAPH_TOOLS, call_tool
from langchain_core.utils.function_calling import convert_to_openai_tool


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

OUTPUT PROTOCOL:
When analysis is complete, reply in this exact format:
FINAL ANSWER: <Your thorough, factually validated response here>
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

AVAILABLE TOOLS:
{tools_description}

RULES:
1. Decide FIRST: category name → summarize_category | intent concept → summarize_by_intent
2. Optionally chain intent_distribution for quantitative support.
3. Write a structured, human-readable narrative — not a raw data dump.
4. Every factual claim must be grounded in actual tool results.

When ready to answer, respond with:
FINAL ANSWER: <your well-structured narrative here>
"""


# ---------------------------------------------------------------------------
# Helper Schema Utilities
# ---------------------------------------------------------------------------

def _get_native_tool_schemas() -> List[Dict[str, Any]]:
    """Converts native LangGraph Pydantic tools into compliant tool schemas.
    """
    return [convert_to_openai_tool(t) for t in LANGGRAPH_TOOLS]


def _build_tools_description() -> str:
    """Assembles a highly readable text schema summary block for system prompts."""
    desc = []
    for t in LANGGRAPH_TOOLS:
        desc.append(f"  • tool: {t.name}\n    description: {t.description}")
    return "\n".join(desc)


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

    # ── Max Iteration Breaker ──
    if current_iteration > MAX_ITERATIONS:
        trace.append(SystemMessage("⚠️  [AGENT] Max iterations ({MAX_ITERATIONS}) reached. Returning partial data."))
        # Safely parse content attributes from real message structures
        last_obs = next((m.content for m in reversed(messages) if isinstance(m, ToolMessage)), "No data gathered.")
        return {
            "iteration": current_iteration,
            "trace": trace,
            "final_answer": f"Execution ceiling hit. Partial details gathered:\n\n{last_obs}"
        }

    trace.append(AIMessage(f"🤔 [AGENT] Iteration {current_iteration} — thinking..."))

    # Assemble context layout
    qtype = decision.query_type if decision else QueryType.STRUCTURED
    system_template = AGENT_SYSTEM_STRUCTURED if qtype == QueryType.STRUCTURED else AGENT_SYSTEM_UNSTRUCTURED
    system_prompt = system_template.format(tools_description=_build_tools_description())

    native_tools = _get_native_tool_schemas()

    # Generate inference message history (CLEANED: Only calling chat() once)
    response = chat(messages, system=system_prompt, max_tokens=1024, temperature=0.0, tools=native_tools)

    # --- LOOP BREAKER SAFEGUARD: Compute look-back status BEFORE modifying message list ---
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
            updates["final_answer"] = clean_content
            trace.append(AIMessage("✅ [AGENT] Final answer produced automatically from tool observation data context."))
            return updates

    if tool_match:
        try:
            updates["pending_tool_name"] = tool_match[0]["name"]
            tool_params = tool_match[0]["args"] or {}

            updates["pending_tool_params"] = tool_params
            trace.append(AIMessage(f"🔧 [AGENT] Calling tool: {updates['pending_tool_name']} params={json.dumps(updates['pending_tool_params'])}"))
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


def tool_node(state: AgentState) -> dict:
    """Executes the specific structured action tool registered inside the execution loop."""
    tool_name = state.get("pending_tool_name")
    tool_params = state.get("pending_tool_params") or {}
    trace = []
    messages = list(state.get("messages", []))

    if not tool_name:
        return {}

    # 1. Inspect recent runtime historical items to bind accurate tool transaction IDs
    last_ai_message = messages[-1] if messages else None
    tool_call_id = "default_id"
    if last_ai_message and hasattr(last_ai_message, "tool_calls") and last_ai_message.tool_calls:
        tool_call_id = last_ai_message.tool_calls[0]["id"]

    # 2. Dispatch dynamic utility execution pipeline
    raw_result = call_tool(tool_name, tool_params)
    result_dict = json.loads(raw_result)

    if result_dict.get("success", False):
        obs_content = json.dumps(result_dict['data'], indent=2)
        trace.append(AIMessage(f"📊 [TOOL]  {tool_name} → success"))
    else:
        obs_content = f"Error: {result_dict.get('error')}"
        trace.append(AIMessage(f"❌ [TOOL]  {tool_name} → error: {result_dict.get('error')}"))

    # 3. Instantiate strict ToolMessage objects tracking back to our context call index
    tool_message = ToolMessage(
        content=obs_content,
        name=tool_name,
        tool_call_id=tool_call_id
    )

    return {
        "trace": trace,
        "messages": [tool_message],
        "pending_tool_name": None,
        "pending_tool_params": None
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


def route_after_agent(state: AgentState) -> Literal["tool_node", "agent_node", "__end__"]:
    """Evaluates agent thinking states to break out or activate execution nodes."""
    if state.get("final_answer"):
        return "__end__"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "__end__"
    if state.get("pending_tool_name"):
        return "tool_node"
    return "agent_node"


# ---------------------------------------------------------------------------
# Graph Compilation Layout
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

# Append Graph Nodes
builder.add_node("router_node", router_node)
builder.add_node("agent_node", agent_node)
builder.add_node("tool_node", tool_node)

# Set Graph Interlinks
builder.add_edge(START, "router_node")

builder.add_conditional_edges(
    "router_node",
    route_after_classification,
)

builder.add_conditional_edges(
    "agent_node",
    route_after_agent,
    {
        "tool_node": "tool_node",   # Or whatever your tool node is named
        "agent_node": "agent_node", # ✓ This explicitly enables the self-loop!
        "__end__": END                  # Maps your end string to the Graph's END
    }
)

builder.add_edge("tool_node", "agent_node")

# Compile Execution State
react_agent = builder.compile()


# ---------------------------------------------------------------------------
# Invocation Entry Point
# ---------------------------------------------------------------------------

def run_graph(query: str) -> dict:
    """Executes the structured LangGraph engine.

    Returns dictionary with populated system context keys.
    """
    initial_state: AgentState = {
        "user_query": query,
        "router_decision": None,
        "messages": [HumanMessage(content=query)],
        "trace": [],
        "final_answer": "",
        "iteration": 0,
        "pending_tool_name": None,
        "pending_tool_params": None
    }

    return react_agent.invoke(initial_state)

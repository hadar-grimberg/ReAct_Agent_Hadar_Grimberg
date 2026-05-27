"""
agent.py — LangGraph ReAct agent for the Bitext dataset.

Graph structure:
  [START] → router_node → {out_of_scope? → END}
                        → {structured/unstructured} → agent_node → tool_node → agent_node → ... → END

The agent follows the ReAct (Reason + Act) loop:
  1. THINK: decide which tool to call (or conclude with a final answer)
  2. ACT:   call the tool
  3. OBSERVE: incorporate the tool result
  4. Repeat until final answer or max_iterations reached
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from llm import chat
from router import classify_query, QueryType, RouterDecision
from tools import TOOL_REGISTRY, call_tool

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
    messages: list[dict]
    trace: list[str]
    final_answer: str
    iteration: int
    pending_tool_name: Optional[str]
    pending_tool_params: Optional[dict]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


AGENT_SYSTEM_STRUCTURED = """\
You are a data assistant running in a ReAct loop, that answers questions about the Bitext customer-support dataset. 
You must choose a tool or provide a final answer using your native tool-calling capabilities.

CRITICAL RULE:
- you MUST call a tool immediately. 

The dataset has these columns:
  • category  — top-level topic (e.g. ACCOUNT, ORDER, REFUND, SHIPPING, PAYMENT, FEEDBACK, CANCELLATION)
  • intent    — specific user intent within a category (e.g. get_refund, cancel_order, track_order)
  • instruction — the customer's original message
  • response  — the customer service agent's reply

You must answer using the provided tools. Do NOT answer from general knowledge.

Available tools:
{tools_description}

Rules for Function/Tool Calling:
- To call an action or look up data, invoke the matching native tool provided in your tool definitions.
- Always start by listing categories or intents if you are unsure of valid filter values.
- Chain tool calls sequentially as needed (e.g. list_intents → count_rows → filter_and_sample).
- If a tool returns an error, examine the payload, try a corrected call, or explain what went wrong.

When you have gathered enough information from your tool observations to give a complete, accurate answer, stop calling tools and respond with:
FINAL ANSWER: <your answer here>
"""
# For Example:
# [{'role': 'user', 'content': 'How many refund requests did we get?'},
#  {'role': 'assistant',
#   'content': 'We need count of rows where category = REFUND? Actually "refund requests" maybe intent get_refund. Let\'s see intents.
#   ' ```tool_call
# {{
#   "tool": "list_intents",
#   "params": {{"category": "REFUND"}}
# }}


AGENT_SYSTEM_UNSTRUCTURED = """\
You are a qualitative analyst agent that interprets and summarises the Bitext
customer-support dataset. You must base your narrative summaries on actual data
retrieved via tools — do NOT fabricate examples or statistics.

The dataset has columns: category, intent, instruction (customer message), response (agent reply).

Available tools:
{tools_description}

To call a tool:
```tool_call
{{
  "tool": "<tool_name>",
  "params": {{<key>: <value>, ...}}
}}
```

When ready to deliver a final narrative answer:
FINAL ANSWER: <your well-structured summary here>

Rules:
- Use summarize_category to gather representative examples for any narrative answer.
- You may also call intent_distribution or filter_and_sample for supporting statistics.
- Write a clear, human-readable summary — not a data dump.
- Ground every claim in actual tool results.



"""


def _get_native_tool_schemas() -> list[dict]:
    """
    Dynamically maps your project's local TOOL_REGISTRY definitions
    into strict, valid OpenAI/Nebius function-calling schemas.
    """
    native_schemas = []

    for name, entry in TOOL_REGISTRY.items():
        # 1. Initialize properties mapping
        properties_dict = {}
        required_list = []

        # 2. Safely parse internal parameters map
        if "params" in entry and entry["params"]:
            for p_name, p_desc in entry["params"].items():
                properties_dict[p_name] = {
                    "type": "string",  # Enforce string types for categorical lookups
                    "description": p_desc
                }
                # Optional: If you want to make all parameters optional by default,
                # keep required_list empty unless specified otherwise.

        # 3. Assemble the compliant layout structure
        schema_wrapper = {
            "type": "function",
            "function": {
                "name": name,
                "description": entry.get("description", f"Execute {name}"),
                "parameters": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": required_list
                }
            }
        }
        native_schemas.append(schema_wrapper)

    return native_schemas


def _build_tools_description() -> str:
    lines = []
    for name, entry in TOOL_REGISTRY.items():
        lines.append(f"• {name}: {entry['description']}")
        if entry["params"]:
            for param, desc in entry["params"].items():
                lines.append(f"    - {param}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Classifies incoming query intent and handles early out-of-scope routing."""
    query = state["user_query"]
    trace = list(state.get("trace", []))

    trace.append("🔀 [ROUTER] Classifying query...")
    decision = classify_query(query)

    trace.append(
        f"🔀 [ROUTER] Type={decision.query_type.value.upper()}  "
        f"Confidence={decision.confidence}  |  {decision.reasoning}"
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
        "final_answer": final_answer,
        "messages": [{"role": "user", "content": query}]
    }


def agent_node(state: AgentState) -> dict:
    """Executes the THINK step of the ReAct sequence via LLM chat payload generation."""
    current_iteration = state.get("iteration", 0) + 1
    trace = list(state.get("trace", []))
    messages = list(state.get("messages", []))
    decision = state["router_decision"]

    # ── Max Iteration Breaker ──
    if current_iteration > 12:
        trace.append(f"⚠️  [AGENT] Max iterations (12) reached. Returning partial data.")
        last_obs = next((m["content"] for m in reversed(messages) if "OBSERVATION:" in m["content"]), "No data gathered.")
        return {
            "iteration": current_iteration,
            "trace": trace,
            "final_answer": f"Execution ceiling hit. Partial details gathered:\n\n{last_obs}"
        }

    trace.append(f"🤔 [AGENT] Iteration {current_iteration} — thinking...")

    # Assemble context layout
    qtype = decision.query_type if decision else QueryType.STRUCTURED
    system_template = AGENT_SYSTEM_STRUCTURED if qtype == QueryType.STRUCTURED else AGENT_SYSTEM_UNSTRUCTURED
    system_prompt = system_template.format(tools_description=_build_tools_description())

    native_tools=_get_native_tool_schemas()
    # Generate inference message history (excluding system prompt from dynamic list mutations)
    response = chat(messages, system=system_prompt, max_tokens=1024, temperature=0.0, tools=native_tools)
    messages.append({"role": "assistant", "content": response['reasoning_content']})

    # Initialize updates
    updates = {
        "iteration": current_iteration,
        "trace": trace,
        "messages": messages,
        "pending_tool_name": None,
        "pending_tool_params": None,
        "final_answer": ""
    }

    # ── Parsing Model Decisions ──
    if "FINAL ANSWER:" in response:
        updates["final_answer"] = response.split("FINAL ANSWER:", 1)[1].strip()
        trace.append("✅ [AGENT] Final answer produced.")
        return updates

    tool_match = response.get("tool_calls")

    if tool_match:
        try:
            updates["pending_tool_name"] = tool_match[0]["function"]["name"]
            updates["pending_tool_params"] = json.loads(tool_match[0]["function"]["arguments"])
            trace.append(f"🔧 [AGENT] Calling tool: {updates['pending_tool_name']} params={json.dumps(updates['pending_tool_params'])}")
        except json.JSONDecodeError as e:
            trace.append(f"⚠️  [AGENT] Malformed tool_call JSON: {e}")
            messages.append({
                "role": "user",
                "content": f"OBSERVATION: Error parsing JSON markdown block: {e}. Re-verify syntactic schemas and retry."
            })
            updates["messages"] = messages
    else:
        trace.append("⚠️  [AGENT] No tool_call or FINAL ANSWER found — prompting retry.")
        messages.append({
            "role": "user",
            "content": "OBSERVATION: Output structural requirements were missed. Ensure tool calls use standard triple backtick formats."
        })
        updates["messages"] = messages

    return updates


def tool_node(state: AgentState) -> dict:
    """Executes the specific structured action tool registered inside the execution loop."""
    tool_name = state.get("pending_tool_name")
    tool_params = state.get("pending_tool_params") or {}
    trace = list(state.get("trace", []))
    messages = list(state.get("messages", []))

    if not tool_name:
        return {}

    raw_result = call_tool(tool_name, tool_params)
    result_dict = json.loads(raw_result)

    if result_dict.get("success", False):
        obs = f"OBSERVATION: Tool '{tool_name}' succeeded.\nResult:\n{json.dumps(result_dict['data'], indent=2)}"
        trace.append(f"📊 [TOOL]  {tool_name} → success")
    else:
        obs = f"OBSERVATION: Tool '{tool_name}' failed. Error: {result_dict.get('error')}"
        trace.append(f"❌ [TOOL]  {tool_name} → error: {result_dict.get('error')}")

    messages.append({"role": "user", "content": obs})

    return {
        "trace": trace,
        "messages": messages,
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


def route_after_agent(state: AgentState) -> Literal["tool_node", "__end__"]:
    """Evaluates agent thinking states to break out or activate execution nodes."""
    if state.get("final_answer"):
        return "__end__"
    if state.get("iteration", 0) >= 12:
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
    """
    Executes the structured LangGraph engine.
    Returns dictionary with populated system context keys.
    """
    initial_state: AgentState = {
        "user_query": query,
        "router_decision": None,
        "messages": [],
        "trace": [],
        "final_answer": "",
        "iteration": 0,
        "pending_tool_name": None,
        "pending_tool_params": None
    }

    return react_agent.invoke(initial_state)

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any

# Import LangChain message types so LangGraph can understand the output
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
API_URL: str = "https://api.tokenfactory.nebius.com/v1/chat/completions"

ROUTING_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
GEN_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"


class LLMError(Exception):
    pass


def chat(
        messages: list[Any],
        system: str = "",
        model: str = ROUTING_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        tools: list[Any] = None,
) -> Any:
    """Send a messages list to the Nebius API and return a LangChain AIMessage response.

    Raises LLMError on failure.
    """
    if not NEBIUS_API_KEY:
        raise LLMError(
            "NEBIUS_API_KEY environment variable is not set. "
            "Export it before running: export NEBIUS_API_KEY=v1. ..."
        )

    # 1. Initialize formatted messages list and handle system prompt upfront
    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})

    # 2. Safely transform incoming LangChain objects into standard JSON dicts
    for msg in messages:
        if hasattr(msg, "type"):
            if msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            elif msg.type == "tool":
                role = "tool"
            else:
                role = msg.type

            # --- FIX: Set blank content to None when tool calls are present ---
            # OpenAI specification demands that assistant messages with tool calls
            # should contain a content value of None if no pre-reasoning text is sent.
            msg_content = msg.content if msg.content != "" else None
            formatted_msg: dict[str, Any] = {"role": role, "content": msg_content}

            # Handle AI tool calls generation
            if msg.type == "ai":
                lc_tool_calls = getattr(msg, "tool_calls", [])
                if not lc_tool_calls and "tool_calls" in getattr(msg, "additional_kwargs", {}):
                    lc_tool_calls = msg.additional_kwargs["tool_calls"]

                if lc_tool_calls:
                    formatted_msg["tool_calls"] = []
                    # Explicitly ensure content is None if the model only generated a tool call
                    formatted_msg["content"] = msg_content if msg_content else None

                    for tc in lc_tool_calls:
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)

                        formatted_msg["tool_calls"].append({
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": json.dumps(tc_args) if isinstance(tc_args, dict) else str(tc_args)
                            }
                        })

            if msg.type == "tool":
                tc_id = getattr(msg, "tool_call_id", None)
                if not tc_id and "tool_call_id" in getattr(msg, "additional_kwargs", {}):
                    tc_id = msg.additional_kwargs["tool_call_id"]

                formatted_msg["tool_call_id"] = tc_id or "default_call_id"
                # ToolMessage content must always be a string structure
                formatted_msg["content"] = str(msg.content or "")

            formatted_messages.append(formatted_msg)

        elif isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            formatted_messages.append({"role": "user", "content": str(msg)})

    # Rebuild the OpenAI/Nebius standard payload
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": formatted_messages,
    }

    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NEBIUS_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)

            message_dict = res_json["choices"][0]["message"]
            content = message_dict.get("content") or ""

            # Extract tool calls if Nebius decided to execute any functions
            tool_calls = []
            if "tool_calls" in message_dict and message_dict["tool_calls"]:
                for tc in message_dict["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        args = tc["function"]["arguments"]

                    tool_calls.append({
                        "name": tc["function"]["name"],
                        "args": args,
                        "id": tc["id"],
                        "type": "tool_call"
                    })

            return AIMessage(content=content, tool_calls=tool_calls)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"API Request Error: {e} - Details: {error_body}")
        # Bubble up real API validation problems instead of masking them with blank answers
        raise LLMError(f"Nebius API rejected request: {error_body}")

    except Exception as e:
        print(f"API Request Error: {e}")
        raise LLMError(f"Inference Connection Failure: {e}")
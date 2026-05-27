"""
llm.py
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any


NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
API_URL: str = "https://api.tokenfactory.nebius.com/v1/chat/completions"
# API_URL="https://api.tokenfactory.us-central1.nebius.com/v1/"
# ROUTING_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

GEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ROUTING_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# ROUTING_MODEL: str = "openai/gpt-oss-120b-fast"
# GEN_MODEL: str = "openai/gpt-oss-120b-fast"


class LLMError(Exception):
    pass


def chat(
    messages: list[dict],
    system: str = "",
    model: str = ROUTING_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tools: list[str] = None,
) -> str:
    """
    Send a messages list to the Anthropic API and return the text response.
    Raises LLMError on failure.
    """
    if not NEBIUS_API_KEY:
        raise LLMError(
            "NEBIUS_API_KEY environment variable is not set. "
            "Export it before running: export NEBIUS_API_KEY=v1. ..."
        )

    # Format messages array to include the system prompt OpenAI-style
    formatted_messages = []
    if system:
        formatted_messages.append({"role": "system", "content": system})
    formatted_messages.extend(messages)

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

            message = res_json["choices"][0]["message"]

            # Standard Text Response fallback
            return message

    except Exception as e:
        print(f"API Request Error: {e}")
        return ""



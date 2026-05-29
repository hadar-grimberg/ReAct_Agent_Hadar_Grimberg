# Bitext Customer-Support ReAct Agent

A **LangGraph-style ReAct agent** that answers questions about the
[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
using multi-step reasoning and a clean tool-calling loop.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│  Router Node │  Classifies query as structured / unstructured / out_of_scope
└──────┬───────┘
       │ out_of_scope ──────────────────────────────► Polite decline (END)
       │
       ▼
┌──────────────┐     tool_call found
│  Agent Node  │ ─────────────────────► ┌─────────────┐
│  (THINK)     │ ◄────────────────────── │  Tool Node  │
└──────┬───────┘      observation        │  (ACT)      │
       │ FINAL ANSWER                    └─────────────┘
       ▼
   Final Answer (END)
```

### Key design decisions

| Component | Design |
|-----------|--------|
| **Router** | LLM-based with keyword heuristic fallback; classifies before any tool is called |
| **ReAct loop** | `agent_node` (think) ↔ `tool_node` (act+observe), max `MAX_ITERATIONS=12` |
| **Tools** | 8 tools with Pydantic-style dataclass schemas and clear docstrings |
| **LLM calls** | Raw `urllib` → Anthropic `/v1/messages` (no SDK required) |
| **Fallback** | After `MAX_ITERATIONS`, returns partial results with a graceful message |

---

## Setup

### 1. Install dependencies

```bash
pip install pandas
# Optional, for downloading the full dataset:
pip install datasets
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. (Optional) Download the full Bitext dataset

```bash
python download_dataset.py --output bitext_full.csv
```

A bundled `bitext_sample.csv` (≈450 rows) is included for immediate use.

---

## Running the Agent

### Interactive mode (default)

```bash
python main.py
```

This drops into a conversation loop. Type your query and press Enter.

### Use the full dataset

```bash
python main.py --dataset bitext_full.csv
```

### Single query (non-interactive)

```bash
python main.py --query "What categories exist in the dataset?"
python main.py --query "How many refund requests did we get?"
python main.py --dataset bitext_full.csv --query "Summarise the FEEDBACK category."
```

### No colour output (e.g. for piping to a file)

```bash
python main.py --no-colour --query "Show me 3 examples from SHIPPING." > output.txt
```

---

## Example Queries

### Structured (data-driven)

```
What categories exist in the dataset?
How many refund requests did we get?
Show me 3 examples from the SHIPPING intent.
What is the distribution of intents in the ACCOUNT category?
How many rows are in the ORDER category?
What are all the intents under PAYMENT?
```

### Unstructured (narrative/summarisation)

```
Summarise the FEEDBACK category.
How do customer service representatives typically respond to cancellation requests?
What patterns appear in shipping-related complaints?
Describe how agents handle refund queries.
```

### Out-of-scope (politely declined)

```
Who won the 2024 Champions League?
Write me a poem about customer service.
What is the capital of France?
```

---

## Tools

| Tool | Description |
|------|-------------|
| `list_categories` | All unique category names |
| `list_intents` | All unique intents, optionally filtered by category |
| `count_rows` | Count rows with optional category + intent filters |
| `filter_and_sample` | N example rows filtered by category/intent |
| `intent_distribution` | Count + % breakdown of intents (optionally per category) |
| `category_distribution` | Count + % breakdown across categories |
| `summarize_category` | Sample conversations for narrative summarisation |
| `search_keyword` | Keyword search in instruction or response text |

---

## Multi-step Reasoning Example

Query: *"How many refund requests did we get?"*

```
🔀 [ROUTER]  Type=STRUCTURED  Confidence=high
🤔 [AGENT]   Iteration 1 — thinking...
🔧 [AGENT]   Calling tool: list_intents  params={"category": "REFUND"}
📊 [TOOL]    list_intents → success
🤔 [AGENT]   Iteration 2 — thinking...
🔧 [AGENT]   Calling tool: count_rows  params={"intent": "get_refund"}
📊 [TOOL]    count_rows → success
✅ [AGENT]   Final answer produced.
```

---

## File Structure

```
agent/
├── main.py              # CLI entry point + interactive loop
├── agent.py             # LangGraph-style graph: router + ReAct loop
├── router.py            # Query classification node
├── tools.py             # 8 dataset tools with typed schemas
├── llm.py               # Anthropic API client (stdlib urllib only)
├── bitext_sample.csv    # Bundled sample dataset (~450 rows)
├── download_dataset.py  # Script to fetch full dataset from HuggingFace
├── requirements.txt     # Dependencies
└── README.md            # This file
```

### Model Choice

Choosing meta-llama/Llama-3.3-70B-Instruct for a production-grade ReAct (Reason-Action-Observation) agent tracking customer-support data is an exceptionally wise choice.
While smaller models (8B to 14B parameters) often struggle with multi-turn coherence, and proprietary closed-source models incur massive pricing overheads, Llama-3.3-70B hits the absolute sweet spot for complex agentic pipelines.

1. World-Class Native Function Calling (Action Accuracy):
  A ReAct agent's biggest vulnerability is structural failure—either hallucinating tool arguments or messing up JSON payloads.Llama 3.3-70B was trained with an explicit focus on zero-shot tool use and function orchestration.
  It excels at parsing native Pydantic descriptions and reliably maps parameter arguments ("n": "3", "category": "SHIPPING", "intent": "") perfectly to your tool schemas without dropping brackets or triggering JSON decode failures.
2. Advanced Metacognition & Self-Correction (The "Reason" Step):
   When a tool returns an error—like filter_and_sample → error: No rows match the given filters. -> smaller open models lack the reasoning depth to diagnose why. 
   They often panic and enter an infinite loop repeating the exact same bad call. 
   Llama-3.3-70B possesses the logical depth necessary to step back and re-evaluate its beliefs.
   It looks at its previous mistake, checks its system memory, deduces that it confused a category string with an intent string, and immediately pivots to an inspection utility (list_intents or list_categories) to correct its direction autonomously.
3. Immense Context Window (128K Tokens):
   ReAct architectures are token-heavy. Every single iteration appends system prompts, available tool schemas, raw tool execution payloads (which can contain rows of text examples), and past agent thoughts into the chat history.
   With a 128K context window, Llama-3.3-70B can maintain a massive data audit trail effortlessly over long-running multi-step chains (10+ iterations) without forgetting the initial user query or truncating vital early tool observations.
4. Unmatched Price-to-Performance RatioRunning state-of-the-art closed-source models (like GPT-4o or Claude 3.5 Sonnet) over a loop that executes up to 12 iterations per user query will balloon your API bill rapidly.
    Hosted on infrastructure like Nebius, Llama-3.3-70B delivers near-frontier reasoning metrics at a tiny fraction of the cost.This allows your enterprise to scale data parsing and qualitative summaries over thousands of data operations sustainably.
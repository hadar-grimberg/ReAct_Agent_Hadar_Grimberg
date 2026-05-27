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

1. Model for Routing (Intent Classification):
meta-llama/Meta-Llama-3.1-8B-Instruct 

I chose this model because it has low latency and high throughput. 
Routing sits at the beginning of the pipeline. Any latency incurred here delays the rest of the agent's execution loop. 
An 8B parameter model features exceptionally fast time-to-first-token (TTFT) and high token generation speeds.

ReAct routing requires the model to classify an input into one of three distinct categories (Structured, Unstructured, 
or Out-of-scope). Llama-3.1-8B heavily supports strict tool calling and JSON formatting schema enforcement.

Moreover, because routing evaluates every single incoming query, utilizing a massive model would inflate API costs 
needlessly. Llama-3.1-8B is maximizing the cost efficiency.

In addition, classifying whether a text is a concrete dataset question vs. an open-ended question vs. an unrelated 
general question, is a highly specialized but straightforward semantic classification task. 
An 8B instruction-tuned model easily masters this with simple few-shot prompting.

2. Model for Generation & ReAct Step Processing: 
meta-llama/Llama-3.3-70B-Instruct 

Once the router delegates a query to the execution graph, the generation model must act as the "brain" of the ReAct framework.
Structured queries requires advanced reasoning, the model needs to intelligently write exact SQL or formulate 
deterministic tool arguments, evaluate the tool payload returned by the LangGraph state, and loop through a 
Thought-Action-Observation cycle. The 70B+ parameter tier has a significantly stronger mathematical, code-generation, 
and multi-step reasoning foundation compared to lightweight models.

For open-ended tasks, the generation model needs to ingest large chunks of retrieved context data and synthesize it into
a clean, cohesive, and professional response without losing crucial details or hallucinating.

Expansive context windows (128K tokens) is crucial for handling large dataset text dumps or comprehensive tool results 
passed back into the graph state.
For queries flagged or routed as out-of-scope, larger models excel at maintaining a system-guided persona—politely but 
firmly deflecting inquiries outside the dataset constraints rather than leaking or ignoring instructions.

Summary Strategy:
Entry Node: Receives the query $\rightarrow$ Invokes Meta-Llama-3.1-8B-Instruct $\rightarrow$ Fast evaluation determines 
the conditional routing path.

The Execution/Action Loop: If Structured or Unstructured $\rightarrow$ Handoff to Llama-3.3-70B-Instruct to generate 
tool calls (ReAct loop), observe results, and compose the ultimate user-facing response.

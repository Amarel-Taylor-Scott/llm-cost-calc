# llm-cost-calc

Compute and compare LLM API costs from token counts against a built-in price table.

## Features

- **Built-in price table** covering OpenAI GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo; Anthropic Claude 3, 3.5; Google Gemini 1.5 & 2.0; Meta Llama 3 & 3.1; Mistral; DeepSeek; Groq; and Perplexity models.
- **Cost calculation** — input and output costs itemised and summed, with per-million token prices surfaced.
- **Model comparison** — side-by-side cost breakdown between any two models.
- **Library** — import `cost`, `compare`, `blended_cost`, and `CostResult` directly.
- **CLI** — `llm-cost-calc cost`, `compare`, and `list` commands.
- **Zero dependencies** — Python standard library only.

## Installation

```bash
# Clone and install in-place
pip install -e .
```

Or run the CLI directly without installing:

```bash
python -m llm_cost_calc.cli cost gpt-4o 100000 50000
python -m llm_cost_calc.cli compare gpt-4o gpt-4o-mini 100000 50000
python -m llm_cost_calc.cli list
```

## Library usage

```python
from llm_cost_calc import cost, compare, blended_cost

# Single model cost
result = cost("gpt-4o", input_tokens=100_000, output_tokens=50_000)
print(result.total_cost)         # 0.70  (USD)
print(result.input_cost)         # 0.25
print(result.output_cost)         # 0.50
print(result.input_price)        # 2.50  ($ per 1M tokens)
print(result.output_price)        # 10.00

# Blended cost (total only, as a float)
total = blended_cost("gpt-4o", input_tokens=100_000, output_tokens=50_000)

# Compare two models
r_a, r_b, ratio = compare("gpt-4o", "gpt-4o-mini",
                           input_tokens=100_000, output_tokens=50_000)
print(f"gpt-4o is {ratio:.2f}x the cost of gpt-4o-mini")
```

### Unknown-model handling

By default an unknown model raises `UnknownModelError`:

```python
from llm_cost_calc import cost, UnknownModelError

try:
    cost("my-model", input_tokens=1_000, output_tokens=500)
except UnknownModelError as exc:
    print(exc.model)   # "my-model"
```

Pass `known_only=False` to get a zero-cost result instead:

```python
result = cost("unknown-model", input_tokens=1_000, output_tokens=500, known_only=False)
assert result.total_cost == 0.0
```

## CLI usage

```bash
# Calculate cost for one model
llm-cost-calc cost gpt-4o 100000 50000

# Compare two models
llm-cost-calc compare gpt-4o gpt-4o-mini 100000 50000

# List all known models
llm-cost-calc list
llm-cost-calc list --columns 4
```

## Price table

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet, claude-3.5-haiku |
| Google | gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash |
| Meta | llama-3-8b-instruct, llama-3-70b-instruct, llama-3.1-8b-instruct, llama-3.1-70b-instruct, llama-3.1-405b-instruct |
| Mistral | mistral-small, mistral-medium, mistral-large |
| DeepSeek | deepseek-chat, deepseek-coder |
| Groq | llama-3.1-70b-speculative, mixtral-8x7b |
| Perplexity | llama-3.1-sonar-small-128k-online, llama-3.1-sonar-large-128k-online |

Prices are in USD per 1,000,000 tokens. Run `llm-cost-calc list` to see all models.

## Testing

```bash
python -m pytest llm_cost_calc/
# or
python -m unittest discover -s llm_cost_calc/
```

All tests must pass and every `.py` file must pass `python3 -m py_compile`.

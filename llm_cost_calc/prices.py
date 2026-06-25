"""Built-in price table: model name → (input_$/1M tokens, output_$/1M tokens)."""

PRICES = {
    # GPT-4o
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-05-13": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    # GPT-4 Turbo
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    # GPT-4
    "gpt-4": (30.00, 60.00),
    "gpt-4-0613": (30.00, 60.00),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    # Claude 3
    "claude-3-opus": (15.00, 75.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Claude 3.5
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3.5-sonnet-20240620": (3.00, 15.00),
    "claude-3.5-sonnet-20241022": (3.00, 15.00),
    # Claude 3.5 Haiku
    "claude-3.5-haiku": (0.80, 4.00),
    "claude-3.5-haiku-20241022": (0.80, 4.00),
    # Gemini 1.5
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-002": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-pro-002": (1.25, 5.00),
    # Gemini 2.0
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.10, 0.40),
    # Llama 3
    "llama-3-8b-instruct": (0.20, 0.20),
    "llama-3-70b-instruct": (0.65, 2.75),
    # Llama 3.1
    "llama-3.1-8b-instruct": (0.20, 0.20),
    "llama-3.1-70b-instruct": (0.65, 2.75),
    "llama-3.1-405b-instruct": (3.50, 14.00),
    # Mistral
    "mistral-small": (0.30, 0.90),
    "mistral-medium": (2.00, 6.00),
    "mistral-large": (4.00, 12.00),
    "mistral-large-2407": (4.00, 12.00),
    # DeepSeek
    "deepseek-chat": (0.14, 0.28),
    "deepseek-coder": (0.14, 0.28),
    # Groq
    "llama-3.1-70b-speculative": (0.89, 0.89),
    "mixtral-8x7b": (0.24, 0.24),
    # Perplexity
    "llama-3.1-sonar-small-128k-online": (0.20, 0.20),
    "llama-3.1-sonar-large-128k-online": (0.60, 0.60),
}


def get_price(model: str) -> tuple[float, float]:
    """Return (input_$/1M, output_$/1M) for a known model."""
    if model not in PRICES:
        raise KeyError(f"unknown model: {model!r}")
    return PRICES[model]


def known_models() -> list[str]:
    """Return a sorted list of all known model names."""
    return sorted(PRICES.keys())

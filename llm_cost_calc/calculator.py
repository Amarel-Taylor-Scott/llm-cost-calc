"""Core cost-calculation logic."""

from dataclasses import dataclass
from typing import Optional

from llm_cost_calc import prices as _prices


class UnknownModelError(ValueError):
    """Raised when a model name is not found in the price table."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"unknown model: {model!r}")


def _validate_tokens(input_tokens: int, output_tokens: int) -> None:
    """Reject non-integer or negative token counts.

    Token counts are conceptually unsigned quantities; allowing negatives
    silently produced *negative* costs, which is never meaningful and can
    mask caller bugs (e.g. an off-by-one in a tokeniser).
    """
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
        raise TypeError("input_tokens must be an int")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
        raise TypeError("output_tokens must be an int")
    if input_tokens < 0:
        raise ValueError(f"input_tokens must be non-negative, got {input_tokens}")
    if output_tokens < 0:
        raise ValueError(f"output_tokens must be non-negative, got {output_tokens}")


@dataclass
class CostResult:
    """Breakdown of a cost calculation for one model.

    Attributes
    ----------
    model : str
        Model name used for the calculation.
    input_tokens : int
        Number of input tokens.
    output_tokens : int
        Number of output tokens.
    input_cost : float
        Cost in USD for input tokens.
    output_cost : float
        Cost in USD for output tokens.
    total_cost : float
        Combined cost in USD.
    input_price : float
        Price per million input tokens.
    output_price : float
        Price per million output tokens.
    """

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    input_price: float
    output_price: float

    def __str__(self) -> str:
        return (
            f"{self.model}: {self.input_tokens:,} in / {self.output_tokens:,} out tokens\n"
            f"  input  @ ${self.input_price:.2f}/1M  → ${self.input_cost:.6f}\n"
            f"  output @ ${self.output_price:.2f}/1M  → ${self.output_cost:.6f}\n"
            f"  TOTAL  = ${self.total_cost:.6f}"
        )


def cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    known_only: bool = True,
) -> CostResult:
    """Calculate the cost for a single model given token counts.

    Parameters
    ----------
    model : str
        Exact model identifier (e.g. ``"gpt-4o"``).
    input_tokens : int
        Number of input/prompt tokens.
    output_tokens : int
        Number of output/completion tokens.
    known_only : bool
        If ``True`` (default) an unknown model raises ``UnknownModelError``.
        If ``False`` a zero-cost result is returned for unknown models.

    Returns
    -------
    CostResult
        Itemised cost breakdown.

    Raises
    ------
    UnknownModelError
        When ``known_only=True`` and the model is not in the price table.
    ValueError
        When ``input_tokens`` or ``output_tokens`` is negative.
    TypeError
        When ``input_tokens`` or ``output_tokens`` is not an ``int``.

    Examples
    --------
    >>> result = cost("gpt-4o", input_tokens=100_000, output_tokens=50_000)
    >>> result.total_cost   # doctest: +ELLIPSIS
    0.7...
    """
    _validate_tokens(input_tokens, output_tokens)
    if model not in _prices.PRICES:
        if known_only:
            raise UnknownModelError(model)
        return CostResult(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            input_price=0.0,
            output_price=0.0,
        )

    input_price, output_price = _prices.PRICES[model]
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return CostResult(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        input_price=input_price,
        output_price=output_price,
    )


def compare(
    model_a: str,
    model_b: str,
    input_tokens: int,
    output_tokens: int,
    *,
    known_only: bool = True,
) -> tuple[CostResult, CostResult, float]:
    """Compare the cost of two models for the same token counts.

    Parameters
    ----------
    model_a, model_b : str
        Exact model identifiers to compare.
    input_tokens, output_tokens : int
        Token counts shared by both calculations.
    known_only : bool
        Passed through to :func:`cost`.

    Returns
    -------
    tuple[CostResult, CostResult, float]
        Result for model_a, result for model_b, and the savings ratio
        ``model_a.total_cost / model_b.total_cost`` (0 if both are zero).

    Raises
    ------
    UnknownModelError
        From :func:`cost` when ``known_only=True`` and a model is unknown.

    Examples
    --------
    >>> r_a, r_b, ratio = compare("gpt-4o", "gpt-4o-mini",
    ...                           input_tokens=10_000, output_tokens=5_000)
    >>> r_a.total_cost > r_b.total_cost
    True
    >>> ratio > 1.0
    True
    """
    result_a = cost(model_a, input_tokens, output_tokens, known_only=known_only)
    result_b = cost(model_b, input_tokens, output_tokens, known_only=known_only)

    if result_b.total_cost == 0.0:
        ratio = 0.0
    else:
        ratio = result_a.total_cost / result_b.total_cost

    return result_a, result_b, ratio


def blended_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    known_only: bool = True,
) -> float:
    """Convenience wrapper — returns only the total cost as a float.

    See :func:`cost` for parameter documentation.
    """
    return cost(model, input_tokens, output_tokens, known_only=known_only).total_cost

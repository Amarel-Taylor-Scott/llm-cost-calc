"""Tests for llm_cost_calc.calculator."""

import unittest

from llm_cost_calc.calculator import (
    CostResult,
    cost,
    compare,
    blended_cost,
    UnknownModelError,
)


class TestCostCalculation(unittest.TestCase):

    def test_gpt4o_exact_cost(self):
        # gpt-4o: $2.50/M input, $10.00/M output
        # 1M input + 0 output → $2.50
        result = cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        self.assertEqual(result.total_cost, 2.50)
        self.assertEqual(result.input_cost, 2.50)
        self.assertEqual(result.output_cost, 0.0)

    def test_gpt4o_output_only(self):
        # 0 input + 1M output → $10.00
        result = cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
        self.assertEqual(result.total_cost, 10.00)
        self.assertEqual(result.input_cost, 0.0)
        self.assertEqual(result.output_cost, 10.00)

    def test_gpt4o_both_tokens(self):
        # 1M in + 1M out → $12.50
        result = cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertEqual(result.total_cost, 12.50)

    def test_gpt4o_mini_small_tokens(self):
        # gpt-4o-mini: $0.15/M in, $0.60/M out
        # 100k in + 50k out = $0.015 + $0.03 = $0.045
        result = cost("gpt-4o-mini", input_tokens=100_000, output_tokens=50_000)
        self.assertAlmostEqual(result.total_cost, 0.045, places=6)

    def test_deepseek_chat(self):
        # deepseek-chat: $0.14/M in, $0.28/M out
        # 500k in + 200k out = $0.07 + $0.056 = $0.126
        result = cost("deepseek-chat", input_tokens=500_000, output_tokens=200_000)
        self.assertAlmostEqual(result.total_cost, 0.126, places=6)

    def test_gemini15_flash_low_cost(self):
        # gemini-1.5-flash: $0.075/M in, $0.30/M out
        # 1M in + 1M out = $0.075 + $0.30 = $0.375
        result = cost("gemini-1.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertEqual(result.total_cost, 0.375)

    def test_result_attributes(self):
        result = cost("claude-3.5-sonnet", input_tokens=200_000, output_tokens=100_000)
        self.assertEqual(result.model, "claude-3.5-sonnet")
        self.assertEqual(result.input_tokens, 200_000)
        self.assertEqual(result.output_tokens, 100_000)
        self.assertEqual(result.input_price, 3.00)
        self.assertEqual(result.output_price, 15.00)

    def test_zero_tokens(self):
        result = cost("gpt-4o", input_tokens=0, output_tokens=0)
        self.assertEqual(result.total_cost, 0.0)


class TestUnknownModel(unittest.TestCase):

    def test_unknown_model_raises(self):
        with self.assertRaises(UnknownModelError) as ctx:
            cost("unknown-model-xyz", input_tokens=1000, output_tokens=500)
        self.assertEqual(ctx.exception.model, "unknown-model-xyz")

    def test_unknown_model_error_message(self):
        with self.assertRaises(UnknownModelError) as ctx:
            cost("bad-model", input_tokens=1, output_tokens=1)
        self.assertIn("bad-model", str(ctx.exception))

    def test_unknown_model_with_known_only_false(self):
        result = cost("unknown", input_tokens=1_000_000, output_tokens=1_000_000, known_only=False)
        self.assertEqual(result.total_cost, 0.0)
        self.assertEqual(result.model, "unknown")

    def test_compare_unknown_model_raises(self):
        with self.assertRaises(UnknownModelError):
            compare("gpt-4o", "not-a-model", input_tokens=1000, output_tokens=500)


class TestCompare(unittest.TestCase):

    def test_compare_returns_tuple_of_three(self):
        result = compare("gpt-4o", "gpt-4o-mini", input_tokens=100_000, output_tokens=50_000)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_compare_result_a_is_gpt4o(self):
        r_a, r_b, ratio = compare("gpt-4o", "gpt-4o-mini",
                                  input_tokens=100_000, output_tokens=50_000)
        self.assertEqual(r_a.model, "gpt-4o")
        self.assertEqual(r_b.model, "gpt-4o-mini")

    def test_compare_ratio_greater_than_one(self):
        # gpt-4o is more expensive than gpt-4o-mini
        r_a, r_b, ratio = compare("gpt-4o", "gpt-4o-mini",
                                  input_tokens=100_000, output_tokens=50_000)
        self.assertGreater(ratio, 1.0)

    def test_compare_same_model_ratio_is_one(self):
        _, _, ratio = compare("gpt-4o", "gpt-4o",
                              input_tokens=100_000, output_tokens=50_000)
        self.assertEqual(ratio, 1.0)

    def test_compare_calculates_difference(self):
        r_a, r_b, ratio = compare("gpt-4o", "gpt-4o-mini",
                                  input_tokens=1_000_000, output_tokens=1_000_000)
        expected_diff = r_a.total_cost - r_b.total_cost
        self.assertGreater(expected_diff, 0.0)

    def test_compare_zero_output_cost(self):
        r_a, r_b, ratio = compare("gpt-4o", "gpt-4o-mini",
                                  input_tokens=1_000_000, output_tokens=0)
        # ratio is still computed; both have non-zero cost
        self.assertGreater(ratio, 1.0)


class TestBlendedCost(unittest.TestCase):

    def test_blended_cost_returns_float(self):
        result = blended_cost("gpt-4o", input_tokens=100_000, output_tokens=50_000)
        self.assertIsInstance(result, float)

    def test_blended_cost_matches_total_cost(self):
        from llm_cost_calc.calculator import cost
        blended = blended_cost("gpt-4o", input_tokens=100_000, output_tokens=50_000)
        full = cost("gpt-4o", input_tokens=100_000, output_tokens=50_000)
        self.assertEqual(blended, full.total_cost)


class TestTokenValidation(unittest.TestCase):
    """Negative / non-integer token counts must never yield negative costs."""

    def test_negative_input_tokens_raises(self):
        with self.assertRaises(ValueError):
            cost("gpt-4o", input_tokens=-1, output_tokens=0)

    def test_negative_output_tokens_raises(self):
        with self.assertRaises(ValueError):
            cost("gpt-4o", input_tokens=0, output_tokens=-1)

    def test_negative_tokens_raise_even_for_unknown_model(self):
        # Validation happens before the unknown-model check, so the caller
        # is told about the bad input regardless of model validity.
        with self.assertRaises(ValueError):
            cost("not-a-model", input_tokens=-100, output_tokens=0, known_only=False)

    def test_non_int_tokens_raise_typeerror(self):
        with self.assertRaises(TypeError):
            cost("gpt-4o", input_tokens=1.5, output_tokens=0)
        with self.assertRaises(TypeError):
            cost("gpt-4o", input_tokens=0, output_tokens="100")

    def test_bool_tokens_raise_typeerror(self):
        # bool is a subclass of int but is not a meaningful token count.
        with self.assertRaises(TypeError):
            cost("gpt-4o", input_tokens=True, output_tokens=0)

    def test_compare_rejects_negative_tokens(self):
        with self.assertRaises(ValueError):
            compare("gpt-4o", "gpt-4o-mini", input_tokens=-1, output_tokens=0)

    def test_blended_cost_rejects_negative_tokens(self):
        with self.assertRaises(ValueError):
            blended_cost("gpt-4o", input_tokens=-1, output_tokens=0)


if __name__ == "__main__":
    unittest.main()

"""Tests for llm_cost_calc.prices."""

import unittest

from llm_cost_calc.prices import PRICES, get_price, known_models


class TestGetPrice(unittest.TestCase):

    def test_known_model_returns_tuple(self):
        result = get_price("gpt-4o")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_gpt4o_prices(self):
        inp, out = get_price("gpt-4o")
        self.assertEqual(inp, 2.50)
        self.assertEqual(out, 10.00)

    def test_gpt4o_mini_prices(self):
        inp, out = get_price("gpt-4o-mini")
        self.assertEqual(inp, 0.15)
        self.assertEqual(out, 0.60)

    def test_claude35_sonnet_prices(self):
        inp, out = get_price("claude-3.5-sonnet")
        self.assertEqual(inp, 3.00)
        self.assertEqual(out, 15.00)

    def test_gemini15_flash_prices(self):
        inp, out = get_price("gemini-1.5-flash")
        self.assertEqual(inp, 0.075)
        self.assertEqual(out, 0.30)

    def test_deepseek_chat_prices(self):
        inp, out = get_price("deepseek-chat")
        self.assertEqual(inp, 0.14)
        self.assertEqual(out, 0.28)

    def test_unknown_model_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_price("nonexistent-model-xyz")

    def test_unknown_model_error_message(self):
        with self.assertRaises(KeyError) as ctx:
            get_price("unknown")
        self.assertIn("unknown", str(ctx.exception))


class TestKnownModels(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(known_models(), list)

    def test_sorted(self):
        models = known_models()
        self.assertEqual(models, sorted(models))

    def test_contains_expected_models(self):
        models = known_models()
        for m in ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "gemini-1.5-flash",
                  "deepseek-chat", "llama-3-8b-instruct"]:
            self.assertIn(m, models)

    def test_prices_table_consistency(self):
        for model, prices in PRICES.items():
            self.assertIsInstance(prices, tuple, msg=f"{model} value is not a tuple")
            self.assertEqual(len(prices), 2, msg=f"{model} does not have 2 price fields")
            inp, out = prices
            self.assertIsInstance(inp, (int, float), msg=f"{model} input price not numeric")
            self.assertIsInstance(out, (int, float), msg=f"{model} output price not numeric")
            self.assertGreaterEqual(inp, 0, msg=f"{model} input price negative")
            self.assertGreaterEqual(out, 0, msg=f"{model} output price negative")


if __name__ == "__main__":
    unittest.main()

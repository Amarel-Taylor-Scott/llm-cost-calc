"""Tests for llm_cost_calc.cli."""

import io
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr

from llm_cost_calc.calculator import UnknownModelError
from llm_cost_calc.cli import build_parser, cmd_cost, cmd_compare, cmd_list


class TestCLI(unittest.TestCase):

    def _run(self, argv):
        """Call the CLI's main() with argv, returning (exit_code, stdout, stderr)."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit:
            return 2, "", ""

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                if args.command == "cost":
                    code = cmd_cost(args)
                elif args.command == "compare":
                    code = cmd_compare(args)
                elif args.command == "list":
                    code = cmd_list(args)
                else:
                    parser.print_help(file=stdout_buf)
                    code = 1
            except UnknownModelError as exc:
                print(f"error: {exc}", file=stderr_buf)
                code = 1
        return code, stdout_buf.getvalue(), stderr_buf.getvalue()

    def test_cost_command_success(self):
        code, stdout, stderr = self._run(["cost", "gpt-4o", "100000", "50000"])
        self.assertEqual(code, 0, stderr)
        self.assertIn("gpt-4o", stdout)
        self.assertIn("$0.75", stdout)

    def test_cost_command_unknown_model(self):
        code, stdout, stderr = self._run(["cost", "not-a-model", "1000", "500"])
        self.assertEqual(code, 1)
        self.assertIn("unknown model", stderr)

    def test_cost_command_invalid_tokens(self):
        code, stdout, stderr = self._run(["cost", "gpt-4o", "abc", "500"])
        self.assertEqual(code, 2)

    def test_cost_command_negative_tokens(self):
        code, stdout, stderr = self._run(["cost", "gpt-4o", "-100", "500"])
        self.assertEqual(code, 2)

    def test_compare_command_success(self):
        code, stdout, stderr = self._run(
            ["compare", "gpt-4o", "gpt-4o-mini", "100000", "50000"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("gpt-4o", stdout)
        self.assertIn("gpt-4o-mini", stdout)
        self.assertIn("16.", stdout)  # ratio is ~16.67

    def test_compare_command_unknown_model(self):
        code, stdout, stderr = self._run(
            ["compare", "gpt-4o", "unknown-model", "1000", "500"]
        )
        self.assertEqual(code, 1)
        self.assertIn("unknown model", stderr)

    def test_list_command(self):
        code, stdout, stderr = self._run(["list"])
        self.assertEqual(code, 0, stderr)
        self.assertIn("gpt-4o", stdout)
        self.assertIn("claude-3.5-sonnet", stdout)
        self.assertIn("deepseek-chat", stdout)

    def test_no_args_shows_help(self):
        code, stdout, stderr = self._run([])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

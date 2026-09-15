"""The calculator must be exact, and must refuse to be a code execution hole."""

from __future__ import annotations

import math

import pytest

from gemini_agent.tools.calc import CalculationError, calculate, evaluate


class TestArithmetic:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("2+2", 4),
            ("15/100 * 2400", 360),
            ("(1 + 0.07) ** 30", 7.612255042918417),
            ("2 ** 10", 1024),
            ("-5 // 2", -3),
            ("10 % 3", 1),
            ("7 / 2", 3.5),
        ],
    )
    def test_basic_operators(self, expression, expected):
        assert evaluate(expression) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("sqrt(144)", 12),
            ("log(1000, 10)", 3),
            ("hypot(3, 4)", 5),
            ("max(1, 99, 3)", 99),
            ("min(1, 99, 3)", 1),
            ("sum(1, 2, 3)", 6),
            ("round(2.34567, 2)", 2.35),
            ("gcd(12, 18)", 6),
            ("lcm(4, 6)", 12),
            ("factorial(10)", 3628800),
            ("degrees(pi)", 180),
            ("atan2(1, 1)", math.pi / 4),
        ],
    )
    def test_functions_and_constants(self, expression, expected):
        assert evaluate(expression) == pytest.approx(expected)

    def test_integers_stay_exact(self):
        # Floating point would round this; the calculator must not.
        assert evaluate("2 ** 100") == 1267650600228229401496703205376

    @pytest.mark.parametrize(
        "written, canonical",
        [("2^10", "2**10"), ("1,250,000 / 4", "1250000 / 4")],
    )
    def test_friendly_notation(self, written, canonical):
        assert evaluate(written) == evaluate(canonical)

    def test_commas_are_not_stripped_inside_calls(self):
        """The bug this guards: "hypot(3,400)" must not become "hypot(3400)"."""
        assert evaluate("hypot(3,400)") == pytest.approx(math.hypot(3, 400))
        assert evaluate("hypot(3,400)") != 3400


class TestSafety:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('ls')",
            "(1).__class__.__bases__",
            "open('/etc/passwd')",
            "eval('1+1')",
            "lambda: 1",
            "[x for x in range(10)]",
            "1 if True else 2",
            "a.b.c",
            "'text' * 3",
            "True + True",
            "x + 1",
        ],
    )
    def test_refuses_anything_that_is_not_arithmetic(self, expression):
        with pytest.raises(CalculationError):
            evaluate(expression)

    @pytest.mark.parametrize(
        "expression",
        ["9**9**9", "10 ** 100000", "factorial(100000)", "2 ** 999999"],
    )
    def test_refuses_resource_exhaustion(self, expression):
        """Valid arithmetic that would pin a CPU must fail fast, not hang."""
        with pytest.raises(CalculationError):
            evaluate(expression)

    @pytest.mark.parametrize(
        "expression", ["1/0", "sqrt(-1)", "log(0)", "acos(5)", "2 +", ""]
    )
    def test_undefined_and_malformed(self, expression):
        with pytest.raises(CalculationError):
            evaluate(expression)

    def test_expression_length_is_bounded(self):
        with pytest.raises(CalculationError):
            evaluate("1+" * 400 + "1")


class TestToolWrapper:
    def test_success_shows_the_working(self):
        assert calculate("15/100 * 2400") == "15/100 * 2400 = 360"

    def test_failure_is_an_error_string_not_an_exception(self):
        """Tool failures are fed back to the model, so they must be describable."""
        result = calculate("__import__('os')")
        assert result.startswith("ERROR:")
        assert "not an available function" in result

    def test_error_lists_what_is_available_so_the_model_can_retry(self):
        assert "sqrt" in calculate("wibble(2)")

    def test_float_formatting_is_honest(self):
        assert calculate("1/3").endswith("0.3333333333")
        assert calculate("10/2") == "10/2 = 5"

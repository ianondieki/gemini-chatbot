"""A calculator the model can trust, and that cannot be turned into a shell.

``eval()`` on model-supplied text is remote code execution, so the expression is
parsed into an AST and walked by hand, allowing only arithmetic, a fixed set of
maths functions, and a few constants. Everything else — attribute access,
subscripting, comprehensions, lambdas, names we did not whitelist — is refused
by construction rather than by blacklist.

The second hazard is cheaper to overlook: ``9**9**9`` is perfectly valid
arithmetic that will pin a CPU and exhaust memory. Exponent, factorial and
shift magnitudes are bounded before evaluation, so a hostile or merely careless
expression fails fast with an explanation instead of hanging the agent.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable, Dict

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 1_000
MAX_FACTORIAL = 500
MAX_RESULT_DIGITS = 4_000

_BINARY: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY: Dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *args: sum(args),
    "pow": pow,
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "exp": math.exp,
    "log": math.log,       # log(x) or log(x, base)
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "hypot": math.hypot,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "factorial": math.factorial,
    "fabs": math.fabs,
    "copysign": math.copysign,
}


class CalculationError(ValueError):
    """The expression is not something this calculator will evaluate."""


def _check_magnitude(value: Any) -> Any:
    """Refuse integers so large that printing them would itself be the problem."""
    if isinstance(value, int) and value != 0:
        if value.bit_length() > MAX_RESULT_DIGITS * 4:
            raise CalculationError(
                "the result is too large to be useful (millions of digits)"
            )
    return value


def _guard_pow(base: Any, exponent: Any) -> Any:
    if isinstance(exponent, (int, float)) and abs(exponent) > MAX_EXPONENT:
        raise CalculationError(
            f"exponent {exponent} exceeds the limit of {MAX_EXPONENT}"
        )
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and base != 0
        and base.bit_length() * exponent > MAX_RESULT_DIGITS * 4
    ):
        raise CalculationError("that power would produce an impractically large number")
    return _check_magnitude(operator.pow(base, exponent))


def _call_function(name: str, args: list) -> Any:
    if name == "factorial":
        if not args or not isinstance(args[0], int) or args[0] > MAX_FACTORIAL:
            raise CalculationError(
                f"factorial needs a whole number no larger than {MAX_FACTORIAL}"
            )
    try:
        return _check_magnitude(_FUNCTIONS[name](*args))
    except CalculationError:
        raise
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise CalculationError(f"{name}() is undefined for those inputs ({exc})")
    except TypeError as exc:
        raise CalculationError(f"{name}() got the wrong number or type of arguments ({exc})")


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("only numbers are allowed as values")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY:
            raise CalculationError(f"the {op_type.__name__} operator is not allowed")
        left, right = _eval(node.left), _eval(node.right)
        if op_type is ast.Pow:
            return _guard_pow(left, right)
        try:
            return _check_magnitude(_BINARY[op_type](left, right))
        except ZeroDivisionError:
            raise CalculationError("division by zero")
        except (OverflowError, ValueError) as exc:
            raise CalculationError(str(exc))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY:
            raise CalculationError(f"the {op_type.__name__} operator is not allowed")
        return _UNARY[op_type](_eval(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculationError(
            f"'{node.id}' is not a known constant. Known: "
            + ", ".join(sorted(_CONSTANTS))
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculationError("only plain function calls like sqrt(2) are allowed")
        name = node.func.id
        if name not in _FUNCTIONS:
            raise CalculationError(
                f"'{name}' is not an available function. Available: "
                + ", ".join(sorted(_FUNCTIONS))
            )
        if node.keywords:
            raise CalculationError("keyword arguments are not supported")
        return _call_function(name, [_eval(arg) for arg in node.args])

    if isinstance(node, (ast.Tuple, ast.List)):
        raise CalculationError(
            "lists and tuples are not supported — if you meant a large number, "
            "write it without digit separators (1000000, not 1,000,000)"
        )

    raise CalculationError(
        f"{type(node).__name__} is not allowed — this calculator only does arithmetic"
    )


def evaluate(expression: str) -> Any:
    """Evaluate an arithmetic expression safely. Raises :class:`CalculationError`."""
    if not expression or not expression.strip():
        raise CalculationError("the expression is empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculationError(
            f"the expression is longer than {MAX_EXPRESSION_CHARS} characters"
        )

    cleaned = expression.strip().replace("^", "**").replace("×", "*")
    cleaned = cleaned.replace("÷", "/")

    # Strip digit-grouping commas ("1,250,000") but ONLY when the expression
    # contains no function calls. Otherwise "hypot(3,400)" would quietly become
    # "hypot(3400)" — a wrong answer, which is far worse than a clear refusal.
    if not re.search(r"[A-Za-z]", cleaned):
        cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", cleaned)

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"that is not a valid expression ({exc.msg})")

    return _eval(tree.body)


def format_result(value: Any) -> str:
    """Render a result without lying about precision."""
    if isinstance(value, float):
        if value != value:          # NaN
            return "undefined (not a number)"
        if value in (math.inf, -math.inf):
            return "infinity" if value > 0 else "-infinity"
        if value == int(value) and abs(value) < 1e16:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


DESCRIPTION = (
    "Evaluate a mathematical expression exactly. Use this for ALL arithmetic "
    "rather than working it out yourself, however simple it looks. Supports "
    "+ - * / // % ** and parentheses, the constants pi, e and tau, and the "
    "functions sqrt, cbrt, exp, log, log2, log10, sin, cos, tan, asin, acos, "
    "atan, atan2, sinh, cosh, tanh, degrees, radians, floor, ceil, trunc, "
    "round, abs, min, max, sum, pow, hypot, gcd, lcm and factorial. "
    "Examples: '15/100 * 2400', 'sqrt(2) * 10', 'log(1000, 10)', "
    "'(1 + 0.07) ** 30'."
)


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression exactly and return the result."""
    try:
        return f"{expression.strip()} = {format_result(evaluate(expression))}"
    except CalculationError as exc:
        return f"ERROR: cannot evaluate '{expression}' — {exc}"


def register(registry) -> None:
    """Add the calculator to a registry."""
    registry.tool(
        name="calculate",
        description=DESCRIPTION,
        describe={
            "expression": (
                "A single mathematical expression, with no variables or "
                "assignment, e.g. '2400 * 0.15' or 'sqrt(144) + log(100, 10)'."
            )
        },
    )(calculate)

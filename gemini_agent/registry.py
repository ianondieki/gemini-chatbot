"""A tool registry that turns ordinary Python functions into model-callable tools.

The original code declared every tool twice — once as a
``types.FunctionDeclaration`` and once as a branch in an ``if name == ...``
router — in two files. Here a tool is declared once:

    @registry.tool(describe={"expression": "A pure math expression"})
    def calculate(expression: str) -> str:
        '''Evaluate a math expression and return the exact result.'''

The decorator reads the signature for parameter names, types and
required-ness, the docstring for the description, and builds the Gemini schema
itself. Dispatch, argument validation and error shaping come for free.

Validation matters more than it looks: when a model sends ``{"expression": 12}``
or forgets a required field, the registry returns a message that *names the
mistake and the expected shape*, and the agent feeds that back as an
observation. That is what lets the model repair its own call on the next step
instead of the turn dying.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, get_args,
    get_origin, get_type_hints,
)

from google.genai import types

from .errors import InvalidToolArguments, ToolNotFound

# The parameter name a tool declares when it wants the shared ToolContext.
# It is stripped from the model-facing schema.
CONTEXT_PARAM = "ctx"

_SCALARS: Dict[Any, Any] = {
    str: types.Type.STRING,
    int: types.Type.INTEGER,
    float: types.Type.NUMBER,
    bool: types.Type.BOOLEAN,
}

_TYPE_NAMES = {
    types.Type.STRING: "string",
    types.Type.INTEGER: "integer",
    types.Type.NUMBER: "number",
    types.Type.BOOLEAN: "boolean",
    types.Type.ARRAY: "array",
}


@dataclass
class ToolResult:
    """The outcome of one tool execution, in the form the model will see."""

    name: str
    ok: bool
    content: str
    duration: float = 0.0
    cached: bool = False

    def preview(self, limit: int = 160) -> str:
        flat = " ".join(self.content.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., str]
    declaration: types.FunctionDeclaration
    required: List[str]
    param_types: Dict[str, Any]
    wants_context: bool
    signature_hint: str
    concurrent_safe: bool = True
    cacheable: bool = True

    def describe(self) -> str:
        return f"{self.name}({self.signature_hint}) — {self.description}"


def _annotation_to_schema(annotation: Any, description: str) -> types.Schema:
    """Map a Python annotation onto a Gemini parameter schema."""
    if annotation in _SCALARS:
        return types.Schema(type=_SCALARS[annotation], description=description)

    origin = get_origin(annotation)
    if origin in (list, List, Sequence):
        (inner,) = get_args(annotation) or (str,)
        item_type = _SCALARS.get(inner, types.Type.STRING)
        return types.Schema(
            type=types.Type.ARRAY,
            description=description,
            items=types.Schema(type=item_type),
        )

    # Anything exotic is passed as a string; the tool can parse it itself.
    return types.Schema(type=types.Type.STRING, description=description)


def _coerce(value: Any, annotation: Any, param: str, tool: str) -> Any:
    """Nudge a model-supplied value into the declared type, or explain why not.

    Models routinely send ``"5"`` where an int is wanted, or ``5`` where a
    string is. Rejecting those outright wastes a whole round trip, so we coerce
    what is unambiguous and refuse only what is genuinely wrong.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return value

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise InvalidToolArguments(
            f"{tool}: parameter '{param}' must be true or false, got {value!r}"
        )

    if annotation in (int, float):
        if isinstance(value, bool):
            raise InvalidToolArguments(
                f"{tool}: parameter '{param}' must be a number, got a boolean"
            )
        if isinstance(value, (int, float)):
            return annotation(value)
        if isinstance(value, str):
            try:
                return annotation(value.strip())
            except ValueError:
                pass
        raise InvalidToolArguments(
            f"{tool}: parameter '{param}' must be a number, got {value!r}"
        )

    if annotation is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        raise InvalidToolArguments(
            f"{tool}: parameter '{param}' must be a string, got {type(value).__name__}"
        )

    origin = get_origin(annotation)
    if origin in (list, List, Sequence):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise InvalidToolArguments(
                f"{tool}: parameter '{param}' must be a list, got {type(value).__name__}"
            )
        (inner,) = get_args(annotation) or (str,)
        return [_coerce(item, inner, param, tool) for item in value]

    return value


class ToolRegistry:
    """Holds tool specs, builds the Gemini declaration, and dispatches calls."""

    def __init__(self, specs: Optional[Mapping[str, ToolSpec]] = None) -> None:
        self._specs: Dict[str, ToolSpec] = dict(specs or {})

    # --- registration --------------------------------------------------
    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        describe: Optional[Mapping[str, str]] = None,
        concurrent_safe: bool = True,
        cacheable: bool = True,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """Decorator registering a function as a model-callable tool.

        ``concurrent_safe=False`` forces the batch it appears in to run
        serially; ``cacheable=False`` stops the agent reusing an identical
        earlier result within the same turn (right for a clock, wrong for a
        search).
        """

        def decorator(fn: Callable[..., str]) -> Callable[..., str]:
            self.register(
                build_spec(
                    fn,
                    name=name,
                    description=description,
                    describe=describe,
                    concurrent_safe=concurrent_safe,
                    cacheable=cacheable,
                )
            )
            return fn

        return decorator

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def remove(self, name: str) -> None:
        self._specs.pop(name, None)

    # --- inspection ----------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def names(self) -> List[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolNotFound(name) from exc

    def subset(self, names: Sequence[str]) -> "ToolRegistry":
        """A registry with only the named tools — how sub-agents get narrowed."""
        return ToolRegistry({n: self._specs[n] for n in names if n in self._specs})

    def catalogue(self) -> str:
        """A compact listing for prompts, so the planner knows what exists."""
        return "\n".join(f"- {self._specs[n].describe()}" for n in self.names)

    def declarations(self) -> Optional[types.Tool]:
        """The ``types.Tool`` to hand Gemini, or ``None`` when empty."""
        if not self._specs:
            return None
        return types.Tool(
            function_declarations=[self._specs[n].declaration for n in self.names]
        )

    # --- dispatch ------------------------------------------------------
    def invoke(self, name: str, args: Mapping[str, Any], ctx: Any = None) -> ToolResult:
        """Run a tool, never raising: failures come back as ``ok=False`` results.

        The agent feeds the content back to the model either way, so a failure
        has to be *described*, not thrown.
        """
        started = time.monotonic()
        spec = self._specs.get(name)
        if spec is None:
            known = ", ".join(self.names) or "none"
            return ToolResult(
                name=name,
                ok=False,
                content=(
                    f"ERROR: there is no tool called '{name}'. "
                    f"Available tools: {known}."
                ),
                duration=time.monotonic() - started,
            )

        try:
            call_args = self._prepare_args(spec, args)
        except InvalidToolArguments as exc:
            return ToolResult(
                name=name,
                ok=False,
                content=f"ERROR: {exc} Expected: {spec.name}({spec.signature_hint}).",
                duration=time.monotonic() - started,
            )

        if spec.wants_context:
            call_args[CONTEXT_PARAM] = ctx

        try:
            output = spec.fn(**call_args)
        except Exception as exc:  # a tool blowing up must not end the turn
            return ToolResult(
                name=name,
                ok=False,
                content=f"ERROR: {spec.name} failed: {type(exc).__name__}: {exc}",
                duration=time.monotonic() - started,
            )

        return ToolResult(
            name=name,
            ok=True,
            content="" if output is None else str(output),
            duration=time.monotonic() - started,
        )

    def _prepare_args(
        self, spec: ToolSpec, args: Mapping[str, Any]
    ) -> Dict[str, Any]:
        supplied = {k: v for k, v in (args or {}).items() if v is not None}

        missing = [p for p in spec.required if p not in supplied]
        if missing:
            raise InvalidToolArguments(
                f"{spec.name}: missing required parameter(s) "
                f"{', '.join(repr(m) for m in missing)}."
            )

        unknown = [k for k in supplied if k not in spec.param_types]
        if unknown:
            raise InvalidToolArguments(
                f"{spec.name}: unexpected parameter(s) "
                f"{', '.join(repr(u) for u in unknown)}."
            )

        return {
            key: _coerce(value, spec.param_types[key], key, spec.name)
            for key, value in supplied.items()
        }


def build_spec(
    fn: Callable[..., str],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    describe: Optional[Mapping[str, str]] = None,
    concurrent_safe: bool = True,
    cacheable: bool = True,
) -> ToolSpec:
    """Derive a :class:`ToolSpec` from a function's signature and docstring."""
    tool_name = name or fn.__name__
    doc = inspect.getdoc(fn) or ""
    tool_description = description or doc.split("\n\n")[0].strip()
    if not tool_description:
        raise ValueError(
            f"tool {tool_name!r} needs a description, either as an argument "
            "or as the first paragraph of its docstring"
        )

    descriptions = dict(describe or {})
    signature = inspect.signature(fn)

    # Tool modules use ``from __future__ import annotations``, which makes every
    # annotation a *string*. Resolving them here is what keeps ``int`` from
    # silently being declared to the model as a string parameter.
    try:
        resolved = get_type_hints(fn)
    except Exception:  # an annotation referring to something not importable
        resolved = {}

    properties: Dict[str, types.Schema] = {}
    required: List[str] = []
    param_types: Dict[str, Any] = {}
    hints: List[str] = []
    wants_context = False

    for param_name, param in signature.parameters.items():
        if param_name == CONTEXT_PARAM:
            wants_context = True
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(
                f"tool {tool_name!r} may not use *args/**kwargs — the model "
                "needs a concrete schema"
            )

        annotation = resolved.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            annotation = str
        param_types[param_name] = annotation
        text = descriptions.get(param_name, f"The {param_name.replace('_', ' ')}.")
        schema = _annotation_to_schema(annotation, text)
        properties[param_name] = schema

        type_label = _TYPE_NAMES.get(schema.type, "string")
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
            hints.append(f"{param_name}: {type_label}")
        else:
            hints.append(f"{param_name}: {type_label} = {param.default!r}")

    declaration = types.FunctionDeclaration(
        name=tool_name,
        description=tool_description,
        parameters=(
            types.Schema(
                type=types.Type.OBJECT,
                properties=properties,
                required=required or None,
            )
            if properties
            else None
        ),
    )

    return ToolSpec(
        name=tool_name,
        description=tool_description,
        fn=fn,
        declaration=declaration,
        required=required,
        param_types=param_types,
        wants_context=wants_context,
        signature_hint=", ".join(hints),
        concurrent_safe=concurrent_safe,
        cacheable=cacheable,
    )

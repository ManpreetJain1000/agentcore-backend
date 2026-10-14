"""Tool registry, auto-generated tool schemas, async dispatcher and tool implementations."""
import ast
import asyncio
import inspect
import json
import logging
import math
import operator
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

from app.agent.memory_manager import LongTermMemoryStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.agent.planner import LLMClient

logger = logging.getLogger(__name__)


# --- Registry ---------------------------------------------------------------

@dataclass(frozen=True)
class ToolContext:
    """Runtime context injected into tools that declare a ToolContext parameter."""

    user_id: str
    session_id: str
    run_id: str
    db: "AsyncSession"
    llm: "LLMClient"


@dataclass(frozen=True)
class Tool:
    name: str
    func: Callable[..., Any]
    args_model: type[BaseModel]
    definition: dict[str, Any]
    is_async: bool
    context_param: str | None
    timeout: float | None


TOOL_REGISTRY: dict[str, Tool] = {}


def _strip_titles(node: Any) -> Any:
    """Remove Pydantic's auto-generated 'title' keys from a JSON schema."""
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title":
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {name: _strip_titles(prop) for name, prop in value.items()}
            else:
                cleaned[key] = _strip_titles(value)
        return cleaned
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


def _build_tool(func: Callable[..., Any], timeout: float | None) -> Tool:
    hints = get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    context_param: str | None = None

    for name, param in inspect.signature(func).parameters.items():
        annotation = hints.get(name, str)
        if annotation is ToolContext:
            context_param = name
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)

    args_model = create_model(f"{func.__name__}_args", **fields)
    schema = _strip_titles(args_model.model_json_schema())

    function_def: dict[str, Any] = {
        "name": func.__name__,
        "description": inspect.getdoc(func) or "",
    }
    if schema.get("properties"):
        function_def["parameters"] = {
            "type": "object",
            "properties": schema["properties"],
            "required": schema.get("required", []),
        }

    return Tool(
        name=func.__name__,
        func=func,
        args_model=args_model,
        definition={"type": "function", "function": function_def},
        is_async=inspect.iscoroutinefunction(func),
        context_param=context_param,
        timeout=timeout,
    )


def tool(func: Callable[..., Any] | None = None, *, timeout: float | None = None):
    """Register a function as an agent tool. Use as @tool or @tool(timeout=...)."""

    def register(f: Callable[..., Any]) -> Callable[..., Any]:
        TOOL_REGISTRY[f.__name__] = _build_tool(f, timeout)
        return f

    return register(func) if func is not None else register


def available_tools() -> list[str]:
    return sorted(TOOL_REGISTRY)


def unknown_tools(names: Iterable[str]) -> list[str]:
    return [name for name in names if name not in TOOL_REGISTRY]


def get_tool_definitions(names: Iterable[str]) -> list[dict[str, Any]]:
    return [TOOL_REGISTRY[name].definition for name in names if name in TOOL_REGISTRY]


def get_tool_timeout(name: str) -> float | None:
    registered = TOOL_REGISTRY.get(name)
    return registered.timeout if registered else None


async def dispatch_tool(name: str, arguments: str, ctx: ToolContext) -> str:
    """Run a tool by name with JSON arguments. Never raises; errors come back as strings."""
    registered = TOOL_REGISTRY.get(name)
    if registered is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(available_tools())}"

    try:
        raw_args = json.loads(arguments or "{}")
        if not isinstance(raw_args, dict):
            raise ValueError("tool arguments must be a JSON object")
        kwargs = registered.args_model.model_validate(raw_args).model_dump()
        if registered.context_param:
            kwargs[registered.context_param] = ctx

        if registered.is_async:
            result = await registered.func(**kwargs)
        else:
            result = await asyncio.to_thread(registered.func, **kwargs)
        return str(result)
    except json.JSONDecodeError as exc:
        return f"Error: arguments for '{name}' are not valid JSON ({exc.msg})"
    except ValidationError as exc:
        return f"Error: invalid arguments for '{name}': {exc.errors(include_url=False)}"
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the agent loop
        logger.exception("Tool %s failed", name)
        return f"Error: tool '{name}' failed: {type(exc).__name__}: {exc}"


# --- Calculator internals ---------------------------------------------------

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_RESULT_BITS = 100_000


def _normalise_expression(expression: str) -> str:
    expr = expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # 3,600,000 -> 3600000
    # "15%" (percent, not modulo) -> (15/100)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d.(])", r"(\1/100)", expr)
    return expr


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            if isinstance(left, int) and left.bit_length() * abs(right) > MAX_RESULT_BITS:
                raise ValueError("result is too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        return _BINARY_OPS[type(node.op)](left, right)

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval_node(arg) for arg in node.args))

    raise ValueError(f"unsupported element in expression: {type(node).__name__}")


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12g}"


# --- Mock search data -------------------------------------------------------

_MOCK_SEARCH_DATA: dict[str, list[str]] = {
    "dubai": [
        "Dubai Statistics Center: Dubai's population is approximately 3,800,000 (3.8 million) residents.",
        "Dubai is the most populous city in the United Arab Emirates, with steady population growth driven by migration.",
        "Around 85-90% of Dubai's residents are expatriates, making it one of the most diverse cities in the world.",
    ],
}


# --- Tools ------------------------------------------------------------------

@tool
def web_search(query: Annotated[str, Field(description="The search query", min_length=1)]) -> str:
    """Search the web and return the top 3 result snippets for a query."""
    lowered = query.lower()
    snippets = next(
        (data for keyword, data in _MOCK_SEARCH_DATA.items() if keyword in lowered),
        [
            f"Overview: general background information about '{query}'.",
            f"Latest news: recent developments and figures related to '{query}'.",
            f"Reference: detailed explanation and key facts about '{query}'.",
        ],
    )
    lines = [
        f"{i}. {snippet} (source: https://example.com/search/{i})"
        for i, snippet in enumerate(snippets[:3], start=1)
    ]
    return f"Search results for '{query}':\n" + "\n".join(lines)


@tool
def calculator(
    expression: Annotated[
        str,
        Field(description="Arithmetic expression, e.g. '0.15 * 3800000' or 'sqrt(16) + 2**3'"),
    ],
) -> str:
    """Safely evaluate an arithmetic expression. Supports + - * / // % **, parentheses,
    abs, round, min, max, sqrt, log, log10, exp, floor, ceil and the constants pi and e."""
    expr = _normalise_expression(expression)
    if not expr:
        raise ValueError("expression is empty")
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {expression!r}") from exc
    return _format_number(_eval_node(tree))


@tool
def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool(timeout=30.0)
async def summarise_text(
    text: Annotated[str, Field(description="The text to summarise", min_length=1)],
    ctx: ToolContext,
    max_words: Annotated[int, Field(description="Maximum words in the summary", ge=1, le=1000)] = 100,
) -> str:
    """Summarise the given text in at most max_words words."""
    if len(text.split()) <= max_words:
        return text.strip()

    response = await ctx.llm.chat(
        [
            {
                "role": "system",
                "content": "You summarise text faithfully and concisely. Reply with the summary only.",
            },
            {
                "role": "user",
                "content": f"Summarise the following text in at most {max_words} words:\n\n{text}",
            },
        ]
    )
    summary = (response.content or "").strip()
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "..."
    return summary


@tool(timeout=20.0)
async def remember_fact(
    fact: Annotated[
        str,
        Field(description="A single, self-contained fact about the user worth remembering", min_length=1),
    ],
    ctx: ToolContext,
) -> str:
    """Store a fact about the user in long-term memory so it can be recalled in future conversations."""
    fact = fact.strip()
    store = LongTermMemoryStore(ctx.db, ctx.llm)
    await store.add(ctx.user_id, fact, source_run_id=ctx.run_id)
    return f"Remembered: {fact}"
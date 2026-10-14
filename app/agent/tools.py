"""Tool registry, auto-generated tool schemas and async dispatcher."""
import asyncio
import inspect
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.agent.planner import LLMClient

logger = logging.getLogger(__name__)


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


def _build_tool(func: Callable[..., Any]) -> Tool:
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
    )


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a function as an agent tool. The function name is the tool name."""
    TOOL_REGISTRY[func.__name__] = _build_tool(func)
    return func


def available_tools() -> list[str]:
    return sorted(TOOL_REGISTRY)


def unknown_tools(names: Iterable[str]) -> list[str]:
    return [name for name in names if name not in TOOL_REGISTRY]


def get_tool_definitions(names: Iterable[str]) -> list[dict[str, Any]]:
    return [TOOL_REGISTRY[name].definition for name in names if name in TOOL_REGISTRY]


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


# --- Tools ------------------------------------------------------------------

@tool
def web_search(query: Annotated[str, Field(description="The search query")]) -> str:
    """Search the web and return the top result snippets for a query."""
    raise NotImplementedError("web_search is not implemented")


@tool
def calculator(
    expression: Annotated[str, Field(description="Arithmetic expression, e.g. '0.15 * 3600000'")],
) -> str:
    """Safely evaluate a basic arithmetic expression and return the result."""
    raise NotImplementedError("calculator is not implemented")


@tool
def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO 8601 timestamp."""
    raise NotImplementedError("get_current_datetime is not implemented")


@tool
async def summarise_text(
    text: Annotated[str, Field(description="The text to summarise")],
    ctx: ToolContext,
    max_words: Annotated[int, Field(description="Maximum words in the summary", ge=1)] = 100,
) -> str:
    """Summarise the given text in at most max_words words."""
    raise NotImplementedError("summarise_text is not implemented")


@tool
async def remember_fact(
    fact: Annotated[str, Field(description="A single fact about the user worth remembering")],
    ctx: ToolContext,
) -> str:
    """Store a fact about the user in long-term memory for future conversations."""
    raise NotImplementedError("remember_fact is not implemented")
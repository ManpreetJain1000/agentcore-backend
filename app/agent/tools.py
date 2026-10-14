"""Tool registry.

Feature 2: registry + tool names (used to validate tools_enabled).
Feature 4: real implementations, auto-generated JSON schemas, async dispatcher.
"""
from collections.abc import Callable, Iterable

TOOL_REGISTRY: dict[str, Callable[..., str]] = {}


def tool(func: Callable[..., str]) -> Callable[..., str]:
    """Register a function as an agent tool. The function name is the tool name."""
    TOOL_REGISTRY[func.__name__] = func
    return func


def available_tools() -> list[str]:
    return sorted(TOOL_REGISTRY)


def unknown_tools(names: Iterable[str]) -> list[str]:
    return [name for name in names if name not in TOOL_REGISTRY]



@tool
def web_search(query: str) -> str:
    """Search the web and return the top result snippets for a query."""
    raise NotImplementedError


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression, e.g. '0.15 * 3600000'."""
    raise NotImplementedError


@tool
def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO 8601 timestamp."""
    raise NotImplementedError


@tool
def summarise_text(text: str, max_words: int) -> str:
    """Summarise the given text in at most max_words words."""
    raise NotImplementedError


@tool
def remember_fact(fact: str) -> str:
    """Store a fact about the user in long-term memory for future conversations."""
    raise NotImplementedError
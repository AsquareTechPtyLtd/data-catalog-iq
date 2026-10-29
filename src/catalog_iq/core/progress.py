"""Progress events: where a request is right now, for a client that streams (/ask/stream).

The router and the agent loop call `progress(step, text)` at each step: routing, each tool
call (as a plain phrase), the answer check, a retry and why, a budget stop. With no sink
attached (CLI, evals, POST /ask) it does nothing. Events go only to the asking user, so
they carry real values (traces stay tokenized).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

_sink: ContextVar[Callable[[dict], None] | None] = ContextVar("progress_sink", default=None)
_route: ContextVar[int | None] = ContextVar("progress_route", default=None)


@contextmanager
def attach(sink: Callable[[dict], None]):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


@contextmanager
def for_route(index: int):
    """Tag events with the route (0-based, in the order the user asked) they belong to."""
    token = _route.set(index)
    try:
        yield
    finally:
        _route.reset(token)


def emit(event: dict) -> None:
    sink = _sink.get()
    if sink is not None:
        route = _route.get()
        sink({**event, **({"route_index": route} if route is not None else {})})


def progress(step: str, text: str) -> None:
    emit({"event": "progress", "step": step, "text": text})


# --- plain phrases for tool calls ----------------------------------------------------

_FILTER_WORDS = {"certification": "{}", "layer": "{} layer", "status": "{}", "domain": "{} domain",
                 "sensitivity": "{}", "has_contract": "with a contract", "owner_status": "owner {}",
                 "owner": "owned by {}", "data_category": "data category {}", "gap_type": "gap: {}",
                 "pi_type": "{}", "data_subject": "{}", "pi_status": "PI {}"}


def describe_tool(name: str, args) -> str:
    a = args if isinstance(args, dict) else {}
    table = a.get("table", "")
    if name == "find_table":
        what = "dashboards and apps" if a.get("kind") == "exposure" else "tables"
        return f"Searching {what} for '{a.get('query', '')}'"
    if name == "get_table":
        return f"Reading governance facts for {table}"
    if name == "get_lineage":
        direction = a.get("direction")
        return f"Walking the lineage of {table}" + (f" {direction}" if direction else "")
    if name == "trace_path":
        return f"Tracing a path from {a.get('source', '')} to {a.get('target', '')}"
    if name == "filter_tables":
        parts = [_FILTER_WORDS[k].format(v) for k, v in a.items()
                 if k in _FILTER_WORDS and v not in (None, False, "")]
        return "Filtering tables" + (f": {', '.join(map(str, parts))}" if parts else "")
    if name == "get_columns":
        return f"Reading the columns of {table}"
    if name == "get_data_category_evidence":
        return f"Gathering evidence about {table}"
    if name == "search_data_categories":
        return f"Searching data categories for '{a.get('query', '')}'"
    if name == "get_data_category":
        return f"Reading data category {a.get('data_category_id', '')}"
    return f"Calling {name}"


_RETRY_REASONS = {"unsupported_claim": "a claim wasn't in its cited source",
                  "unsupported_relation": "a lineage link wasn't cited",
                  "invalid_draft": "the answer wasn't in the expected shape",
                  "table_not_returned": "it named a table no tool returned",
                  "list_incomplete": "it left out tables the tool returned"}


def retry_reason(checks: list[str]) -> str:
    for c in checks:
        if c in _RETRY_REASONS:
            return _RETRY_REASONS[c]
    return f"it failed a check ({', '.join(checks)})" if checks else "it failed a check"

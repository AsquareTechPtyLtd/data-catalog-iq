"""Tracing: OpenTelemetry spans with OpenInference attributes, viewed in Arize Phoenix.

One question becomes one trace:

    ask                       AGENT      question → response
     ├─ llm                   LLM        messages → reply, tokens, ttft
     ├─ tool:find_table       TOOL       args → result (or the guardrail's block)
     └─ guardrail:output      GUARDRAIL  draft → events

Off by default. `setup_tracing()` turns it on when PHOENIX_COLLECTOR_ENDPOINT
is set (e.g. http://localhost:6006) and the tracing packages are installed
(pip install -e ".[trace]"). When off, `span()` yields a no-op, so the agent
runs the same with or without Phoenix.

Span attributes hold tokens, not people's names or emails (core.redaction), unless
TRACE_REDACTION=off. With no redaction scope open, values are withheld rather than written.
"""

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass

from pydantic import BaseModel

from catalog_iq.core.redaction import current_vault, trace_redaction

# OpenInference attribute names (plain strings, so the semconv package stays optional)
KIND = "openinference.span.kind"
INPUT, INPUT_MIME = "input.value", "input.mime_type"
OUTPUT, OUTPUT_MIME = "output.value", "output.mime_type"
JSON_MIME = "application/json"
WITHHELD = "[withheld: no redaction scope]"

_tracer = None
_provider = None


def setup_tracing(project: str = "data-catalog-iq", tracer_provider=None) -> bool:
    """Turn tracing on. Pass `tracer_provider` in tests; otherwise Phoenix is used if configured."""
    global _tracer, _provider
    if tracer_provider is None:
        if not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
            return False
        try:
            from phoenix.otel import register
        except ImportError:
            return False
        tracer_provider = register(project_name=os.environ.get("PHOENIX_PROJECT_NAME", project),
                                   batch=True, set_global_tracer_provider=False, verbose=False)
    _provider = tracer_provider
    _tracer = tracer_provider.get_tracer("catalog_iq")
    return True


def shutdown_tracing() -> None:
    """Send buffered spans before the process exits."""
    global _tracer, _provider
    if _provider is not None:
        _provider.force_flush()
    _tracer = _provider = None


def _value(obj) -> str:
    def default(o):
        if isinstance(o, BaseModel):
            return o.model_dump()
        if is_dataclass(o):
            return asdict(o)
        return str(o)
    return obj if isinstance(obj, str) else json.dumps(obj, default=default)


def _scrub(value):
    if not isinstance(value, (str, list)) or not trace_redaction():
        return value
    vault = current_vault()
    if vault is None:
        return WITHHELD if isinstance(value, str) else [WITHHELD]
    return vault.tokenize(value)


class _Span:
    """Thin wrapper so callers set inputs, outputs and attributes the same way traced or not."""

    def __init__(self, otel_span=None):
        self._span = otel_span

    def set(self, **attrs) -> None:
        if self._span is None:
            return
        for key, value in attrs.items():
            if value is None:
                continue
            is_str_list = isinstance(value, list) and all(isinstance(v, str) for v in value)
            if not isinstance(value, (str, bool, int, float)) and not is_str_list:
                value = _value(value)
            self._span.set_attribute(key.replace("__", "."), _scrub(value))

    def input(self, value) -> None:
        self.set(**{INPUT: _value(value), INPUT_MIME: JSON_MIME})

    def output(self, value) -> None:
        self.set(**{OUTPUT: _value(value), OUTPUT_MIME: JSON_MIME})


@contextmanager
def span(name: str, kind: str, **attrs):
    """Open a span of an OpenInference kind (AGENT, LLM, TOOL, GUARDRAIL, CHAIN)."""
    if _tracer is None:
        yield _Span()
        return
    with _tracer.start_as_current_span(name) as otel_span:
        s = _Span(otel_span)
        s.set(**{KIND: kind}, **attrs)
        yield s


def llm_messages(messages: list[dict]) -> list[dict]:
    """Neutral messages as plain JSON for the trace (provider-native `raw` left out)."""
    out = []
    for m in messages:
        if m["role"] == "assistant":
            r = m["reply"]
            out.append({"role": "assistant", "text": r.text,
                        "tool_calls": [{"name": c.name, "args": c.args} for c in r.tool_calls]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "results": [{"name": r.call.name, "result": r.result}
                                                    for r in m["results"]]})
        else:
            out.append(m)
    return out

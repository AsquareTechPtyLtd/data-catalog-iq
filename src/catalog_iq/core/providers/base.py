"""Provider interface: the only thing the agent loop knows about an LLM.

Each adapter (Claude and Gemini now, LiteLLM later) converts between its SDK's format
and these neutral types, so the loop and the guardrails never change when the
provider does.

Messages are a list of neutral dicts:
    {"role": "user", "text": "..."}
    {"role": "assistant", "reply": Reply}          # adapters may reuse reply.raw
    {"role": "tool", "results": [ToolResult, ...]}
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


class QuotaExhausted(RuntimeError):
    """The provider refuses calls for longer than is worth waiting (e.g. a daily quota)."""


@dataclass
class ToolCallRequest:
    id: str
    name: str
    args: Any  # usually a dict; kept as given so the tool guardrail can reject bad shapes


@dataclass
class ToolResult:
    call: ToolCallRequest
    result: dict


@dataclass
class LLMCall:
    """Performance of one LLM call. TPOT and TPS are derived from these."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    ttft_s: float | None   # time to first streamed chunk
    total_s: float
    cached_input_tokens: int = 0  # part of input_tokens read from the prompt cache

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)

    @property
    def tps(self) -> float | None:
        return self.output_tokens / self.total_s if self.total_s > 0 else None


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCallRequest]
    usage: LLMCall
    raw: Any = field(default=None, repr=False)  # provider-native message, for the adapter only


class Provider(Protocol):
    name: str
    model: str

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        """One LLM call. `tools` are neutral definitions: name, description, input_schema."""
        ...

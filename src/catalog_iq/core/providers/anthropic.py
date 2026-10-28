"""Claude adapter (anthropic SDK).

- Our neutral tool definitions (name, description, input_schema) are already
  Anthropic's format, so tools pass through unchanged.
- The SDK runs no tools itself: tool_use blocks come back to the loop and go
  through the tool guardrail.
- Responses are streamed so time to first token can be measured.
- Prompt caching (cache_control) reuses the system prompt, tool definitions
  and earlier turns across the calls of one question, which cuts input cost.
- Connection, rate-limit, overload and server errors are retried by
  providers/retry.py (honouring retry-after) within the run's time left; the
  SDK's own retries are off. A billing or credit error is raised as QuotaExhausted.
"""

import json
import time

import anthropic

from catalog_iq.core.providers.base import LLMCall, QuotaExhausted, Reply, ToolCallRequest
from catalog_iq.core.providers.retry import with_retries

MAX_TOKENS = 8192
RETRY_DELAYS_S = (1, 2, 4, 8)  # 4 retries, like the SDK default policy


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, client: anthropic.Anthropic | None = None,
                 max_tokens: int = MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic(  # api_key None → ANTHROPIC_API_KEY
            api_key=api_key, max_retries=0)

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        return with_retries(lambda timeout: self._stream(system, messages, tools, timeout), retry_wait,
                            RETRY_DELAYS_S)

    def _stream(self, system: str, messages: list[dict], tools: list[dict], timeout: float) -> Reply:
        start = time.perf_counter()
        ttft = None
        try:
            with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=system,
                                             messages=to_messages(messages), tools=tools,
                                             cache_control={"type": "ephemeral"}, timeout=timeout) as stream:
                for event in stream:
                    if ttft is None and event.type in ("content_block_start", "content_block_delta"):
                        ttft = time.perf_counter() - start
                final = stream.get_final_message()
        except anthropic.APIStatusError as e:
            if "credit balance" in str(e).lower() or e.status_code == 402:
                raise QuotaExhausted(f"{self.model}: {e}") from e
            raise
        total = time.perf_counter() - start

        blocks = [block_dict(b) for b in final.content]
        u = final.usage
        cached = (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        return Reply(
            text="".join(b["text"] for b in blocks if b["type"] == "text"),
            tool_calls=[ToolCallRequest(id=b["id"], name=b["name"], args=b["input"])
                        for b in blocks if b["type"] == "tool_use"],
            raw=blocks,
            usage=LLMCall(provider=self.name, model=self.model, input_tokens=u.input_tokens + cached,
                          output_tokens=u.output_tokens, ttft_s=ttft, total_s=total,
                          cached_input_tokens=u.cache_read_input_tokens or 0))


def retry_wait(e: Exception) -> float | None:
    """The SDK's retry policy: connection errors and timeouts, 408, 409, 429 and 5xx."""
    if isinstance(e, anthropic.APIConnectionError):
        return 0.0
    if isinstance(e, anthropic.APIStatusError) and (e.status_code in (408, 409, 429) or e.status_code >= 500):
        try:
            return float(e.response.headers.get("retry-after", 0))
        except (TypeError, ValueError):
            return 0.0
    return None


def block_dict(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump(exclude_none=True)


def to_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            reply = m["reply"]
            content = reply.raw if reply.raw is not None else (
                ([{"type": "text", "text": reply.text}] if reply.text else [])
                + [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in reply.tool_calls])
            out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r.call.id, "content": json.dumps(r.result),
                 **({"is_error": True} if r.result.get("status") == "error" else {})}
                for r in m["results"]]})
    return out

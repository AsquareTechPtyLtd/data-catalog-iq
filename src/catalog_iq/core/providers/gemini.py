"""Gemini adapter (google-genai SDK).

- Automatic function calling is off: the SDK must never run a tool itself,
  every call goes back to the loop and through the tool guardrail.
- Responses are streamed so time to first token can be measured.
- The model's own message (reply.raw) is sent back unchanged in later turns,
  because Gemini needs its thought signatures returned.
- Rate-limit and server errors are retried with backoff by providers/retry.py,
  within the run's time left. A 429 that asks to wait longer than MAX_WAIT_S
  (a daily quota) raises QuotaExhausted at once.
"""

import re
import time
import uuid

from google import genai
from google.genai import errors, types

from catalog_iq.core.providers.base import LLMCall, QuotaExhausted, Reply, ToolCallRequest
from catalog_iq.core.providers.retry import TIMEOUT_S, with_retries

RETRY_CODES = {429, 500, 503, 504}
RETRY_DELAYS_S = (5, 20, 60)
MAX_WAIT_S = 60  # a 429 asking for a longer wait is a quota, not a burst limit
RETRY_IN_RE = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, temperature: float = 0.0, api_key: str | None = None,
                 client: genai.Client | None = None, retry_delays_s: tuple[float, ...] = RETRY_DELAYS_S):
        self.model = model
        self.temperature = temperature
        self.client = client or genai.Client(  # api_key None → GEMINI_API_KEY
            api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))
        self.retry_delays_s = retry_delays_s

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[types.Tool(function_declarations=declarations(tools))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = to_contents(messages)

        def attempt(timeout: float) -> Reply:
            timed = config.model_copy(update={"http_options": types.HttpOptions(timeout=int(timeout * 1000))})
            return self._stream(contents, timed)

        def retry_wait(e: Exception) -> float | None:
            if not isinstance(e, errors.APIError) or e.code not in RETRY_CODES:
                return None
            wait = retry_after_s(str(e)) if e.code == 429 else None
            if wait is not None and wait > MAX_WAIT_S:
                raise QuotaExhausted(f"{self.model}: quota exhausted, retry in {wait / 3600:.1f}h") from e
            return wait or 0.0

        return with_retries(attempt, retry_wait, self.retry_delays_s)

    def _stream(self, contents: list[types.Content], config: types.GenerateContentConfig) -> Reply:
        start = time.perf_counter()
        ttft = None
        parts: list[types.Part] = []
        usage = None
        for chunk in self.client.models.generate_content_stream(model=self.model, contents=contents,
                                                                config=config):
            if ttft is None:
                ttft = time.perf_counter() - start
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                parts += chunk.candidates[0].content.parts
            usage = chunk.usage_metadata or usage
        total = time.perf_counter() - start

        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [ToolCallRequest(id=p.function_call.id or uuid.uuid4().hex[:8], name=p.function_call.name,
                                 args=p.function_call.args if p.function_call.args is not None else {})
                 for p in parts if p.function_call]
        return Reply(
            text=text, tool_calls=calls, raw=types.Content(role="model", parts=parts),
            usage=LLMCall(provider=self.name, model=self.model,
                          input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                          output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
                          if usage else 0,
                          ttft_s=ttft, total_s=total))


def declarations(tools: list[dict]) -> list[types.FunctionDeclaration]:
    return [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["input_schema"]) for t in tools]


def to_contents(messages: list[dict]) -> list[types.Content]:
    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["text"])]))
        elif m["role"] == "assistant":
            reply = m["reply"]
            if reply.raw is not None:
                contents.append(reply.raw)
            else:
                parts = [types.Part.from_text(text=reply.text)] if reply.text else []
                parts += [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args))
                          for c in reply.tool_calls]
                contents.append(types.Content(role="model", parts=parts))
        elif m["role"] == "tool":
            contents.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(id=r.call.id, name=r.call.name,
                                                                    response=r.result))
                for r in m["results"]]))
    return contents


def retry_after_s(message: str) -> float | None:
    """Seconds from Gemini's 'Please retry in 4h42m37.4s' hint, if present."""
    m = RETRY_IN_RE.search(message)
    if not m or not any(m.groups()):
        return None
    h, mins, sec = m.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(sec or 0)

"""A provider wrapper that keeps people's names and emails out of LLM calls.

Outgoing: system prompt, tool definitions and every message are tokenized by the Vault.
Incoming: reply text and tool-call args are restored, so the loop, tools and guardrails
only ever see real values. reply.raw is left as the provider sent it (tokens only), since
adapters replay it in later turns.

Without an explicit vault it uses the request's (redaction_scope) and refuses to call the
LLM when no scope is open, rather than send real values.
"""

from catalog_iq.core.providers.base import Provider, Reply, ToolCallRequest, ToolResult
from catalog_iq.core.redaction import Vault, current_vault


class RedactingProvider:
    def __init__(self, inner: Provider, vault: Vault | None = None):
        self.inner = inner
        self.vault = vault
        self.name = inner.name
        self.model = inner.model

    def _vault(self) -> Vault:
        v = self.vault or current_vault()
        if v is None:
            raise RuntimeError("redaction is on but no redaction_scope is open for this LLM call")
        return v

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        v = self._vault()
        reply = self.inner.chat(v.tokenize(system), [self._out(m) for m in messages], v.tokenize(tools))
        return Reply(text=v.restore(reply.text), usage=reply.usage, raw=reply.raw,
                     tool_calls=[ToolCallRequest(c.id, c.name, v.restore(c.args)) for c in reply.tool_calls])

    def _out(self, m: dict) -> dict:
        v = self._vault()
        if m["role"] == "assistant":
            r = m["reply"]
            return {**m, "reply": Reply(text=v.tokenize(r.text), usage=r.usage, raw=r.raw,
                                        tool_calls=[ToolCallRequest(c.id, c.name, v.tokenize(c.args))
                                                    for c in r.tool_calls])}
        if m["role"] == "tool":
            return {**m, "results": [ToolResult(ToolCallRequest(r.call.id, r.call.name, v.tokenize(r.call.args)),
                                                v.tokenize(r.result)) for r in m["results"]]}
        return v.tokenize(m)

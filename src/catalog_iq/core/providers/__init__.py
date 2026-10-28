"""LLM providers behind one interface (see base.py)."""

from catalog_iq.core.providers.base import LLMCall, Provider, Reply, ToolCallRequest, ToolResult


def make_provider(cfg: dict) -> Provider:
    """Build a provider from an agent's config (config/agents.yaml), behind redaction unless REDACTION=off."""
    from catalog_iq.core.providers.redacting import RedactingProvider
    from catalog_iq.core.redaction import llm_redaction
    provider = _make(cfg)
    return RedactingProvider(provider) if llm_redaction() else provider


def _make(cfg: dict) -> Provider:
    if cfg["provider"] == "anthropic":
        from catalog_iq.core.providers.anthropic import AnthropicProvider
        return AnthropicProvider(model=cfg["model"])
    if cfg["provider"] == "gemini":
        from catalog_iq.core.providers.gemini import GeminiProvider
        return GeminiProvider(model=cfg["model"], temperature=cfg.get("temperature", 0.0))
    raise ValueError(f"unknown provider '{cfg['provider']}'")


__all__ = ["LLMCall", "Provider", "Reply", "ToolCallRequest", "ToolResult", "make_provider"]

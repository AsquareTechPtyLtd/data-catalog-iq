"""Reversible redaction: people's names and email addresses become tokens before
anything leaves the trust boundary (LLM calls, traces), and are put back in what
the user sees.

    vault = Vault.from_catalog(conn)
    vault.tokenize("Ask Maya Okafor (maya.okafor@fernwood.example)")
        → "Ask PERSON_3f9a2c1b (EMAIL_8d04e7aa)"
    vault.restore(...)  → the original

One Vault per request. Tokens are an HMAC with a key made for the request: the same
value gets the same token within the request (the LLM can tell people apart) and an
unlinkable one in the next. The token → value map lives only in memory.

The request's Vault sits in a contextvar, opened by `redaction_scope` in each entry
point (route, ask, recommend_pi, recommend_data_category, /ask). A scope already open
is reused, so the router and the agents it picks share one set of tokens.

Two switches, both on unless set to off: REDACTION (LLM calls, providers/redacting.py)
and TRACE_REDACTION (span attributes, tracing.py). TRACE_REDACTION=off gives readable
traces for local debugging.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TOKEN = re.compile(r"\b(?:PERSON|EMAIL)_[0-9a-f]{8}\b")


def _norm(name: str) -> str:
    return " ".join(name.split()).lower()


class Vault:
    def __init__(self, names):
        self._key = secrets.token_bytes(32)
        self._values: dict[str, str] = {}
        self._people = {_norm(n): " ".join(n.split()) for n in names if n and n.strip()}
        # longest first so "Anna Maria Lopez" wins over "Anna Maria"
        alts = [r"\s+".join(map(re.escape, n.split())) for n in sorted(self._people, key=len, reverse=True)]
        self._names = re.compile(rf"(?<!\w)(?:{'|'.join(alts)})(?!\w)", re.IGNORECASE) if alts else None

    @classmethod
    def from_catalog(cls, conn: sqlite3.Connection) -> "Vault":
        return cls(r[0] for r in conn.execute("SELECT full_name FROM people"))

    def _token(self, kind: str, value: str) -> str:
        digest = hmac.new(self._key, f"{kind}:{value.lower()}".encode(), hashlib.sha256).hexdigest()[:8]
        token = f"{kind}_{digest}"
        self._values.setdefault(token, value)
        return token

    def _tokenize_str(self, text: str) -> str:
        text = EMAIL.sub(lambda m: self._token("EMAIL", m.group()), text)
        if self._names:
            text = self._names.sub(lambda m: self._token("PERSON", self._people[_norm(m.group())]), text)
        return text

    def tokenize(self, obj):
        """Strings, and the strings inside dicts, lists and tuples. Dict keys and other values are kept."""
        return self._walk(obj, self._tokenize_str)

    def restore(self, obj):
        """Tokens this vault issued become their values again; any other token-like text is left as is."""
        return self._walk(obj, lambda s: TOKEN.sub(lambda m: self._values.get(m.group(), m.group()), s))

    def _walk(self, obj, fn):
        if isinstance(obj, str):
            return fn(obj)
        if isinstance(obj, dict):
            return {k: self._walk(v, fn) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._walk(v, fn) for v in obj)
        return obj


_current: ContextVar[Vault | None] = ContextVar("redaction_vault", default=None)


def _on(var: str) -> bool:
    return os.environ.get(var, "on").strip().lower() not in ("off", "false", "0")


def llm_redaction() -> bool:
    return _on("REDACTION")


def trace_redaction() -> bool:
    return _on("TRACE_REDACTION")


def current_vault() -> Vault | None:
    return _current.get()


@contextmanager
def redaction_scope(conn: sqlite3.Connection):
    """Open a Vault for this request, or reuse the one already open. Does nothing when both switches are off."""
    if _current.get() is not None or not (llm_redaction() or trace_redaction()):
        yield _current.get()
        return
    token = _current.set(Vault.from_catalog(conn))
    try:
        yield _current.get()
    finally:
        _current.reset(token)

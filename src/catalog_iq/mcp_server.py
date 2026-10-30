"""The catalog tools as one MCP server, one layer below the agents.

    server = CatalogServer(conn)                    # our agents: real values
    reply = CatalogClient(server).call("get_table", {"table": "dim_customers"})

    catalog-iq mcp                                  # outside callers, over stdio: tokens

Every caller goes through the same checks here: the tool exists, its arguments
match its schema, and the result is trimmed to a size limit with its citations
kept whole. What only an agent knows stays in the agent's Guard: its own subset
of tools, its call budget per run and the checks on its answer.

Two kinds of caller:
- Our agents (in-process): real values, as today; their LLM calls are redacted
  at the provider boundary (core/providers/redacting.py).
- Outside callers (`outside=True`, e.g. Claude Desktop over stdio): their tool
  results go straight into their own LLM, so the server redacts. One Vault per
  session turns every person in `people` and every email into a token, in tool
  listings (filter_tables lists owners) and results; tokens in arguments are put
  back before the tool runs. There is no way back to names. Also a per-session
  rate limit and an entitlement hook (allow-all in the demo).

Eval-only options are start-up options, never tool arguments: `blind` (PI and
data category evidence without recorded answers) and the catalog connection
itself (edited catalogs for grounding evals).

Our agents call through an in-process MCP session (`CatalogClient`): the MCP
SDK dispatches straight to the handlers, on the caller's thread, with no
network and no JSON-RPC framing, about a millisecond a call. Results travel as
JSON text, so every caller gets exactly the same bytes.
"""

import copy
import json
import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import anyio
import mcp_types as types
from mcp import Client
from mcp.server import Server

from catalog_iq.core.redaction import Vault, vault_scope
from catalog_iq.core.tracing import span
from catalog_iq.ratelimit import RateLimiter
from catalog_iq.tools.columns import column_tools
from catalog_iq.tools.data_category_evidence import data_category_evidence_tools
from catalog_iq.tools.filter import FILTER_TOOLS
from catalog_iq.tools.lineage import LINEAGE_TOOLS
from catalog_iq.tools.tables import TOOLS as TABLE_TOOLS

NAME = "catalog-iq"
INSTRUCTIONS = ("Read-only tools over a data catalog (synthetic metadata for a fictional retailer): find a table, "
                "its owners, contract and data category, lineage, PI per column, data categories. Every fact in a "
                "result carries a citation id; cite it. People appear as PERSON_/EMAIL_ tokens.")
MAX_RESULT_CHARS = 20_000
OUTSIDE_PER_MINUTE, OUTSIDE_BURST = 60, 20  # tool calls per outside session
META = "catalog-iq"  # our key in a result's _meta: why a call was blocked, or that its result was trimmed
READ_ONLY = types.ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                  open_world_hint=False)

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict}

Entitled = Callable[[str, dict], str | None]  # (tool, arguments) → why the caller may not, or None


def catalog_tools(blind: bool = False) -> list[dict]:
    """All nine: Catalog Lookup's five, PI Classifier's one, Data Category Recommender's three."""
    return [*TABLE_TOOLS, *LINEAGE_TOOLS, *FILTER_TOOLS, *column_tools(blind), *data_category_evidence_tools(blind)]


def validate_args(args, schema: dict) -> list[str]:
    """Check tool arguments against the tool's input_schema (the subset our tools use)."""
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props = schema.get("properties", {})
    problems = [f"missing required argument '{k}'" for k in schema.get("required", []) if k not in args]
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            problems.append(f"unknown argument '{key}'")
            continue
        expected = _JSON_TYPES.get(spec.get("type"))
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool)):
            problems.append(f"'{key}' must be {spec['type']}")
            continue
        if "minimum" in spec and value < spec["minimum"]:
            problems.append(f"'{key}' must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            problems.append(f"'{key}' must be <= {spec['maximum']}")
        if spec.get("type") == "string" and not value.strip():
            problems.append(f"'{key}' must not be empty")
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"'{key}' must be one of {', '.join(spec['enum'])}")
    return problems


def trim(result: dict, max_chars: int) -> tuple[dict, int | None]:
    """The result within max_chars, and its size before when it had to be trimmed."""
    size = len(json.dumps(result))
    if size <= max_chars:
        return result, None
    result = copy.deepcopy(result)
    while len(json.dumps(result)) > max_chars:
        # halve the longest list; citations stay whole so cited ids remain checkable
        lists = [k for k, v in result.items() if isinstance(v, list) and v and k != "citations"]
        if not lists:
            break
        key = max(lists, key=lambda k: len(json.dumps(result[k])))
        result[key] = result[key][: len(result[key]) // 2]
    result["truncated"] = f"Result trimmed from {size} characters; some list items are omitted."
    return result, size


@dataclass
class Reply:
    """One tool call's result, and why it was blocked (tool_unknown, bad_args, rate_limited, not_entitled)."""

    result: dict
    blocked: str | None = None
    trimmed_from: int | None = None


class CatalogServer:
    def __init__(self, conn: sqlite3.Connection, *, blind: bool = False, tools: list[dict] = (),
                 outside: bool = False, max_result_chars: int = MAX_RESULT_CHARS, entitled: Entitled | None = None,
                 limiter: RateLimiter | None = None):
        """`tools` replace catalog tools of the same name (an agent's blind variants, or test doubles)."""
        self.conn = conn
        self.tools = {t["name"]: t for t in [*catalog_tools(blind), *tools]}
        self.outside = outside
        self.vault = Vault.from_catalog(conn) if outside else None
        self.limiter = limiter or (RateLimiter(OUTSIDE_PER_MINUTE, OUTSIDE_BURST) if outside else None)
        self.entitled = entitled
        self.max_result_chars = max_result_chars

    def definitions(self) -> list[dict]:
        defs = [{k: t[k] for k in ("name", "description", "input_schema")} for t in self.tools.values()]
        return self.vault.tokenize(defs) if self.outside else defs

    def call(self, name: str, args) -> Reply:
        if not self.outside:
            return self._call(name, args)
        with vault_scope(self.vault), span(f"tool:{name}", "TOOL", tool__name=name) as s:
            args = self.vault.restore(args)
            s.input(args)  # spans are tokenized by the open vault
            reply = self._call(name, args)
            reply.result = self.vault.tokenize(reply.result)
            s.output(reply.result)
            s.set(metadata={"blocked": reply.blocked, "caller": "outside"})
            return reply

    def _call(self, name: str, args) -> Reply:
        if self.limiter and (wait := self.limiter.take("session")):
            return _blocked("rate_limited", f"Too many tool calls; try again in {math.ceil(wait)} s.")
        tool = self.tools.get(name)
        if tool is None:
            return _blocked("tool_unknown", f"Unknown tool '{name}'. Available: {', '.join(self.tools)}.")
        problems = validate_args(args, tool["input_schema"])
        if problems:
            return _blocked("bad_args", "; ".join(problems))
        if self.entitled and (why := self.entitled(name, args)):
            return _blocked("not_entitled", why)
        result, size = trim(tool["function"](self.conn, **args), self.max_result_chars)
        return Reply(result, trimmed_from=size)


def _blocked(check: str, detail: str) -> Reply:
    return Reply({"status": "error", "error": detail}, blocked=check)


def mcp_server(catalog: CatalogServer) -> Server:
    """The MCP protocol side: tools/list and tools/call over a CatalogServer."""

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(name=d["name"], description=d["description"],
                                                       input_schema=d["input_schema"], annotations=READ_ONLY)
                                            for d in catalog.definitions()])

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        reply = catalog.call(params.name, params.arguments or {})
        meta = {k: v for k, v in (("blocked", reply.blocked), ("trimmed_from", reply.trimmed_from)) if v}
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(reply.result))],
                                    is_error=reply.blocked is not None, meta={META: meta} if meta else None)

    return Server(NAME, instructions=INSTRUCTIONS, on_list_tools=list_tools, on_call_tool=call_tool)


class CatalogClient:
    """Our agents' side: each call is an in-process MCP session on the caller's thread."""

    def __init__(self, catalog: CatalogServer):
        self.catalog = catalog
        self._server = mcp_server(catalog)

    def call(self, name: str, args: dict) -> Reply:
        result = anyio.run(self._call, name, args)
        meta = (result.meta or {}).get(META, {})
        return Reply(json.loads(result.content[0].text), meta.get("blocked"), meta.get("trimmed_from"))

    async def _call(self, name: str, args: dict) -> types.CallToolResult:
        async with Client(self._server) as client:
            return await client.call_tool(name, args)


def serve_stdio(conn: sqlite3.Connection) -> None:
    """`catalog-iq mcp`: the nine tools for one outside client over stdin/stdout, with tokens for names."""
    from mcp.server.stdio import stdio_server

    server = mcp_server(CatalogServer(conn, outside=True))

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)

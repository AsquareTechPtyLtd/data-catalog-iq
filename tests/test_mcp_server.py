"""Tests for the MCP tool layer: the nine tools, the checks every caller gets, redaction for
outside callers, and replay parity with saved eval runs (no LLM)."""

import json
import re
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.core.redaction import EMAIL
from catalog_iq.mcp_server import CatalogClient, CatalogServer, catalog_tools, mcp_server
from catalog_iq.ratelimit import RateLimiter
from catalog_iq.tools.data_categories import search_data_categories
from catalog_iq.tools.filter import OWNERS

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

EVALS = Path(__file__).parents[1] / "evals"
# runs saved since the tools last changed; eval runs are gitignored, so a fresh clone skips these
REPLAY_RUNS = ["catalog_lookup/runs/20261027-221915", "catalog_lookup/runs/20261027-223201",
               "catalog_lookup/runs/20261028-135537", "data_category_recommender/runs/20261027-215807",
               "pi_classifier/runs/20261027-220947", "pi_classifier/runs/20261027-221619"]
BLIND = {"pi_classifier", "data_category_recommender"}  # their evals run blind

CALLS = [  # one or more realistic calls per tool
    ("find_table", {"query": "loyalty"}), ("get_table", {"table": "dim_customers"}),
    ("get_lineage", {"table": "dim_customers", "direction": "upstream"}),
    ("trace_path", {"source": "raw_shop.customers", "target": "dim_customers"}),
    ("filter_tables", {"owner": "Maya Okafor"}), ("get_columns", {"table": "raw_returns.return_requests"}),
    ("get_data_category_evidence", {"table": "raw_web.page_views"}),
    ("search_data_categories", {"query": "website page views sessions"}),
    ("get_data_category", {"data_category_id": "DC-001"}),
]


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def call(server, name, args):
    """The raw MCP result, as any MCP client sees it."""
    async def go():
        async with Client(mcp_server(server)) as client:
            return await client.call_tool(name, args)
    return anyio.run(go)


def listing(server):
    async def go():
        async with Client(mcp_server(server)) as client:
            return (await client.list_tools()).tools
    return anyio.run(go)


def test_lists_the_nine_tools_read_only(conn):
    tools = listing(CatalogServer(conn))
    assert [t.name for t in tools] == [t["name"] for t in catalog_tools()] == [name for name, _ in CALLS]
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)


@pytest.mark.parametrize("name, args", CALLS)
def test_same_result_as_calling_the_tool(conn, name, args):
    tool = {t["name"]: t for t in catalog_tools()}[name]
    reply = CatalogClient(CatalogServer(conn)).call(name, args)
    assert reply.blocked is None and reply.result["status"] != "error"
    assert reply.result == json.loads(json.dumps(tool["function"](conn, **args)))


@pytest.mark.parametrize("run", REPLAY_RUNS)
def test_replays_saved_eval_runs(conn, run):
    """Every tool call a saved eval run made returns exactly what the agent saw then."""
    path = EVALS / run / "results.jsonl"
    if not path.exists():
        pytest.skip(f"{run} not saved here")
    client = CatalogClient(CatalogServer(conn, blind=run.split("/")[0] in BLIND))
    replayed = 0
    for row in map(json.loads, path.read_text().splitlines()):
        if "error" in row:
            continue
        calls = [e for e in row["response"]["evidence"] if e["result_status"] != "blocked"]
        for e, saved in zip(calls, row["tool_results"], strict=True):
            assert client.call(e["tool"], e["args"]).result == saved, (row["id"], e)
            replayed += 1
    assert replayed


@pytest.mark.parametrize("name, args, blocked, message", [
    ("drop_table", {"table": "dim_customers"}, "tool_unknown", "Unknown tool 'drop_table'"),
    ("get_table", {"table": 5}, "bad_args", "'table' must be string"),
    ("get_table", {}, "bad_args", "missing required argument 'table'"),
    ("filter_tables", {"owner": "Rahul Menon"}, "bad_args", "'owner' must be one of"),
])
def test_blocked_calls_are_tool_errors(conn, name, args, blocked, message):
    r = call(CatalogServer(conn), name, args)
    assert r.is_error and r.meta["catalog-iq"] == {"blocked": blocked}
    body = json.loads(r.content[0].text)
    assert body["status"] == "error" and body["error"].startswith(message)


def test_large_results_are_trimmed_with_citations_kept(conn):
    args = {"table": "dim_customers", "direction": "upstream"}
    full = CatalogClient(CatalogServer(conn)).call("get_lineage", args).result
    reply = CatalogClient(CatalogServer(conn, max_result_chars=3000)).call("get_lineage", args)
    assert reply.trimmed_from == len(json.dumps(full)) and "truncated" in reply.result
    assert reply.result["citations"] == full["citations"]


def test_blind_is_a_start_up_option(conn):
    args = {"table": "raw_returns.return_requests"}
    seen = CatalogClient(CatalogServer(conn)).call("get_columns", args).result
    blind = CatalogClient(CatalogServer(conn, blind=True)).call("get_columns", args).result
    assert seen != blind
    assert "blind" not in json.dumps([t.input_schema for t in listing(CatalogServer(conn))])


def test_search_one_data_category(conn):
    words = "website page views sessions"
    top = search_data_categories(conn, words)["data_categories"][0]["data_category_id"]
    one = search_data_categories(conn, words, data_category_id=top.lower())
    assert one["matched"] == 1 and [b["data_category_id"] for b in one["data_categories"]] == [top]
    other = next(t for t in (f"DC-{n:03d}" for n in range(1, 14))
                 if t not in {b["data_category_id"] for b in search_data_categories(conn, words, k=50)
                              ["data_categories"] if b["matched"]})
    assert search_data_categories(conn, words, data_category_id=other)["matched"] == 0


# --- outside callers -------------------------------------------------------------------

def people(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT full_name FROM people")]


def leaks(text: str, names: list[str]) -> list[str]:
    return [n for n in names if re.search(re.escape(n), text, re.IGNORECASE)] + EMAIL.findall(text)


def test_our_agents_get_real_names(conn):
    owners = json.dumps(CatalogClient(CatalogServer(conn)).call("get_table", {"table": "dim_customers"}).result)
    assert leaks(owners, people(conn))


def test_outside_callers_get_tokens_never_names(conn):
    server = CatalogServer(conn, outside=True)
    names = people(conn)
    assert not leaks(json.dumps([t.model_dump() for t in listing(server)]), names)
    for name, args in CALLS:
        r = call(server, name, args if name != "filter_tables" else {})
        assert not r.is_error and not leaks(r.content[0].text, names), name
    assert "PERSON_" in call(server, "get_table", {"table": "dim_customers"}).content[0].text


def test_outside_callers_can_use_the_tokens_they_were_given(conn):
    server = CatalogServer(conn, outside=True)
    owner = next(p for p in listing(server) if p.name == "filter_tables").input_schema["properties"]["owner"]
    assert not set(owner["enum"]) & set(OWNERS) and all(o.startswith("PERSON_") for o in owner["enum"])
    token = server.vault.tokenize("Maya Okafor")
    got = json.loads(call(server, "filter_tables", {"owner": token}).content[0].text)
    real = CatalogClient(CatalogServer(conn)).call("filter_tables", {"owner": "Maya Okafor"}).result
    assert got["status"] == real["status"] == "found" and len(got["tables"]) == len(real["tables"])


def test_tokens_differ_between_sessions(conn):
    first, second = CatalogServer(conn, outside=True), CatalogServer(conn, outside=True)
    assert first.vault.tokenize("Maya Okafor") != second.vault.tokenize("Maya Okafor")


def test_outside_callers_are_rate_limited(conn):
    server = CatalogServer(conn, outside=True, limiter=RateLimiter(per_minute=1, burst=2))
    assert [call(server, "get_table", {"table": "dim_customers"}).is_error for _ in range(3)] == [False, False, True]
    assert call(server, "get_table", {"table": "dim_customers"}).meta["catalog-iq"] == {"blocked": "rate_limited"}


def test_entitlement_hook(conn):
    def entitled(tool, args):
        return "not entitled to Confidential tables" if args.get("table") == "dim_customers" else None

    server = CatalogServer(conn, outside=True, entitled=entitled)
    assert call(server, "get_table", {"table": "dim_customers"}).meta["catalog-iq"] == {"blocked": "not_entitled"}
    assert not call(server, "get_table", {"table": "mart_web_sessions"}).is_error


def test_catalog_iq_mcp_over_stdio():
    """The real command, as Claude Desktop starts it: a subprocess speaking MCP on stdin/stdout."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "catalog_iq.cli", "mcp"],
                                   env={"PYTHONPATH": str(Path(__file__).parents[1] / "src")})

    async def go():
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            return tools, await client.call_tool("get_table", {"table": "dim_customers"})

    tools, result = anyio.run(go)
    assert len(tools) == 9 and not result.is_error
    body = json.loads(result.content[0].text)
    assert body["status"] == "found" and "PERSON_" in json.dumps(body["owners"])

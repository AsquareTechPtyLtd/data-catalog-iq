"""Catalog Lookup agent: governance facts, table-level lineage and cross-table filters.

    run = ask("Who owns dim_customers?")
    print(run.response.answer)

`ask` is the single entry point: the CLI, the eval runner and (later) the
API all call it, so they all get the same guardrails.
"""

import sqlite3
from pathlib import Path

from catalog_iq.catalog.db import connect, settings
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.loop import AgentRun, run_agent
from catalog_iq.core.providers import Provider, make_provider
from catalog_iq.core.redaction import redaction_scope
from catalog_iq.tools.filter import FILTER_TOOLS
from catalog_iq.tools.lineage import LINEAGE_TOOLS
from catalog_iq.tools.tables import TOOLS as TABLE_TOOLS

AGENT = "catalog_lookup"
PROMPT_PATH = Path(__file__).with_name("prompt.md")
TOOLS = TABLE_TOOLS + LINEAGE_TOOLS + FILTER_TOOLS  # find_table, get_table, get_lineage, trace_path, filter_tables


def system_prompt(conn: sqlite3.Connection) -> str:
    s = settings(conn)
    return PROMPT_PATH.read_text().format(team=s["escalation_team"], email=s["escalation_email"],
                                          slack=s["escalation_slack"])


def ask(question: str, conn: sqlite3.Connection | None = None, provider: Provider | None = None) -> AgentRun:
    own_conn = conn is None
    conn = conn or connect()
    try:
        if provider is None:
            load_env()
            provider = make_provider(agent_config(AGENT))
        with redaction_scope(conn):
            return run_agent(question, conn, provider, system_prompt(conn), AGENT, tools=TOOLS)
    finally:
        if own_conn:
            conn.close()

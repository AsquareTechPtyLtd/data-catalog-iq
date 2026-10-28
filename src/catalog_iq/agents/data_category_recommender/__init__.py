"""Data Category Recommender agent (A9): recommends one Data Category for a table that has none.

    run = recommend_data_category("raw_web.page_views")
    r = run.response
    print(r.status, r.recommendation.data_category_id if r.recommendation else None, r.confidence)

Recommendations are pending steward review; the catalog is never changed.
`blind` hides the table's and its lineage neighbours' data categories (evals only, N2);
the LLM cannot set it.
"""

import sqlite3
from pathlib import Path

from catalog_iq.agents.data_category_recommender.guard import DataCategoryGuard
from catalog_iq.catalog.db import connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.loop import AgentRun, run_agent
from catalog_iq.core.providers import Provider, make_provider
from catalog_iq.core.redaction import redaction_scope
from catalog_iq.tools.data_category_evidence import data_category_evidence_tools

AGENT = "data_category_recommender"
PROMPT_PATH = Path(__file__).with_name("prompt.md")
# get_data_category_evidence once, search_data_categories up to 3 times, a get_data_category or a corrected name
MAX_TOOL_CALLS = 5


def request(table: str) -> str:
    return f"Recommend a Data Category for {table}."


def recommend_data_category(table: str, conn: sqlite3.Connection | None = None, provider: Provider | None = None,
                            blind: bool = False) -> AgentRun:
    own_conn = conn is None
    conn = conn or connect()
    try:
        if provider is None:
            load_env()
            provider = make_provider(agent_config(AGENT))
        tools = data_category_evidence_tools(blind)
        question = request(table)
        guard = DataCategoryGuard(conn, agent=AGENT, question=question, tools=tools, max_tool_calls=MAX_TOOL_CALLS)
        with redaction_scope(conn):
            return run_agent(question, conn, provider, PROMPT_PATH.read_text(), AGENT, guard=guard, tools=tools)
    finally:
        if own_conn:
            conn.close()

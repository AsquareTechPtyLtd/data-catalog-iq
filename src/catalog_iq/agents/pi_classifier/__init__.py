"""PI Classifier agent (A8): recommends PI classifications for one table's open columns.

    run = recommend_pi("raw_returns.return_requests")
    for r in run.response.recommendations:
        print(r.column, r.recommendation, r.category, r.confidence)

Recommendations are pending steward review; the catalog is never changed.
`blind` hides recorded PI (evals only, D4); the LLM cannot set it.
"""

import sqlite3
from pathlib import Path

from catalog_iq.agents.pi_classifier.guard import PIGuard
from catalog_iq.catalog.db import connect
from catalog_iq.core.config import agent_config, load_env
from catalog_iq.core.loop import AgentRun, run_agent
from catalog_iq.core.providers import Provider, make_provider
from catalog_iq.tools.columns import column_tools

AGENT = "pi_classifier"
PROMPT_PATH = Path(__file__).with_name("prompt.md")
MAX_TOOL_CALLS = 3  # one get_columns call is the plan; room for a retry with a corrected name


def request(table: str) -> str:
    return f"Recommend PI classifications for the open columns of {table}."


def recommend_pi(table: str, conn: sqlite3.Connection | None = None, provider: Provider | None = None,
                 blind: bool = False) -> AgentRun:
    own_conn = conn is None
    conn = conn or connect()
    try:
        if provider is None:
            load_env()
            provider = make_provider(agent_config(AGENT))
        tools = column_tools(blind)
        question = request(table)
        guard = PIGuard(conn, agent=AGENT, question=question, tools=tools, max_tool_calls=MAX_TOOL_CALLS)
        return run_agent(question, conn, provider, PROMPT_PATH.read_text(), AGENT, guard=guard, tools=tools)
    finally:
        if own_conn:
            conn.close()

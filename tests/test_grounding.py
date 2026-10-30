"""The grounding eval's answer key and catalog edits (no LLM): every counterfactual edit must
change what the agent's own tools return, or the question tests nothing."""

import json

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.evals.answer_key import load_questions
from catalog_iq.evals.run import edited_catalog
from catalog_iq.evals.scoring import Vocabulary, score, summarize
from catalog_iq.tools.lineage import get_lineage
from catalog_iq.tools.tables import get_table
from test_scoring import make_run

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")

QUESTIONS = load_questions("grounding")
EDITED = [q for q in QUESTIONS if q.catalog_edits]


def test_answer_key_shape():
    assert {q.set for q in QUESTIONS} == {"dev", "heldout"}
    assert {s for q in QUESTIONS for s in q.slices} == {"false_premise", "planted", "counterfactual"}
    assert all(q.catalog_edits for q in QUESTIONS if {"planted", "counterfactual"} & set(q.slices))
    assert not any(q.catalog_edits for q in QUESTIONS if "false_premise" in q.slices)


def tool_view(conn, q) -> str:
    """What the agent would read for this question, as text."""
    table = q.expected.tables[0]
    result = get_lineage(conn, table, "upstream") if "feed" in q.question else get_table(conn, table)
    return json.dumps(result)


@pytest.mark.parametrize("q", EDITED, ids=[q.id for q in EDITED])
def test_edit_changes_what_the_tools_return(q, tmp_path):
    real = connect()
    edited = connect(edited_catalog(q.catalog_edits, tmp_path))
    before, after = tool_view(real, q), tool_view(edited, q)
    assert before != after
    for m in q.expected.forbidden_mentions:
        if "counterfactual" in q.slices:  # the replaced value is gone from the edited catalog
            assert m in before and m not in after, m
        else:  # the planted text is there for the agent to read
            assert m.split()[0] in after, m
    for m in q.expected.mentions:
        for option in (m if isinstance(m, list) else [m])[:1]:
            assert option in after or any(o in after for o in m if isinstance(m, list)), option
    real.close()
    edited.close()


def test_an_edit_that_changes_nothing_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="changed nothing"):
        edited_catalog(["UPDATE tables SET description = 'x' WHERE fqn = 'no.such.table'"], tmp_path)


def test_forbidden_mention_fails_and_high_but_wrong_is_counted():
    conn = connect()
    q = next(q for q in QUESTIONS if q.id == "G5")
    fqn = q.expected.tables[0]
    wrong = make_run(q, [get_table(conn, fqn)], "answered", "fct_orders is owned by Bob Smith.",
                     facts=[("Owner of fct_orders: Bob Smith", ["contract:data/sources/contracts/marts.fct_orders.yaml"])],
                     tables=[fqn])
    result = score(wrong, q, Vocabulary(conn))
    assert "forbidden_mention: Bob Smith" in result.correctness.reasons
    assert "missing_fact: Grace Whitfield" in result.correctness.reasons
    assert summarize([result])["derived"]["high_but_wrong"] == 1
    conn.close()

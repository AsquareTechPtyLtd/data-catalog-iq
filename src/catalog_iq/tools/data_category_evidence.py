"""Evidence tool for the Data Category Recommender: get_data_category_evidence.

A data category categorizes a whole table (data-model §4), so the evidence is about the
table: its description, domain and columns, the recorded PI of its columns
(context only: PI never decides the data category), the data categories of its lineage neighbours
and of other tables in the same domain, and the signals code derives from
them. Nothing here reads the warehouse.

Citation ids:
  catalog:<fqn>              a table's catalog entry: description, domain, data category
  catalog:<fqn>#<column>     a column's catalog entry: description, recorded PI
  collibra:DC-001           a data category (from the signals' keyword search)

Signals (data category agent N3) each name the data category(s) they suggest:
  neighbour      a lineage neighbour (any hops, up or down) has this data category (strong)
  definition     the table's words match this data category's name, definition or
                 glossary terms (search_data_categories), top DEFINITION_TOP ranks (weak:
                 keywords cannot tell what a table is about)
  data_category_domain     a data category named by the signals above belongs to the table's
                 governance domain (weak; one signal per data category)
  domain_tables  other tables in the same domain use this data category (weak)
The guard computes confidence from them, so the LLM's own certainty does not
count.

Blind mode (evals only, N2): the table's own data category and every lineage neighbour's
data category are hidden, so an agent can be measured on tables that already have one.
Same-domain tables stay visible (weaker evidence). The LLM cannot turn it on:
it is not in the tool's input schema.
"""

import re
import sqlite3
from collections import deque
from functools import partial

from catalog_iq.catalog.db import settings
from catalog_iq.tools.data_categories import DATA_CATEGORY_TOOLS, fts_query, search_data_categories
from catalog_iq.tools.tables import _Citations, _resolve

DEFINITION_TOP = 3


def _signal(kind: str, suggests: list[str], detail: str, sources: list[str], strong: bool = False) -> dict:
    return {"signal": kind, "suggests": suggests, "strong": strong, "detail": detail, "sources": sources}


def lineage_neighbours(conn: sqlite3.Connection, fqn: str) -> list[tuple[str, str, int]]:
    """(fqn, direction, hops) of every table reachable up or down the table lineage; exposures excluded."""
    out = []
    for direction, sql in (("upstream", "SELECT upstream_id FROM lineage_edges WHERE downstream_id = ?"),
                           ("downstream", "SELECT downstream_id FROM lineage_edges WHERE upstream_id = ? "
                                          "AND downstream_type = 'table'")):
        seen, queue = {fqn}, deque([(fqn, 0)])
        while queue:
            node, hops = queue.popleft()
            for (nxt,) in conn.execute(sql, (node,)):
                if nxt not in seen:
                    seen.add(nxt)
                    out.append((nxt, direction, hops + 1))
                    queue.append((nxt, hops + 1))
    return sorted(out, key=lambda n: (n[1], n[2], n[0]))


def table_text(name: str, description: str | None, columns: list[dict]) -> str:
    """The words that say what a table is about: its name, description and column names."""
    words = [name.replace("_", " "), description or ""]
    words += [re.sub(r"_(id|at|usd|ts)$", "", c["column"]).replace("_", " ") for c in columns]
    return " ".join(words)


def definition_signal(conn: sqlite3.Connection, evidence: dict, data_category_id: str) -> dict | None:
    """A definition signal for one chosen data category, if the table's words match it at any rank.

    The evidence's definition signals are the top DEFINITION_TOP of the whole
    catalog; in a catalog of thousands, distractors crowd a right data category out of
    those, so the guard checks the chosen one directly.
    """
    t = evidence["table"]
    q = fts_query(table_text(t["fqn"].split(".")[-1], t["description"], evidence["columns"]))
    if q is None or conn.execute("SELECT 1 FROM data_category_search WHERE data_category_search MATCH ? AND data_category_id = ?",
                                 (q, data_category_id)).fetchone() is None:
        return None
    return _signal("definition", [data_category_id], f"table name, description and column names match {data_category_id}'s "
                                           "definition (checked for the chosen data category)", [f"collibra:{data_category_id}"])


def signals(conn: sqlite3.Connection, t: sqlite3.Row, columns: list[dict], neighbours: list[dict],
            domain_data_categories: list[dict], cites: _Citations) -> list[dict]:
    """Each signal cites what it rests on, so a data category or table it names can be cited in an answer."""
    def data_category_cite(data_category_id):
        return cites.add("collibra", data_category_id, f"{settings(conn)['base_url']}/data-category/{data_category_id}")

    out = []
    for n in neighbours:
        if n.get("data_category_id"):
            out.append(_signal("neighbour", [n["data_category_id"]], f"{n['direction']} {n['fqn']} ({n['hops']} hop"
                                                           f"{'s' if n['hops'] > 1 else ''}) has {n['data_category_id']}",
                               [*n["sources"], data_category_cite(n["data_category_id"])], strong=True))
    text = table_text(t["name"], t["description"], columns)
    matches = [b for b in search_data_categories(conn, text)["data_categories"] if b["matched"]][:DEFINITION_TOP]
    for b in matches:
        out.append(_signal("definition", [b["data_category_id"]],
                           f"table name, description and column names match {b['data_category_id']} {b['name']} "
                           f"(keyword rank {b['rank']})", [data_category_cite(b["data_category_id"])]))
    # only domain data categories that other evidence names: a large catalog has hundreds per domain
    named = [b["data_category_id"] for b in matches] + [d["data_category_id"] for d in domain_data_categories]
    in_domain = [r["data_category_id"] for r in conn.execute(
        f"SELECT data_category_id FROM data_categories WHERE lower(replace(domain, ' ', '_')) = ? AND data_category_id IN "
        f"({','.join('?' * len(named))}) ORDER BY data_category_id", (t["domain"], *named))] if named else []
    for b in in_domain:  # one signal per data category, so citing a data category never vouches for another
        out.append(_signal("data_category_domain", [b], f"{b} belongs to the {t['domain']} governance domain", [data_category_cite(b)]))
    for d in domain_data_categories:
        out.append(_signal("domain_tables", [d["data_category_id"]],
                           f"{len(d['tables'])} other {t['domain']} table(s) use {d['data_category_id']}: "
                           f"{', '.join(d['tables'])}", [*d["sources"], data_category_cite(d["data_category_id"])]))
    return out


def get_data_category_evidence(conn: sqlite3.Connection, table: str, blind: bool = False) -> dict:
    t, error = _resolve(conn, table)
    if error:
        return error
    cites = _Citations(settings(conn)["base_url"])
    fqn = t["fqn"]
    head = {"fqn": fqn, "layer": t["layer"], "domain": t["domain"], "description": t["description"],
            "status": t["status"], "row_count": t["row_count"], "sources": [cites.add("catalog", fqn, t["url"])]}
    if t["layer"] in ("staging", "intermediate"):
        return {"status": "not_applicable", "table": head,
                "note": ("stg_/int_ tables have no data category of their own: a stg_ table inherits its source's, an int_ "
                         "table its built_for mart's (data-model §6). Recommend for that table instead."),
                "inherits_from": t["data_category_inherited_from"], "citations": cites.as_list(compact=True)}

    if t["data_category_id"] and not blind:
        b = conn.execute("SELECT data_category_id, name, status FROM data_categories WHERE data_category_id = ?", (t["data_category_id"],)).fetchone()
        return {"status": "assigned", "table": head,
                "data_category": {"data_category_id": b["data_category_id"], "name": b["name"], "status": b["status"],
                        "sources": [cites.add("collibra", fqn, t["url"])]},
                "note": "A data category is already assigned in Collibra; nothing to recommend.",
                "citations": cites.as_list(compact=True)}

    columns = []
    for c in conn.execute("SELECT * FROM columns WHERE table_fqn = ? ORDER BY ordinal", (fqn,)):
        col = {"column": c["name"], "data_type": c["data_type"], "description": c["description"],
               "sources": [cites.add("catalog", f"{fqn}#{c['name']}", c["url"])]}
        if c["pi_status"] in ("confirmed", "suggested"):
            col["pi"] = {"status": c["pi_status"], "category": c["pi_type"], "data_subject": c["data_subject"]}
        columns.append(col)

    neighbours, near = [], set()
    for nfqn, direction, hops in lineage_neighbours(conn, fqn):
        n = conn.execute("SELECT fqn, layer, description, data_category_id, url FROM tables WHERE fqn = ?", (nfqn,)).fetchone()
        near.add(nfqn)
        entry = {"fqn": nfqn, "direction": direction, "hops": hops, "layer": n["layer"],
                 "description": n["description"]}
        if not blind:
            entry["data_category_id"] = n["data_category_id"]
        entry["sources"] = [cites.add("catalog", nfqn, n["url"])]
        neighbours.append(entry)

    by_data_category: dict[str, dict] = {}
    for r in conn.execute("SELECT t.fqn, t.url, t.data_category_id, b.name FROM tables t JOIN data_categories b USING (data_category_id) "
                          "WHERE t.domain = ? AND t.layer IN ('source', 'mart') AND t.fqn != ? ORDER BY t.fqn",
                          (t["domain"], fqn)):
        if r["fqn"] in near:  # a neighbour's data category is shown (or hidden) with the neighbour
            continue
        d = by_data_category.setdefault(r["data_category_id"], {"data_category_id": r["data_category_id"], "name": r["name"], "tables": [], "sources": []})
        d["tables"].append(r["fqn"])
        d["sources"].append(cites.add("catalog", r["fqn"], r["url"]))
    domain_data_categories = list(by_data_category.values())

    return {"status": "found", "table": head, "columns": columns, "neighbours": neighbours,
            "same_domain_data_categories": domain_data_categories,
            "signals": signals(conn, t, columns, neighbours, domain_data_categories, cites),
            "citations": cites.as_list(compact=True)}


INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "table": {"type": "string", "description": "e.g. fernwood_dw.raw_web.page_views or page_views"},
    },
    "required": ["table"],
}
DESCRIPTION = (
    "Get the evidence for categorizing one table under a Data Category: its description, domain and "
    "columns (with recorded PI, as context only), its lineage neighbours and their data categories, the data categories other tables "
    "in its domain use, and signals code computed from these. Status `assigned` means it already has a data category; "
    "`not_applicable` means a stg_/int_ table, which inherits its data category. Accepts a fully qualified name, "
    "schema.name or an exact table name. Cite facts with the ids in their `sources`."
)


def data_category_evidence_tools(blind: bool = False) -> list[dict]:
    """The Data Category Recommender's tools: evidence (blind in evals) plus the shared data category tools."""
    return [{"name": "get_data_category_evidence", "description": DESCRIPTION, "input_schema": INPUT_SCHEMA,
             "function": partial(get_data_category_evidence, blind=blind)}, *DATA_CATEGORY_TOOLS]


DATA_CATEGORY_EVIDENCE_TOOLS = data_category_evidence_tools()

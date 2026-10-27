"""Filter tool: filter_tables, cross-table questions ("which tables have X?").

Starts from conditions instead of a named table. Typed parameters with enums,
combined with AND, run as fixed SQL; the LLM never writes SQL. Same citation
ids as tools.tables and tools.lineage, plus one for the query itself:

  catalog:filter?<sorted filters>      the result set: its count, or "none match"
  catalog:<fqn>                        catalog entry: domain, sensitivity, data category
  catalog:<fqn>#governance_gaps        matched gaps, unclassified candidates

`citations` lists only ids that carry a link (the query); every other id is
in the `sources` next to its fact, so a 23-table list stays small.

Each matched table has the lineage node shape (owner, certification,
loaded_from), plus contract status (with the owner: the contract file is the
owner's source), domain, sensitivity, data category, and the PI columns and gaps that
matched. Links are left out (code fills table links), as in tools.lineage. By default
only search-visible tables are searched (sources and marts, not in
development); include_internal adds staging, intermediate and in-development
tables. with_consumers adds the marts and exposures downstream of each table,
listed once in `nodes`, with the subgraph's `edges` so indirect links
(raw → stg → mart) can be checked.
"""

import json
import sqlite3
from urllib.parse import urlencode

from catalog_iq.tools.lineage import _Graph, _Nodes

DEFAULT_LIMIT = 25   # fits the largest realistic list (23 certified tables) under TARGET_CHARS
MAX_LIMIT = 25
MAX_CONSUMER_TABLES = 10
TARGET_CHARS = 18_000  # the tool guardrail trims at 20k; stay clear of it
LAYER_ORDER = {"mart": 0, "source": 1, "staging": 2, "intermediate": 3}

# Enum values for the tool schema; tests/test_filter.py checks they match catalog.db.
CERTIFICATION = ["certified", "uncertified", "not_applicable"]
LAYERS = ["source", "staging", "intermediate", "mart"]
STATUSES = ["active", "deprecated", "in_development"]
DOMAINS = ["customer", "digital", "finance", "marketing", "people", "supply_chain", "support"]
SENSITIVITIES = ["Internal", "Confidential", "Highly Confidential"]
OWNER_STATUSES = ["contract", "inherited", "none"]
OWNERS = ["Aisha Karim", "Carlos Mendes", "Grace Whitfield", "Hannah Lindqvist", "Kenji Mori", "Lucia Ferraro",
          "Maya Okafor", "Priya Nair", "Samuel Adeyemi", "Tom Brennan", "Wei Zhang"]
DATA_CATEGORIES = [f"DC-{n:03d}" for n in range(1, 14)]
GAP_TYPES = ["data_category_missing", "column_not_in_contract", "column_not_in_warehouse", "column_not_registered",
             "contract_column_missing", "contract_missing", "contract_not_active", "dbt_column_not_in_warehouse",
             "owner_missing", "pi_suggested", "pi_unclassified", "retention_not_defined",
             "table_not_in_warehouse"]
PI_TYPES = ["address", "date_of_birth", "email", "free_text", "government_id", "hashed_email", "ip_address",
                 "joining_date", "name", "payment_card", "personal_email", "phone", "salary", "termination_date"]
DATA_SUBJECTS = ["customer", "employee"]
PI_STATUSES = ["confirmed", "suggested", "unclassified"]

TABLE_FILTERS = {  # parameter → column on `tables`
    "certification": "certification", "layer": "layer", "status": "status", "domain": "domain",
    "sensitivity": "sensitivity", "has_contract": "has_contract", "owner_status": "owner_status",
    "data_category": "data_category_id",
}
COLUMN_FILTERS = {"pi_type": "pi_type", "data_subject": "data_subject", "pi_status": "pi_status"}
CONDITIONS = [*TABLE_FILTERS, "owner", "gap_type", *COLUMN_FILTERS]

# words in an unclassified column's name that suggest it may hold a PI type
PI_TYPE_WORDS = {"date_of_birth": ["birth", "dob"], "ip_address": ["ip"], "payment_card": ["card", "pan"],
                  "government_id": ["ssn", "passport", "national_id"], "hashed_email": ["email"],
                  "personal_email": ["email"], "free_text": ["text", "notes", "comment"]}


def _where(filters: dict, include_internal: bool) -> tuple[list[str], list]:
    where, params = [], []
    if not include_internal:
        where.append("t.search_visible = 1")
    for key, column in TABLE_FILTERS.items():
        if key in filters:
            where.append(f"t.{column} = ?")
            params.append(int(filters[key]) if key == "has_contract" else filters[key])
    if "owner" in filters:
        where.append("EXISTS (SELECT 1 FROM ownership o WHERE o.table_fqn = t.fqn AND o.role = 'Owner' "
                     "AND o.full_name = ?)")
        params.append(filters["owner"])
    if "gap_type" in filters:
        gap = "" if filters["gap_type"] == "any" else " AND g.gap_type = ?"
        where.append(f"EXISTS (SELECT 1 FROM governance_gaps g WHERE g.table_fqn = t.fqn{gap})")
        params += [] if gap == "" else [filters["gap_type"]]
    col_where, col_params = _column_where(filters)
    if col_where:
        where.append(f"EXISTS (SELECT 1 FROM columns c WHERE c.table_fqn = t.fqn AND {col_where})")
        params += col_params
    return where, params


def _column_where(filters: dict) -> tuple[str, list]:
    """All PI conditions hold on the same column."""
    keys = [k for k in COLUMN_FILTERS if k in filters]
    return " AND ".join(f"c.{COLUMN_FILTERS[k]} = ?" for k in keys), [filters[k] for k in keys]


def _query_id(filters: dict, include_internal: bool) -> str:
    items = sorted((k, str(v).lower() if isinstance(v, bool) else v) for k, v in filters.items())
    if include_internal:
        items.append(("scope", "all"))
    return urlencode(items)


def _describe(filters: dict) -> str:
    return ", ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in sorted(filters.items()))


class _FilterNodes(_Nodes):
    """Lineage nodes plus the fields a filter answer needs."""

    def matched_table(self, fqn: str, filters: dict) -> dict:
        t = self.conn.execute("SELECT * FROM tables WHERE fqn = ?", (fqn,)).fetchone()
        node = self.node(fqn)
        # the contract file is the owner's source, so its status sits with the owner
        node["owner"]["contract"] = (t["contract_status"] if t["has_contract"] else
                                     "not_applicable" if t["layer"] in ("staging", "intermediate") else "missing")
        node["catalog"] = {"domain": t["domain"], "sensitivity": t["sensitivity"],
                           "data_category": t["data_category_id"],
                           "sources": [self.cites.add("catalog", fqn)]}
        col_where, col_params = _column_where(filters)
        if col_where:
            node["matched_columns"] = [self._column(fqn, c) for c in self.conn.execute(
                f"SELECT * FROM columns c WHERE c.table_fqn = ? AND {col_where} ORDER BY c.ordinal",
                (fqn, *col_params))]
        if "gap_type" in filters:
            gap = "" if filters["gap_type"] == "any" else " AND gap_type = ?"
            cite = self.cites.add("catalog", f"{fqn}#governance_gaps")
            node["matched_gaps"] = [
                {"type": g["gap_type"], "severity": g["severity"], "column": g["column_name"], "detail": g["detail"],
                 "sources": [cite]}
                for g in self.conn.execute(
                    f"SELECT g.* FROM governance_gaps g LEFT JOIN columns c ON c.table_fqn = g.table_fqn "
                    f"AND c.name = g.column_name WHERE g.table_fqn = ?{gap} ORDER BY c.ordinal, g.gap_type",
                    (fqn, *([] if gap == "" else [filters["gap_type"]])))]
        return node

    def _column(self, fqn: str, c: sqlite3.Row) -> dict:
        src = (self.cites.add("collibra", f"{fqn}.{c['name']}") if c["classification_source"] == "collibra"
               else self.cites.add("catalog", f"{fqn}#column_lineage"))
        return {"column": c["name"], "status": c["pi_status"], "pi_type": c["pi_type"],
                "data_subject": c["data_subject"], "sensitivity": c["sensitivity"], "sources": [src]}


def _candidates(conn, nodes: _FilterNodes, filters: dict, include_internal: bool, matched: set[str]) -> list[dict]:
    """Unclassified columns that could match the PI type, on tables meeting the other conditions."""
    pi_type = filters.get("pi_type")
    if not pi_type or filters.get("pi_status", "confirmed") != "confirmed":
        return []
    words = PI_TYPE_WORDS.get(pi_type, [w for w in pi_type.split("_") if len(w) > 2])
    others = {k: v for k, v in filters.items() if k not in COLUMN_FILTERS}
    where, params = _where(others, include_internal)
    out = []
    for c in conn.execute(
            "SELECT c.table_fqn, c.name, c.description FROM columns c JOIN tables t ON t.fqn = c.table_fqn "
            f"WHERE c.pi_status = 'unclassified' {''.join(' AND ' + w for w in where)} "
            "ORDER BY c.table_fqn, c.ordinal", params):
        if c["table_fqn"] in matched or not any(w in c["name"].lower() for w in words):
            continue
        out.append({"fqn": c["table_fqn"], "column": c["name"], "description": c["description"],
                    "status": "unclassified",
                    "note": f"Not yet classified; could hold {pi_type} data. Awaiting review.",
                    "sources": [nodes.cites.add("catalog", f"{c['table_fqn']}#governance_gaps")]})
    return out


def _consumers(tables: list[dict], graph: _Graph, nodes: _FilterNodes) -> dict:
    """Marts and exposures downstream of each table, each listed once, plus the subgraph's edges."""
    matched = {t["fqn"] for t in tables}
    per_table, shared, edges = {}, {}, set()
    for t in tables:
        depths, _ = graph.walk(t["fqn"], "downstream", 10)
        walked = {t["fqn"], *depths}
        edges.update((u, d) for d in walked for u in graph.up[d] if u in walked)
        named = [n for n, k in sorted(depths.items(), key=lambda x: (x[1], x[0]))
                 if n.startswith("exposure.") or nodes.conn.execute(
                     "SELECT layer FROM tables WHERE fqn = ?", (n,)).fetchone()["layer"] == "mart"]
        for n in named:
            if n not in matched and n not in shared:
                shared[n] = nodes.node(n)
        per_table[t["fqn"]] = {"marts_and_exposures": named, "staging_intermediate_count": len(depths) - len(named),
                               "sources": t["sources"]}
    return {"per_table": per_table, "nodes": sorted(shared.values(), key=lambda n: n.get("fqn") or n["id"]),
            "edges": sorted([u, d] for u, d in edges)}


def filter_tables(conn: sqlite3.Connection, include_internal: bool = False, with_consumers: bool = False,
                  limit: int = DEFAULT_LIMIT, **filters) -> dict:
    """Tables matching every given condition (AND)."""
    filters = {k: v for k, v in filters.items() if k in CONDITIONS}
    if not filters:
        return {"status": "error", "error": "Give at least one condition, e.g. certification or pi_type "
                                            "(gap_type=\"any\" for tables with any governance gap)."}
    if filters.get("pi_type") or filters.get("data_subject"):
        filters.setdefault("pi_status", "confirmed")
    include_internal = (include_internal or filters.get("layer") in ("staging", "intermediate")
                        or filters.get("status") == "in_development")

    where, params = _where(filters, include_internal)
    rows = conn.execute(f"SELECT t.fqn, t.layer FROM tables t WHERE {' AND '.join(where) or '1 = 1'}",
                        params).fetchall()
    fqns = [r["fqn"] for r in sorted(rows, key=lambda r: (LAYER_ORDER[r["layer"]], r["fqn"]))]

    graph = _Graph(conn)
    nodes = _FilterNodes(conn, graph)
    shown = fqns[:limit]
    qid = _query_id(filters, include_internal)
    scope = ("all tables" if include_internal
             else "sources and marts (staging, intermediate and in-development tables excluded)")
    text = (f"{len(fqns)} tables match {_describe(filters)} in {scope}." if fqns
            else f"No tables match {_describe(filters)} in {scope}.")
    result = {
        "status": "found" if fqns else "no_match",
        "query": {"filters": filters, "scope": "all" if include_internal else "search_visible",
                  "count": len(fqns), "returned": len(shown), "matched": shown, "text": text,
                  "sources": [nodes.cites.add("catalog", f"filter?{qid}", f"{nodes.cfg['base_url']}/search?{qid}")]},
        "tables": [nodes.matched_table(fqn, filters) for fqn in shown],
    }
    if len(shown) < len(fqns):
        result["truncated"] = True
    candidates = _candidates(conn, nodes, filters, include_internal, set(fqns))
    if candidates:
        result["unclassified_candidates"] = candidates
    if with_consumers:
        if len(fqns) > MAX_CONSUMER_TABLES:
            result["consumers_note"] = (f"with_consumers needs {MAX_CONSUMER_TABLES} or fewer matching tables "
                                        f"({len(fqns)} matched). Narrow the filter to see consumers.")
        else:
            before = dict(nodes.cites.items)
            lineage = _consumers(result["tables"], graph, nodes)
            size = len(json.dumps(result)) + len(json.dumps(lineage))
            if size <= TARGET_CHARS:
                for t in result["tables"]:
                    t["consumers"] = lineage["per_table"][t["fqn"]]
                result["nodes"], result["edges"] = lineage["nodes"], lineage["edges"]
            else:  # D5/D9: too big to list safely; the LLM can walk the tables it needs
                nodes.cites.items = before
                result["consumers_note"] = ("Consumers of these tables are too many to list here. Use get_lineage "
                                            "on the tables you need, or narrow the filter.")
    # every id is already inline in `sources`; listing only those with a link keeps big lists small
    result["citations"] = [c for c in nodes.cites.as_list(compact=True) if "url" in c]
    return result


# --- Tool definition (provider-neutral) ------------------------------------------


def _enum(values: list[str], description: str) -> dict:
    return {"type": "string", "enum": values, "description": description}


FILTER_TOOLS = [
    {
        "name": "filter_tables",
        "description": (
            "Find every table matching conditions, for questions that do not name a table "
            "('which certified tables contain customer email?', 'what does Maya Okafor own?'). "
            "All conditions must hold (AND); give at least one. PI conditions (pi_type, "
            "data_subject, pi_status) must hold on the same column: customer email = "
            "pi_type email + data_subject customer; hashed_email and personal_email are separate "
            "PI types; only confirmed PI by default. owner = the Owner role. Searches sources and "
            "marts unless include_internal. Each table has owner, certification, contract, data category and "
            "the columns/gaps that matched; with_consumers adds downstream marts and exposures "
            "(10 or fewer tables). `query` holds the count and a citation for the result set, "
            "including 'no tables match'. `unclassified_candidates` are columns awaiting review that "
            "could match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "certification": _enum(CERTIFICATION, "not_applicable = staging/intermediate"),
                "layer": _enum(LAYERS, "staging/intermediate imply include_internal"),
                "status": _enum(STATUSES, "in_development implies include_internal"),
                "domain": _enum(DOMAINS, "Business domain"),
                "sensitivity": _enum(SENSITIVITIES, "Table sensitivity"),
                "has_contract": {"type": "boolean", "description": "Has its own data contract"},
                "owner_status": _enum(OWNER_STATUSES, "none = no owner on record"),
                "owner": _enum(OWNERS, "Person in the Owner role"),
                "data_category": _enum(DATA_CATEGORIES, "Data Category the table is assigned (or inherits)"),
                "gap_type": _enum(["any", *GAP_TYPES], "Governance gap; any = tables with any gap"),
                "pi_type": _enum(PI_TYPES, "PI type of a column"),
                "data_subject": _enum(DATA_SUBJECTS, "Whose data the PI column holds"),
                "pi_status": _enum(PI_STATUSES, "Default confirmed; suggested = awaiting review"),
                "include_internal": {"type": "boolean", "default": False,
                                     "description": "Also search staging, intermediate and in-development tables"},
                "with_consumers": {"type": "boolean", "default": False,
                                   "description": "Add downstream marts and exposures of each table"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": DEFAULT_LIMIT},
            },
            "required": [],
        },
        "function": filter_tables,
    },
]

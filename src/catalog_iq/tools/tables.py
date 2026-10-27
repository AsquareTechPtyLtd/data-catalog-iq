"""Table lookup tools: find_table and get_table.

Plain functions over the read-only catalog. They never call an LLM. Every fact
they return is tagged with citation ids, and each result carries a
`citations` list describing those ids, so answers can be checked later.

Citation ids are "<source>:<ref>":
  contract:data/sources/contracts/marts.dim_customers.yaml   data contract file
  collibra:fernwood_dw.marts.dim_customers                   Collibra asset (table)
  collibra:DC-001                                           Collibra data category
  collibra:group/Data Strategy                               Collibra user group
  dbt:model.fernwood.dim_customers                           dbt manifest node
  dbt:exposure.fernwood.loyalty_dashboard                    dbt exposure (dashboard, ML model, app)
  catalog:<fqn>#certification                                rule applied at ingestion
  catalog:<fqn>#data_category_inheritance                              rule applied at ingestion (stg_/int_ data category, or none)
  catalog:<fqn>#ownership                                    rule applied at ingestion (no contract / no owner)
  catalog:<fqn>#governance_gaps                              gap detected at ingestion
  catalog:<fqn>                                              catalog entry (search match)
"""

import json
import re
import sqlite3

from catalog_iq.catalog.db import settings

MIN_SCORE = 0.3
FOUND_MARGIN = 0.15
MART_MIN_SCORE = 0.7  # a mart scoring at least this outranks source/staging tables
# words people add around a table name ("the loyalty table") that match nothing and dilute scores
FILLER_WORDS = {"a", "an", "the", "table", "tables", "dataset", "datasets", "model", "models"}


def _norm(text: str) -> str:
    # spaces/hyphens become "_"; existing underscores are kept (stg_x__y has a double one)
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _strip_filler(query: str) -> str:
    words = [w for w in query.split() if w.lower() not in FILLER_WORDS]
    return " ".join(words) or query  # a query of only filler words is kept as typed


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


class _Citations:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.items: dict[str, dict] = {}

    def add(self, source: str, ref: str, url: str | None = None) -> str:
        cid = f"{source}:{ref}"
        self.items.setdefault(cid, {"id": cid, "source": source, "ref": ref, "url": url})
        return cid

    def as_list(self, compact: bool = False) -> list[dict]:
        """compact: id (+ url when there is one) only; source and ref are parts of the id."""
        if compact:
            return [{"id": c["id"], **({"url": c["url"]} if c["url"] else {})} for c in self.items.values()]
        return list(self.items.values())


# --- find_table -------------------------------------------------------------------


def _score(query: str, row: sqlite3.Row) -> float:
    q = _norm(query)
    short = f"{row['schema_name']}.{row['name']}"
    if q in (row["fqn"], short, row["name"]):
        return 1.0
    if len(q) >= 4 and q in row["name"]:
        return 0.8
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    name_overlap = len(q_tokens & _tokens(short)) / len(q_tokens)
    desc_overlap = len(q_tokens & _tokens(row["description"] or "")) / len(q_tokens)
    return max(0.7 * name_overlap, 0.3 * desc_overlap)


def _score_exposure(query: str, row: sqlite3.Row) -> float:
    q = _norm(query)
    if q in (row["exposure_id"], row["name"], _norm(row["label"])):
        return 1.0
    if len(q) >= 4 and q in row["name"]:
        return 0.8
    q_tokens = _tokens(query)
    if not q_tokens:
        return 0.0
    name_overlap = len(q_tokens & (_tokens(row["name"]) | _tokens(row["label"]))) / len(q_tokens)
    desc_overlap = len(q_tokens & _tokens(row["description"] or "")) / len(q_tokens)
    return max(0.7 * name_overlap, 0.3 * desc_overlap)


def _find_exposure(conn: sqlite3.Connection, query: str, limit: int) -> dict:
    terms = _strip_filler(query)
    cites = _Citations(settings(conn)["base_url"])
    matches = []
    for row in conn.execute("SELECT exposure_id, name, label, type, description, owner_name, url FROM exposures"):
        score = _score_exposure(terms, row)
        if score >= MIN_SCORE:
            matches.append({"id": row["exposure_id"], "name": row["name"], "label": row["label"],
                            "exposure_type": row["type"], "description": row["description"],
                            "owner": row["owner_name"], "score": round(score, 2), "url": row["url"]})
    matches.sort(key=lambda m: (-m["score"], m["id"]))
    matches = matches[:limit]
    for m in matches:
        m["sources"] = [cites.add("dbt", m["id"], m.pop("url"))]
    if not matches:
        status = "not_found"
    elif len(matches) == 1 or matches[0]["score"] - matches[1]["score"] >= FOUND_MARGIN:
        status = "found"
    else:
        status = "ambiguous"
    return {"query": query, "kind": "exposure", "status": status, "matches": matches,
            "citations": cites.as_list()}


def find_table(conn: sqlite3.Connection, query: str, limit: int = 5, kind: str = "table") -> dict:
    """Find tables (or, with kind="exposure", dashboards/models/apps) matching a name or description.

    stg_/int_ tables are hidden from search; they are returned only on an exact
    name match, flagged search_visible=false.

    Marts are preferred: a mart scoring at least MART_MIN_SCORE ranks above
    every non-mart, so "customers" finds dim_customers before
    raw_storefront.customers. Several such marts close in score are ambiguous.
    A schema-qualified query ("raw_storefront.customers") names one table, so
    plain score order applies. Filler words ("the", "table") are ignored.
    """
    if kind == "exposure":
        return _find_exposure(conn, query, limit)
    qualified = "." in query
    terms = _strip_filler(query)
    matches = []
    for row in conn.execute("SELECT fqn, schema_name, name, layer, description, status, "
                            "certification, search_visible, url FROM tables"):
        score = _score(terms, row)
        if score < MIN_SCORE or (not row["search_visible"] and score < 1.0):
            continue
        matches.append({"fqn": row["fqn"], "name": row["name"], "layer": row["layer"],
                        "description": row["description"], "status": row["status"],
                        "certification": row["certification"],
                        "search_visible": bool(row["search_visible"]), "score": round(score, 2),
                        "url": row["url"]})

    def preferred(m: dict) -> bool:
        return not qualified and m["layer"] == "mart" and m["score"] >= MART_MIN_SCORE

    matches.sort(key=lambda m: (not preferred(m), -m["score"], m["fqn"]))
    matches = matches[:limit]
    cites = _Citations(settings(conn)["base_url"])
    for m in matches:
        m["sources"] = [cites.add("catalog", m["fqn"], m.pop("url"))]

    if not matches:
        status = "not_found"
    elif (len(matches) == 1 or preferred(matches[0]) != preferred(matches[1])
          or matches[0]["score"] - matches[1]["score"] >= FOUND_MARGIN):
        status = "found"
    else:
        status = "ambiguous"
    return {"query": query, "kind": "table", "status": status, "matches": matches, "citations": cites.as_list()}


# --- get_table --------------------------------------------------------------------


def _resolve(conn: sqlite3.Connection, table: str) -> tuple[sqlite3.Row | None, dict | None]:
    """Accept an FQN, schema.name or bare name; only exact matches resolve."""
    q = _norm(table)
    rows = conn.execute(
        "SELECT * FROM tables WHERE fqn = ? OR schema_name || '.' || name = ? OR name = ?",
        (q, q, q)).fetchall()
    if len(rows) == 1:
        return rows[0], None
    if rows:
        return None, {"status": "ambiguous", "table": table, "candidates": [r["fqn"] for r in rows]}
    return None, {"status": "not_found", "table": table,
                  "hint": "No exact match. Use find_table to search by name or description."}


def _people(conn: sqlite3.Connection, fqn: str) -> list[dict]:
    return [{"name": r["full_name"], "email": r["email"], "role": r["role"]}
            for r in conn.execute("SELECT full_name, email, role FROM ownership WHERE table_fqn = ? "
                                  "ORDER BY role = 'Owner' DESC, full_name", (fqn,))]


def _support(conn: sqlite3.Connection, fqn: str) -> list[dict]:
    return [{"channel": r["channel"], "tool": r["tool"], "scope": r["scope"]}
            for r in conn.execute("SELECT channel, tool, scope FROM support_channels WHERE table_fqn = ?", (fqn,))]


def _contract_cite(conn: sqlite3.Connection, cites: _Citations, fqn: str) -> str | None:
    row = conn.execute("SELECT ref FROM provenance WHERE entity_type = 'table' AND entity_id = ? "
                       "AND source = 'contract'", (fqn,)).fetchone()
    return cites.add("contract", row["ref"]) if row else None


def _escalation(cfg: dict, cites: _Citations) -> dict:
    return {"team": cfg["escalation_team"], "email": cfg["escalation_email"], "slack": cfg["escalation_slack"],
            "sources": [cites.add("collibra", f"group/{cfg['escalation_team']}")]}


def get_table(conn: sqlite3.Connection, table: str) -> dict:
    """Governance facts for one table: contract, owners, certification, data category, PI, gaps."""
    t, error = _resolve(conn, table)
    if error:
        return error
    cfg = settings(conn)
    cites = _Citations(cfg["base_url"])
    fqn = t["fqn"]

    table_src = []
    for p in conn.execute("SELECT source, ref FROM provenance WHERE entity_type = 'table' AND entity_id = ?", (fqn,)):
        if p["source"] == "collibra":
            table_src.append(cites.add("collibra", p["ref"], t["url"]))
        elif p["source"] == "dbt":
            table_src.append(cites.add("dbt", p["ref"]))
    escalation = _escalation(cfg, cites)

    # contract (own contract only; stg_/int_ never have one)
    contract_cite = _contract_cite(conn, cites, fqn)
    if contract_cite:
        contract = {"status": t["contract_status"], "id": t["contract_id"], "version": t["contract_version"],
                    "data_product": t["data_product"],
                    "slas": [dict(r) for r in conn.execute(
                        "SELECT property, value, unit FROM slas WHERE table_fqn = ?", (fqn,))],
                    "sources": [contract_cite]}
    elif t["layer"] in ("staging", "intermediate"):
        contract = {"status": "not_applicable",
                    "note": "stg_/int_ tables have no contract of their own; owners are inherited."}
    else:
        contract = {"status": "missing", "escalation": escalation,
                    "sources": [cites.add("catalog", f"{fqn}#ownership")]}

    # owners and support
    owners = {"status": t["owner_status"], "people": _people(conn, fqn), "support": _support(conn, fqn)}
    if t["owner_status"] == "contract":
        owners["sources"] = [contract_cite]
    elif t["owner_status"] == "inherited":
        owners["inherited_from"] = t["owner_inherited_from"]
        owners["sources"] = [_contract_cite(conn, cites, t["owner_inherited_from"])]
    else:
        if t["owner_inherited_from"]:
            owners["inherited_from"] = t["owner_inherited_from"]
            owners["note"] = f"Owner would be inherited from {t['owner_inherited_from']}, which has no data contract."
        else:
            owners["note"] = "No data contract, so no owner or support team is on record."
        owners["escalation"] = escalation
        owners["sources"] = [cites.add("catalog", f"{fqn}#ownership")]
    if t["contract_status"] == "deprecated":
        owners["note"] = "Contract is deprecated; owners shown are from the deprecated contract."

    certification = {"status": t["certification"],
                     "failed_criteria": json.loads(t["cert_failed_criteria"]),
                     "details": json.loads(t["cert_details"]),
                     "sources": [cites.add("catalog", f"{fqn}#certification")]}

    # Data category (table level): assigned in Collibra, or inherited by stg_/int_ (§4, §6); retention and stewards follow it
    if t["data_category_source"] == "collibra":
        data_category_src = [cites.add("collibra", fqn, t["url"])]
    else:
        data_category_src = [cites.add("catalog", f"{fqn}#data_category_inheritance")]
    if t["data_category_id"]:
        b = conn.execute("SELECT * FROM data_categories WHERE data_category_id = ?", (t["data_category_id"],)).fetchone()
        stewards: dict[str, list] = {}
        for r in conn.execute("SELECT s.role, p.full_name, p.email FROM data_category_stewardship s "
                              "JOIN people p USING (username) WHERE s.data_category_id = ? ORDER BY s.role, p.full_name",
                              (b["data_category_id"],)):
            stewards.setdefault(r["role"], []).append({"name": r["full_name"], "email": r["email"]})
        retention = ({"months": b["retention_months"], "trigger": b["retention_trigger"],
                      "disposal_action": b["disposal_action"], "legal_basis": b["legal_basis"]}
                     if b["retention_months"] is not None else "not defined")
        data_category = {"data_category_id": b["data_category_id"], "name": b["name"], "status": b["status"], "retention": retention,
               "stewards": stewards, "assigned": t["data_category_source"],
               **({"inherited_from": t["data_category_inherited_from"]} if t["data_category_source"] == "inherited" else {}),
               "sources": [cites.add("collibra", b["data_category_id"], b["url"]), *data_category_src]}
    else:
        note = (f"Inherits its data category from {t['data_category_inherited_from']}, which has none." if t["data_category_inherited_from"]
                else "No data category is assigned to this table in Collibra.")
        data_category = {"data_category_id": None, "note": note,
               **({"inherited_from": t["data_category_inherited_from"]} if t["data_category_inherited_from"] else {}),
               "sources": data_category_src}

    pi_columns = []
    for c in conn.execute("SELECT name, pi_status, pi_type, data_subject, sensitivity, detection_confidence, "
                          "classification_source, url FROM columns WHERE table_fqn = ? "
                          "AND pi_status IN ('confirmed', 'suggested') ORDER BY ordinal", (fqn,)):
        src = (cites.add("collibra", f"{fqn}.{c['name']}", c["url"]) if c["classification_source"] == "collibra"
               else cites.add("catalog", f"{fqn}#column_lineage"))
        pi_columns.append({"column": c["name"], "status": c["pi_status"], "category": c["pi_type"],
                           "data_subject": c["data_subject"], "sensitivity": c["sensitivity"],
                           **({"confidence": c["detection_confidence"]} if c["pi_status"] == "suggested" else {}),
                           "classification": c["classification_source"], "sources": [src]})

    gap_cite = cites.add("catalog", f"{fqn}#governance_gaps")
    gaps = [{"column": g["column_name"], "type": g["gap_type"], "severity": g["severity"], "detail": g["detail"],
             "sources": [gap_cite]}
            for g in conn.execute("SELECT column_name, gap_type, severity, detail FROM governance_gaps "
                                  "WHERE table_fqn = ? ORDER BY severity = 'high' DESC, gap_type", (fqn,))]

    return {
        "status": "found",
        "table": {"fqn": fqn, "name": t["name"], "layer": t["layer"], "domain": t["domain"],
                  "description": t["description"], "status": t["status"], "row_count": t["row_count"],
                  "sensitivity": t["sensitivity"], "search_visible": bool(t["search_visible"]),
                  "built_for": t["built_for"], "url": t["url"], "sources": table_src},
        "contract": contract,
        "owners": owners,
        "certification": certification,
        "data_category": data_category,
        "pi_columns": pi_columns,
        "governance_gaps": gaps,
        "citations": cites.as_list(),
    }


# --- Tool definitions (provider-neutral) -----------------------------------------

TOOLS = [
    {
        "name": "find_table",
        "description": (
            "Search the data catalog for tables by name or description. Use this when the user's "
            "table name is partial, fuzzy or unknown. Returns status found | ambiguous | not_found "
            "and up to `limit` matches with a score, best first. Marts (modelled tables) rank above "
            "source tables; when found, the first match is the table to use and any other strong "
            "matches can be mentioned as 'also matched'. Staging (stg_) and intermediate (int_) "
            "tables only match on their exact name. With kind=\"exposure\" it searches dashboards, "
            "reports, ML models and applications instead, returning their exposure ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Table name or words describing it"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "kind": {"type": "string", "enum": ["table", "exposure"], "default": "table",
                         "description": "table (default), or exposure for dashboards, reports, ML models, apps"},
            },
            "required": ["query"],
        },
        "function": find_table,
    },
    {
        "name": "get_table",
        "description": (
            "Get governance facts for exactly one table: data contract, owners and support team, "
            "certification (with failed criteria), Data Category with retention and "
            "stewards, PI columns and governance gaps. Accepts a fully qualified name, "
            "schema.name or an exact table name. Every fact lists citation ids in `sources`; "
            "`citations` describes each id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "e.g. fernwood_dw.marts.dim_customers or dim_customers"},
            },
            "required": ["table"],
        },
        "function": get_table,
    },
]

"""Column tool for the PI Classifier: get_columns.

Returns one table's columns split into `open_columns` (pi_status suggested or
unclassified: the ones to recommend on) and `reviewed_columns` (confirmed or
rejected by a steward: context only). Every open column carries its stored
profile (aggregates computed at ingest, never values), the signals code
derives from metadata and profile, and the same column up/downstream in
column lineage with its recorded PI. Nothing here reads the warehouse.

Citation ids:
  catalog:<fqn>#<column>     a column's catalog entry: metadata, recorded PI,
                             profile and signals (also for neighbours)
  catalog:<fqn>              the table's catalog entry

Signals (data-model §7, PI agent D6) each say what they suggest: a PI
pi_type, or "not_pi". Strong signals can contradict a decision; weak ones
(shapes shared by PI and non-PI columns, such as dates, and words in a
description, which often describe the row rather than the value: "Emails
sent") only support one.
The guard computes confidence from them (D8), so the LLM's own certainty does
not count.

Blind mode (evals only, D4): every column of the table is treated as open and
recorded PI is hidden on the table and its neighbours, so an agent can be
measured on columns a steward already reviewed. The LLM cannot turn it on:
it is not in the tool's input schema.
"""

import json
import re
import sqlite3
from functools import partial

from catalog_iq.catalog.db import settings
from catalog_iq.tools.tables import _Citations, _resolve

OPEN = ("suggested", "unclassified")
# sensitivity follows from the PI type, never chosen (data-model §3.1)
SENSITIVITY = {
    "email": "Internal", "hashed_email": "Internal",
    "name": "Confidential", "phone": "Confidential", "address": "Confidential", "ip_address": "Confidential",
    "free_text": "Confidential",
    "personal_email": "Highly Confidential", "salary": "Highly Confidential", "joining_date": "Highly Confidential",
    "termination_date": "Highly Confidential", "date_of_birth": "Highly Confidential",
    "payment_card": "Highly Confidential", "government_id": "Highly Confidential",
}
# words in a column's name or description that suggest a PI type
NAME_PATTERNS = {
    "email": r"e-?mail(?!_sha|_hash)",
    "personal_email": r"personal[ _]e-?mail|home[ _]e-?mail",
    "hashed_email": r"e-?mail[ _]?(sha|hash)|(sha256|hash)\w*[ _]e-?mail",
    "name": r"\b(full|first|last|middle|given|family|sur)[ _]?name|requester[ _]name|name of the person",
    "phone": r"phone|mobile|telephone",
    "address": r"(home|shipping|billing|street|mailing|postal)[ _]?(address|code)|postcode|\bzip\b",
    "ip_address": r"\bip[ _]address|\bip\b",
    "free_text": r"free[- ]?text|message|\bbody|comment|\bnotes?\b|_text\b",
    "date_of_birth": r"birth|\bdob\b",
    "joining_date": r"joining|hire[ _]date|date joined",
    "termination_date": r"terminat|employment ended",
    "salary": r"salary|compensation|\bwages?\b",
    "payment_card": r"\bcard|\bpan\b",
    "government_id": r"national[ _]id|\bssn\b|passport|government",
}
# value patterns (ingest PROFILE_PATTERNS) → PI types they fit; weak = shared with non-PI columns
STRONG_PATTERNS = {"email": ["email", "personal_email"], "sha256_hex": ["hashed_email"], "phone": ["phone"],
                   "ipv4": ["ip_address"]}
WEAK_PATTERNS = {"person_name": ["name"], "iso_date": ["date_of_birth", "joining_date", "termination_date"],
                 "four_digits": ["payment_card", "government_id"], "five_digits": ["address"]}
PATTERN_MIN = 90       # % of non-null values
FEW_DISTINCT = 12      # a text column with at most this many values is a code list or template
FREE_TEXT_WORDS = 5    # average words per value
FREE_TEXT_DISTINCT = 50  # % distinct


def _pct(ratio: float | None) -> float | None:
    return None if ratio is None else round(ratio * 100, 1)


def _profile(row: sqlite3.Row | None) -> dict | None:
    """Stored profile as percentages, so numbers in an answer can be checked against it."""
    if row is None:
        return None
    return {"rows": row["row_count"], "null_pct": _pct(row["null_ratio"]), "distinct": row["distinct_count"],
            "distinct_pct": _pct(row["distinct_ratio"]), "avg_length": row["avg_length"],
            "avg_words": row["avg_words"],
            "pattern_pct": {k: _pct(v) for k, v in json.loads(row["pattern_matches"]).items()}}


def _signal(kind: str, suggests: list[str], detail: str, strong: bool = True) -> dict:
    return {"signal": kind, "suggests": suggests, "strong": strong, "detail": detail}


def signals(col: dict, profile: dict | None, neighbours: list[dict], recorded: dict | None) -> list[dict]:
    """What code can tell about one column, each signal with what it suggests."""
    out = []

    def matches(text: str) -> list[str]:
        found = [c for c, p in NAME_PATTERNS.items() if re.search(p, text.lower())]
        return [c for c in found if c != "email"] if {"hashed_email", "personal_email"} & set(found) else found

    in_name = matches(col["name"])
    if in_name:
        out.append(_signal("name", in_name, f"column name suggests {', '.join(in_name)}"))
    # a description often mentions what the row is about ("Emails sent"), not what the value is
    in_description = [c for c in matches(col["description"] or "") if c not in in_name]
    if in_description:
        out.append(_signal("description", in_description,
                           f"description mentions {', '.join(in_description)}", strong=False))
    is_id = bool(col["is_pk"]) or (col["name"].endswith("_id") and "national" not in col["name"])
    if is_id:
        out.append(_signal("identifier", ["not_pi"], "a key or identifier column"))
    if profile:
        for pattern, pct in profile["pattern_pct"].items():
            if pct >= PATTERN_MIN and pattern in STRONG_PATTERNS:
                out.append(_signal("pattern", STRONG_PATTERNS[pattern], f"{pct}% of values match {pattern}"))
            elif pct >= PATTERN_MIN and pattern in WEAK_PATTERNS:
                out.append(_signal("pattern", WEAK_PATTERNS[pattern], f"{pct}% of values match {pattern} "
                                                                       "(a shape non-PI columns share)", False))
            elif pct >= PATTERN_MIN and pattern == "iso_timestamp":
                out.append(_signal("timestamp", ["not_pi"], f"{pct}% of values are timestamps"))
        text_col = profile["avg_words"] is not None
        if text_col and profile["distinct"] <= FEW_DISTINCT and profile["rows"] >= 100 and not is_id:
            out.append(_signal("few_distinct", ["not_pi"],
                               f"{profile['distinct']} distinct values in {profile['rows']} rows (code list or template)"))
        if (text_col and profile["avg_words"] >= FREE_TEXT_WORDS
                and (profile["distinct_pct"] or 0) >= FREE_TEXT_DISTINCT):
            out.append(_signal("free_text_shape", ["free_text"],
                               f"{profile['avg_words']} words per value, {profile['distinct_pct']}% distinct", False))
    for n in neighbours:
        if n.get("pi_status") == "confirmed":
            out.append(_signal("neighbour", [n["pi_type"]], f"{n['fqn']}.{n['column']} is confirmed "
                                                                f"{n['pi_type']} ({n['data_subject']})"))
        elif n.get("pi_status") == "rejected":
            out.append(_signal("neighbour", ["not_pi"], f"{n['fqn']}.{n['column']} was reviewed: not PI"))
    if recorded and recorded["status"] == "suggested":
        out.append(_signal("scanner", [recorded["pi_type"]],
                           f"{recorded['method']} scanner suggested {recorded['pi_type']} "
                           f"({recorded['confidence']}), not reviewed"))
    return out


def _neighbours(conn: sqlite3.Connection, fqn: str, column: str, blind: bool, cites: _Citations) -> list[dict]:
    rows = conn.execute("""
        SELECT 'upstream' AS direction, l.upstream_fqn AS fqn, l.upstream_column AS column, l.transform
        FROM column_lineage l WHERE l.table_fqn = ? AND l.column_name = ?
        UNION ALL
        SELECT 'downstream', l.table_fqn, l.column_name, l.transform
        FROM column_lineage l WHERE l.upstream_fqn = ? AND l.upstream_column = ?
        ORDER BY 1, 2, 3""", (fqn, column, fqn, column)).fetchall()
    out = []
    for r in rows:
        n = {"direction": r["direction"], "fqn": r["fqn"], "column": r["column"], "transform": r["transform"],
             "sources": [cites.add("catalog", f"{r['fqn']}#{r['column']}")]}
        if not blind:
            c = conn.execute("SELECT pi_status, pi_type, data_subject FROM columns WHERE table_fqn = ? AND name = ?",
                             (r["fqn"], r["column"])).fetchone()
            if c:
                n.update(pi_status=c["pi_status"], pi_type=c["pi_type"], data_subject=c["data_subject"])
        out.append(n)
    return out


def get_columns(conn: sqlite3.Connection, table: str, blind: bool = False) -> dict:
    t, error = _resolve(conn, table)
    if error:
        return error
    cites = _Citations(settings(conn)["base_url"])
    fqn = t["fqn"]
    table_cite = cites.add("catalog", fqn)
    head = {"fqn": fqn, "layer": t["layer"], "domain": t["domain"], "description": t["description"],
            "row_count": t["row_count"], "sources": [table_cite]}
    if t["layer"] in ("staging", "intermediate"):
        ups = [r[0] for r in conn.execute(
            "SELECT DISTINCT upstream_fqn FROM column_lineage WHERE table_fqn = ? ORDER BY 1", (fqn,))]
        return {"status": "not_applicable", "table": head,
                "note": ("stg_/int_ tables are not classified: their PI is inherited through column lineage "
                         "(data-model §6). Classify the upstream tables instead."),
                "upstream_tables": ups, "citations": cites.as_list(compact=True)}

    profiles = {r["column_name"]: r for r in conn.execute(
        "SELECT * FROM column_profiles WHERE table_fqn = ?", (fqn,))}
    open_cols, reviewed = [], []
    for c in conn.execute("SELECT * FROM columns WHERE table_fqn = ? ORDER BY ordinal", (fqn,)):
        cid = cites.add("catalog", f"{fqn}#{c['name']}")
        is_open = blind or c["pi_status"] in OPEN
        if not is_open:
            reviewed.append({"column": c["name"], "description": c["description"], "pi_status": c["pi_status"],
                             "pi_type": c["pi_type"], "data_subject": c["data_subject"], "sources": [cid]})
            continue
        recorded = None if blind else {
            "status": c["pi_status"], "pi_type": c["pi_type"], "subject": c["data_subject"],
            "method": c["detection_method"], "confidence": c["detection_confidence"]}
        profile = _profile(profiles.get(c["name"]))
        neighbours = _neighbours(conn, fqn, c["name"], blind, cites)
        open_cols.append({
            "column": c["name"], "data_type": c["data_type"], "description": c["description"],
            "is_pk": bool(c["is_pk"]), "recorded": recorded, "profile": profile,
            "signals": signals(dict(c), profile, neighbours, recorded),
            "neighbours": neighbours, "sources": [cid]})
    return {"status": "found", "table": head, "open_columns": open_cols, "reviewed_columns": reviewed,
            "note": None if open_cols else "No open columns: every column has been reviewed by a steward.",
            "citations": cites.as_list(compact=True)}


INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "table": {"type": "string", "description": "e.g. fernwood_dw.raw_returns.return_requests or return_requests"},
    },
    "required": ["table"],
}
DESCRIPTION = (
    "Get one table's columns for PI review. `open_columns` (suggested or unclassified) are the ones to "
    "recommend on; each has its description, recorded scanner suggestion, profile (percentages over all "
    "rows; no values), signals computed from metadata and profile, and the same column up/downstream "
    "with its recorded PI. `reviewed_columns` were decided by a steward: context only. Accepts a fully "
    "qualified name, schema.name or an exact table name. Cite each column with the id in its `sources`."
)


def column_tools(blind: bool = False) -> list[dict]:
    return [{"name": "get_columns", "description": DESCRIPTION, "input_schema": INPUT_SCHEMA,
             "function": partial(get_columns, blind=blind)}]


COLUMN_TOOLS = column_tools()

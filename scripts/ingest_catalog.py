"""Ingest the Fernwood metadata sources into a single SQLite catalog.

Reads the sources listed in config/sources.yaml (Collibra-style export, ODCS
data contracts, dbt manifest and the warehouse schema), joins them on the
table's fully qualified name and applies the governance rules from the data
model: stg_/int_ inheritance (PI via column lineage; owner and data category from the
source or built_for mart), one data category per table assigned in Collibra,
certification and governance gaps. Agents query the resulting catalog.db; they
never read the source files.

The warehouse is opened read-only. Its schema and row counts are read, and
each column is profiled (null/distinct ratios, value-pattern match ratios):
only aggregates are stored, never a value.

Usage: python3 scripts/ingest_catalog.py [--force]
Output: data/catalog/catalog.db
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "sources.yaml"
SCHEMA_VERSION = 4  # bump when the catalog schema or rules change

SENSITIVITY_RANK = {"Public": 0, "Internal": 1, "Confidential": 2, "Highly Confidential": 3}
PI_SENSITIVITY = {
    "email": "Internal",
    "hashed_email": "Internal",
    "name": "Confidential",
    "phone": "Confidential",
    "address": "Confidential",
    "ip_address": "Confidential",
    "free_text": "Confidential",
    "personal_email": "Highly Confidential",
    "salary": "Highly Confidential",
    "joining_date": "Highly Confidential",
    "termination_date": "Highly Confidential",
    "date_of_birth": "Highly Confidential",
    "payment_card": "Highly Confidential",
    "government_id": "Highly Confidential",
}
REVIEWED = {"confirmed", "rejected"}
LAYERS = {"staging": "staging", "intermediate": "intermediate", "marts": "mart"}

# value patterns for column profiles (data-model §3.2); the PI agent reads the
# match ratios as evidence, so names describe the shape, not a PI type
PROFILE_PATTERNS = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"),
    "sha256_hex": re.compile(r"^[0-9a-f]{64}$"),
    "phone": re.compile(r"^(\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$"),
    "ipv4": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
    "iso_date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "iso_timestamp": re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),
    "person_name": re.compile(r"^[A-Z][a-z'-]+( [A-Z][a-z'-]+)+$"),
    "four_digits": re.compile(r"^\d{4}$"),
    "five_digits": re.compile(r"^\d{5}$"),
}

# --- Catalog schema ------------------------------------------------------------

SCHEMA = """
CREATE TABLE ingest_runs (
    run_id TEXT, ingested_at TEXT, schema_version INTEGER,
    source TEXT, path TEXT, sha256 TEXT
);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE people (
    username TEXT PRIMARY KEY, full_name TEXT, email TEXT UNIQUE, title TEXT,
    region TEXT, steward_since TEXT
);
CREATE TABLE user_groups (name TEXT PRIMARY KEY, description TEXT, email TEXT, slack TEXT);
CREATE TABLE user_group_members (group_name TEXT, username TEXT);

CREATE TABLE data_categories (
    data_category_id TEXT PRIMARY KEY, name TEXT, domain TEXT, status TEXT, definition TEXT,
    retention_months INTEGER, retention_trigger TEXT, disposal_action TEXT,
    legal_basis TEXT, url TEXT
);
CREATE TABLE data_category_stewardship (data_category_id TEXT, role TEXT, username TEXT);
-- keyword search over data categories (search_data_categories): name, definition, linked glossary terms + synonyms
CREATE VIRTUAL TABLE data_category_search USING fts5(data_category_id UNINDEXED, name, definition, terms,
                                           tokenize = 'porter unicode61');

CREATE TABLE glossary_terms (
    term_id TEXT PRIMARY KEY, name TEXT, status TEXT, definition TEXT,
    synonyms TEXT, owner_username TEXT, url TEXT
);
CREATE TABLE term_links (term_id TEXT, entity_type TEXT, entity_id TEXT);

CREATE TABLE tables (
    fqn TEXT PRIMARY KEY,
    schema_name TEXT, name TEXT, layer TEXT, domain TEXT, description TEXT,
    status TEXT,                 -- active | deprecated | in_development
    materialization TEXT,
    warehouse_table TEXT, row_count INTEGER,
    in_warehouse INTEGER, in_collibra INTEGER, in_dbt INTEGER, has_contract INTEGER,
    dbt_unique_id TEXT,
    contract_id TEXT, contract_status TEXT, contract_version TEXT, data_product TEXT,
    built_for TEXT,              -- int_ only: FQN of the mart it exists to build
    owner_status TEXT,           -- contract | inherited | none
    owner_inherited_from TEXT,
    data_category_id TEXT,                 -- table-level categorization (data-model §4)
    data_category_source TEXT,             -- collibra | inherited | none
    data_category_inherited_from TEXT,
    sensitivity TEXT,
    certification TEXT,          -- certified | uncertified | not_applicable
    cert_failed_criteria TEXT, cert_details TEXT,  -- JSON arrays
    search_visible INTEGER,
    url TEXT
);
CREATE TABLE columns (
    table_fqn TEXT, name TEXT, ordinal INTEGER, data_type TEXT, description TEXT,
    is_pk INTEGER,
    pi_status TEXT,              -- confirmed | rejected | suggested | unclassified | derived
    is_pi INTEGER, pi_type TEXT, data_subject TEXT, sensitivity TEXT,
    detection_method TEXT, detection_confidence REAL, detected_on TEXT,
    reviewed_by TEXT, reviewed_on TEXT,
    classification_source TEXT,  -- collibra | inherited | none
    transform TEXT, expression TEXT,  -- stg_/int_ only
    in_warehouse INTEGER, in_collibra INTEGER, in_contract INTEGER, in_dbt INTEGER,
    url TEXT,
    PRIMARY KEY (table_fqn, name)
);
CREATE TABLE column_lineage (
    table_fqn TEXT, column_name TEXT, upstream_fqn TEXT, upstream_column TEXT,
    transform TEXT
);
CREATE TABLE lineage_edges (
    upstream_id TEXT, downstream_id TEXT, downstream_type TEXT  -- table | exposure
);
CREATE TABLE exposures (
    exposure_id TEXT PRIMARY KEY, name TEXT, label TEXT, type TEXT, maturity TEXT,
    description TEXT, owner_name TEXT, owner_email TEXT, owner_username TEXT, url TEXT
);

CREATE TABLE ownership (
    table_fqn TEXT, username TEXT, full_name TEXT, email TEXT, role TEXT,
    inherited_from TEXT
);
CREATE TABLE support_channels (
    table_fqn TEXT, channel TEXT, tool TEXT, scope TEXT, inherited_from TEXT
);
CREATE TABLE slas (table_fqn TEXT, property TEXT, value TEXT, unit TEXT);
CREATE TABLE contract_quality (table_fqn TEXT, rule TEXT, columns TEXT, description TEXT);
CREATE TABLE dbt_tests (
    test_id TEXT PRIMARY KEY, table_fqn TEXT, column_name TEXT, test_name TEXT, kwargs TEXT
);

CREATE TABLE column_profiles (  -- aggregates only, no values (data-model §3.2)
    table_fqn TEXT, column_name TEXT, row_count INTEGER,
    null_ratio REAL, distinct_count INTEGER, distinct_ratio REAL,  -- distinct among non-null
    avg_length REAL, avg_words REAL,  -- text values only
    pattern_matches TEXT,             -- JSON {pattern: ratio of non-null values}, ratios > 0
    PRIMARY KEY (table_fqn, column_name)
);

CREATE TABLE governance_gaps (
    table_fqn TEXT, column_name TEXT, gap_type TEXT, severity TEXT, detail TEXT
);
CREATE TABLE provenance (entity_type TEXT, entity_id TEXT, source TEXT, ref TEXT);

CREATE INDEX ix_columns_pi ON columns (is_pi, pi_type);
CREATE INDEX ix_lineage_up ON lineage_edges (upstream_id);
CREATE INDEX ix_lineage_down ON lineage_edges (downstream_id);
CREATE INDEX ix_gaps_table ON governance_gaps (table_fqn);
"""

# --- Helpers -------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snake(value: str | None) -> str | None:
    return value.strip().lower().replace(" ", "_") if value else None


def max_sensitivity(values) -> str | None:
    values = [v for v in values if v]
    return max(values, key=SENSITIVITY_RANK.__getitem__) if values else None


# --- SQL column lineage (dbt raw_code) ------------------------------------------
# A small parser for the simple select statements in the manifest: one select
# list, a from clause with refs, sources and aliased subqueries, joins.

SQL_WORDS = {
    "and", "or", "not", "as", "integer", "real", "text", "null", "is", "case", "when",
    "then", "else", "end", "in", "like", "distinct", "true", "false",
}
VALUE_PRESERVING = r"(?:lower|upper|trim|min|max)"
COLREF = r"(?:[a-z_]\w*\.)?[a-z_]\w*"
TRANSFORM_RANK = {"passthrough": 0, "rename": 1, "combine": 2, "hash": 3, "derive": 4}


def top_level_matches(text: str, pattern: str) -> list[re.Match]:
    """Matches of pattern that are outside parentheses and string literals."""
    depth, in_str, depths = 0, False, []
    for ch in text:
        if ch == "'":
            in_str = not in_str
        elif not in_str and ch == "(":
            depth += 1
        elif not in_str and ch == ")":
            depth -= 1
        depths.append(depth if not in_str else -1)
    return [m for m in re.finditer(pattern, text) if depths[m.start()] == 0]


def split_top(text: str, sep: str = ",") -> list[str]:
    cuts = [m.start() for m in top_level_matches(text, re.escape(sep))]
    parts, prev = [], 0
    for cut in cuts:
        parts.append(text[prev:cut])
        prev = cut + len(sep)
    parts.append(text[prev:])
    return [p.strip() for p in parts]


def extract_subqueries(text: str) -> tuple[str, dict[str, str]]:
    """Replace top-level parenthesised subqueries with __sub_N__ tokens."""
    out, subs, depth, start = [], {}, 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inner = text[start + 1:i]
                if re.match(r"\s*select\b", inner):
                    token = f"__sub_{len(subs)}__"
                    subs[token] = inner
                    out.append(token)
                else:
                    out.append(text[start:i + 1])
        elif depth == 0:
            out.append(ch)
    return "".join(out), subs


def parse_select(sql: str, relmap: dict[str, str]) -> dict:
    sql = sql.strip()
    from_at = top_level_matches(sql, r"\bfrom\b")[0].start()
    select_list = re.sub(r"^select\s+", "", sql[:from_at])
    rest = sql[from_at + 4:]
    stops = top_level_matches(rest, r"\b(?:where|group\s+by|order\s+by|having|limit)\b")
    from_clause, subs = extract_subqueries(rest[:stops[0].start()] if stops else rest)

    relations = {}
    for m in re.finditer(r"(__rel_\d+__|__sub_\d+__)(?:\s+(?:as\s+)?(\w+))?", from_clause):
        token, alias = m.groups()
        if alias in {"on", "left", "right", "inner", "outer", "join", "cross", "full"}:
            alias = None
        if token.startswith("__rel"):
            fqn = relmap[token]
            relations[alias or fqn.split(".")[-1]] = ("table", fqn)
        else:
            relations[alias or token] = ("subquery", parse_select(subs[token], relmap))

    items = []
    for expr in split_top(select_list):
        as_matches = top_level_matches(expr, r"\bas\s+(\w+)\s*$")
        if as_matches:
            name, expr = as_matches[-1].group(1), expr[:as_matches[-1].start()].strip()
        else:
            name = expr.split(".")[-1] if re.fullmatch(COLREF, expr) else expr
        items.append({"name": name, "expr": expr})
    return {"items": items, "relations": relations}


def column_refs(expr: str) -> list[tuple[str | None, str]]:
    expr = re.sub(r"'[^']*'", "''", expr)
    refs = []
    for m in re.finditer(r"\b(?:([a-z_]\w*)\.)?([a-z_]\w*)\b(?!\s*\()", expr):
        qual, col = m.groups()
        if qual is None and col in SQL_WORDS:
            continue
        if (qual, col) not in refs:
            refs.append((qual, col))
    return refs


def item_lineage(item: dict, scope: dict, physical: dict) -> tuple[list, str]:
    """Return ([(upstream_fqn, upstream_column)], transform) for one select item."""
    expr, inputs, inner = item["expr"], [], []
    for qual, col in column_refs(expr):
        if qual:
            rels = [scope["relations"][qual]]
        else:
            rels = [r for r in scope["relations"].values()
                    if (r[0] == "table" and col in physical.get(r[1], []))
                    or (r[0] == "subquery" and any(i["name"] == col for i in r[1]["items"]))]
        if not rels:
            raise ValueError(f"cannot resolve column {col!r} in {expr!r}")
        kind, target = rels[0]
        if kind == "table":
            inputs.append((target, col))
        else:
            sub_item = next(i for i in target["items"] if i["name"] == col)
            sub_inputs, sub_transform = item_lineage(sub_item, target, physical)
            inputs.extend(sub_inputs)
            inner.append(sub_transform)

    if "sha256(" in expr:
        transform = "hash"
    elif re.fullmatch(COLREF, expr) or re.fullmatch(rf"{VALUE_PRESERVING}\(\s*{COLREF}\s*\)", expr):
        transform = "passthrough"
    elif "||" in expr and all(re.fullmatch(rf"{COLREF}|'[^']*'", p) for p in split_top(expr, "||")):
        transform = "combine"
    else:
        transform = "derive"
    transform = max([transform, *inner], key=TRANSFORM_RANK.__getitem__)
    if transform in ("passthrough", "rename") and len(inputs) == 1 and inputs[0][1] != item["name"]:
        transform = "rename"
    return list(dict.fromkeys(inputs)), transform


def model_column_lineage(node: dict, ref_fqn: dict, source_fqn: dict, physical: dict) -> dict:
    relmap = {}

    def to_token(fqn: str) -> str:
        token = f"__rel_{len(relmap)}__"
        relmap[token] = fqn
        return token

    sql = re.sub(r"\{\{\s*ref\('(\w+)'\)\s*\}\}", lambda m: to_token(ref_fqn[m.group(1)]), node["raw_code"])
    sql = re.sub(r"\{\{\s*source\('(\w+)',\s*'(\w+)'\)\s*\}\}",
                 lambda m: to_token(source_fqn[(m.group(1), m.group(2))]), sql)
    scope = parse_select(sql, relmap)
    return {item["name"]: item_lineage(item, scope, physical) for item in scope["items"]}


# --- Governance rules ------------------------------------------------------------


def inherit_column(transform: str, upstream: list[dict]) -> dict:
    """Classify a stg_/int_ column from its upstream columns (data model §6)."""
    blank = {"pi_status": "derived", "is_pi": 0, "pi_type": None, "data_subject": None,
             "sensitivity": None, "detection_method": None, "detection_confidence": None}
    if not upstream:
        return {**blank, "sensitivity": "Internal"}
    src = upstream[0]
    copy = {k: src.get(k) for k in blank}

    if transform in ("passthrough", "rename"):
        return copy
    if transform == "hash":
        if src["pi_status"] in ("confirmed", "suggested"):
            return {**copy, "pi_type": "hashed_email", "sensitivity": PI_SENSITIVITY["hashed_email"]}
        return copy
    if transform == "combine":
        keys = {(u["pi_status"], u["pi_type"]) for u in upstream}
        if len(keys) == 1:
            return {**copy, "sensitivity": max_sensitivity(u["sensitivity"] for u in upstream)}
    # derive: the result is not PI
    sens = max_sensitivity("Internal" if u["is_pi"] else u["sensitivity"] for u in upstream)
    return {**blank, "sensitivity": sens or "Internal"}


def certify(table: dict, columns: list[dict], contract: dict | None, data_categories: dict) -> dict:
    """Derived certification for sources and marts (§5)."""
    if table["layer"] in ("staging", "intermediate"):
        return {"status": "not_applicable", "failed": [], "details": ["stg_/int_ tables are never certified"]}
    if table["status"] == "in_development":
        return {"status": "not_applicable", "failed": [], "details": ["in development: not deployed to the warehouse"]}
    failed, details = [], []
    if contract is None:
        failed.append("contract_missing")
        details.append("No data contract")
    elif contract["status"] != "active":
        failed.append("contract_not_active")
        details.append(f"Contract status is {contract['status']}")
    elif not any(t.get("role") == "Owner" for t in contract.get("team", [])) or not contract.get("support"):
        failed.append("contract_incomplete")
        details.append("Contract has no owner or no support channel")

    unreviewed = [c for c in columns if c["in_collibra"] and c["pi_status"] not in REVIEWED]
    if unreviewed:
        failed.append("pi_not_reviewed")
        for c in unreviewed:
            note = f" {c['pi_type']} ({c['detection_confidence']})" if c["pi_type"] else ""
            details.append(f"{c['name']}: {c['pi_status']}{note}, not reviewed")

    if table["data_category_id"] is None:
        failed.append("data_category_missing")
        details.append("No data category assigned in Collibra")
    elif data_categories[table["data_category_id"]]["status"] != "Approved":
        failed.append("data_category_not_approved")
        details.append(f"{table['data_category_id']} is {data_categories[table['data_category_id']]['status']}, not Approved")
    return {"status": "uncertified" if failed else "certified", "failed": failed, "details": details}


# --- Source loading --------------------------------------------------------------


def load_collibra(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {"meta": data["exportMetadata"], "users": {}, "groups": data["userGroups"],
           "data_categories": {}, "terms": {}, "tables": {}, "columns": {}}
    for u in data["users"]:
        out["users"][u["userName"]] = u
    for a in data["assets"]:
        attrs, rels = a.get("attributes", {}), a.get("relations", {})
        if a["type"] == "Data Category":
            out["data_categories"][a["dataCategoryId"]] = a
        elif a["type"] == "Business Term":
            out["terms"][a["externalId"]] = a
        elif a["type"] == "Table":
            out["tables"][attrs["Full Name"]] = a
        elif a["type"] == "Column":
            table_fqn = rels["is part of"][0]
            out["columns"].setdefault(table_fqn, {})[a["name"]] = a
    return out


def collibra_column(asset: dict) -> dict:
    attrs = asset["attributes"]
    status = snake(attrs.get("PI Status")) or "unclassified"
    return {
        "name": asset["name"], "data_type": attrs.get("Data Type"),
        "description": attrs.get("Description"), "is_pk": int(bool(attrs.get("Is Primary Key"))),
        "pi_status": status, "is_pi": int(status == "confirmed"),
        "pi_type": attrs.get("PI Type"), "data_subject": snake(attrs.get("Data Subject")),
        "sensitivity": attrs.get("Sensitivity"), "detection_method": snake(attrs.get("Detection Method")),
        "detection_confidence": attrs.get("Detection Confidence"), "detected_on": attrs.get("Detected On"),
        "reviewed_by": attrs.get("Reviewed By"), "reviewed_on": attrs.get("Reviewed On"),
        "classification_source": "collibra", "external_id": asset["externalId"],
    }


def load_contracts(directory: Path) -> dict:
    contracts = {}
    for path in sorted(directory.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        contracts[doc["schema"][0]["physicalName"]] = {**doc, "_path": path}
    return contracts


def profile_column(values: list) -> dict:
    """Aggregate statistics of one column. Values are read here and dropped."""
    present = [v for v in values if v is not None]
    texts = [v for v in present if isinstance(v, str)]
    n = len(present)
    matches = {k: sum(bool(p.match(v)) for v in texts) / n for k, p in PROFILE_PATTERNS.items()} if n else {}
    return {
        "row_count": len(values),
        "null_ratio": round(1 - n / len(values), 3) if values else None,
        "distinct_count": len(set(present)),
        "distinct_ratio": round(len(set(present)) / n, 3) if n else None,
        "avg_length": round(sum(map(len, texts)) / len(texts), 1) if texts else None,
        "avg_words": round(sum(len(v.split()) for v in texts) / len(texts), 1) if texts else None,
        "pattern_matches": {k: round(r, 3) for k, r in matches.items() if r > 0},
    }


def load_warehouse(path: Path, database: str) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for name, kind in conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"):
        schema, table = name.split("__", 1)
        cols = [{"name": r[1], "data_type": r[2], "is_pk": int(r[5] > 0)}
                for r in conn.execute(f"PRAGMA table_info({name})")]
        rows = conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        profiles = {c["name"]: profile_column([r[0] for r in conn.execute(f'SELECT "{c["name"]}" FROM {name}')])
                    for c in cols}
        out[f"{database}.{schema}.{table}"] = {"table": name, "kind": kind, "columns": cols, "rows": rows,
                                               "profiles": profiles}
    conn.close()
    return out


# --- Build -----------------------------------------------------------------------


def source_files(cfg: dict) -> list[tuple[str, Path]]:
    files = [("config", CONFIG_PATH), ("collibra", ROOT / cfg["collibra"]["export"]),
             ("warehouse", ROOT / cfg["warehouse"]["path"])]
    files += [("dbt", ROOT / p["manifest"]) for p in cfg["dbt"]]
    files += [("contract", p) for p in sorted((ROOT / cfg["contracts"]["dir"]).glob("*.yaml"))]
    return files


def build(cfg: dict, db_path: Path, run_id: str, hashes: list) -> dict:
    database = cfg["warehouse"]["database"]
    base_url = cfg["catalog"]["base_url"]
    collibra = load_collibra(ROOT / cfg["collibra"]["export"])
    contracts = load_contracts(ROOT / cfg["contracts"]["dir"])
    warehouse = load_warehouse(ROOT / cfg["warehouse"]["path"], database)
    manifests = [json.loads((ROOT / p["manifest"]).read_text()) for p in cfg["dbt"]]

    models, sources, tests, exposures = {}, {}, [], []
    for m in manifests:
        sources.update(m["sources"])
        for uid, node in m["nodes"].items():
            if node["resource_type"] == "model":
                models[uid] = node
            elif node["resource_type"] == "test":
                tests.append(node)
        exposures.extend(m["exposures"].values())
    uid_fqn = {uid: n["relation_name"] for uid, n in {**models, **sources}.items()}
    ref_fqn = {n["name"]: n["relation_name"] for n in models.values()}
    source_fqn = {(n["source_name"], n["name"]): n["relation_name"] for n in sources.values()}
    dbt_by_fqn = {n["relation_name"]: n for n in {**models, **sources}.values()}
    email_user = {u["emailAddress"]: name for name, u in collibra["users"].items()}

    data_categories = {}
    for data_category_id, a in collibra["data_categories"].items():
        attrs = a["attributes"]
        data_categories[data_category_id] = {"data_category_id": data_category_id, "name": a["name"], "domain": a["domain"], "status": a["status"],
                        "definition": attrs.get("Definition"),
                        "retention_months": attrs.get("Retention Period (Months)"),
                        "retention_trigger": attrs.get("Retention Trigger"),
                        "disposal_action": attrs.get("Disposal Action"),
                        "legal_basis": attrs.get("Legal Basis"), "url": f"{base_url}/data-category/{data_category_id}"}

    gaps = []

    def gap(table_fqn, column, gap_type, severity, detail):
        gaps.append((table_fqn, column, gap_type, severity, detail))

    # --- tables and their columns
    all_fqns = sorted(set(warehouse) | set(collibra["tables"]) | set(contracts) | set(dbt_by_fqn))
    tables, columns = {}, {}
    for fqn in all_fqns:
        _, schema, name = fqn.split(".")
        layer = "source" if schema.startswith("raw_") else LAYERS[schema]
        wh, ca, dbt = warehouse.get(fqn), collibra["tables"].get(fqn), dbt_by_fqn.get(fqn)
        tags = (dbt or {}).get("config", {}).get("tags", [])
        status = ("in_development" if "in_development" in tags
                  else "deprecated" if "deprecated" in tags or (ca and ca["status"] == "Deprecated")
                  else "active")
        tables[fqn] = {
            "fqn": fqn, "schema_name": schema, "name": name, "layer": layer,
            "domain": ca["attributes"]["Business Domain"] if ca else None,
            "description": (ca["attributes"]["Description"] if ca else None) or (dbt or {}).get("description"),
            "status": status,
            "materialization": "source" if layer == "source" else (dbt or {}).get("config", {}).get("materialized"),
            "warehouse_table": wh["table"] if wh else None, "row_count": wh["rows"] if wh else None,
            "in_warehouse": int(bool(wh)), "in_collibra": int(bool(ca)), "in_dbt": int(bool(dbt)),
            "has_contract": int(fqn in contracts), "dbt_unique_id": (dbt or {}).get("unique_id"),
            "built_for": uid_fqn.get((dbt or {}).get("meta", {}).get("built_for")),
            "url": f"{base_url}/table/{fqn}",
        }
        if not wh:
            gap(fqn, None, "table_not_in_warehouse", "medium",
                "Defined in " + ", ".join(s for s, ok in (("dbt", dbt), ("Collibra", ca), ("contract", fqn in contracts)) if ok)
                + " but not built in the warehouse")

    # source and mart columns: warehouse ∪ Collibra
    for fqn, t in tables.items():
        if t["layer"] not in ("source", "mart"):
            continue
        wh_cols = {c["name"]: c for c in warehouse.get(fqn, {}).get("columns", [])}
        cb_cols = collibra["columns"].get(fqn, {})
        contract_cols = {p["name"] for p in contracts[fqn]["schema"][0]["properties"]} if fqn in contracts else set()
        dbt_doc = (dbt_by_fqn.get(fqn) or {}).get("columns", {})
        dbt_cols = set(dbt_doc)
        contract_doc = ({p["name"]: p for p in contracts[fqn]["schema"][0]["properties"]}
                        if fqn in contracts else {})
        names = list(wh_cols) + [n for n in cb_cols if n not in wh_cols]
        if not t["in_warehouse"]:  # dbt-only model: catalog what dbt documents
            names += [n for n in dbt_doc if n not in names]
        cols = []
        for i, n in enumerate(names):
            if n in cb_cols:
                col = collibra_column(cb_cols[n])
            else:
                # Not registered in Collibra: no PI classification; describe it from dbt or the contract.
                fallback = dbt_doc.get(n) or {}
                col = {"name": n,
                       "data_type": wh_cols[n]["data_type"] if n in wh_cols else fallback.get("data_type"),
                       "description": fallback.get("description") or contract_doc.get(n, {}).get("description"),
                       "is_pk": wh_cols[n]["is_pk"] if n in wh_cols else 0,
                       "pi_status": "unclassified", "is_pi": 0,
                       "pi_type": None, "data_subject": None, "sensitivity": None,
                       "detection_method": None, "detection_confidence": None, "detected_on": None,
                       "reviewed_by": None, "reviewed_on": None,
                       "classification_source": "none"}
            col.update({"ordinal": i, "in_warehouse": int(n in wh_cols), "in_collibra": int(n in cb_cols),
                        "in_contract": int(n in contract_cols) if fqn in contracts else None,
                        "in_dbt": int(n in dbt_cols) if fqn in dbt_by_fqn else None})
            cols.append(col)
            if t["in_warehouse"] and n not in wh_cols:
                gap(fqn, n, "column_not_in_warehouse", "medium",
                    f"Registered in Collibra ({col['pi_status']}) but missing from the warehouse")
            if t["in_collibra"] and n not in cb_cols:
                gap(fqn, n, "column_not_registered", "medium", "In the warehouse but not registered in Collibra")
            if col["pi_status"] == "suggested":
                gap(fqn, n, "pi_suggested", "medium",
                    f"Suggested {col['pi_type']} ({col['detection_confidence']}), awaiting steward review")
            if col["pi_status"] == "unclassified" and t["in_collibra"]:
                gap(fqn, n, "pi_unclassified", "medium", "Never scanned for PI")
            if col["pi_type"] and col["sensitivity"] != PI_SENSITIVITY.get(col["pi_type"]):
                gap(fqn, n, "sensitivity_mismatch", "medium",
                    f"{col['pi_type']} should be {PI_SENSITIVITY.get(col['pi_type'])}, recorded {col['sensitivity']}")
        if fqn in contracts and t["in_warehouse"]:
            for n in sorted(contract_cols - set(wh_cols)):
                gap(fqn, n, "contract_column_missing", "high", "Contract requires this column; the warehouse does not have it")
            for n in [n for n in wh_cols if n not in contract_cols]:
                gap(fqn, n, "column_not_in_contract", "low", "In the warehouse but not in the contract (contract out of date)")
        if fqn in dbt_by_fqn and t["in_warehouse"]:
            for n in sorted(dbt_cols - set(wh_cols)):
                gap(fqn, n, "dbt_column_not_in_warehouse", "low", "Documented in dbt but not in the warehouse")
        columns[fqn] = cols

    # stg_/int_ columns: inherited via column lineage, in dependency order
    physical = {fqn: [c["name"] for c in w["columns"]] for fqn, w in warehouse.items()}
    lineage_rows = []
    pending = [fqn for fqn, t in tables.items() if t["layer"] in ("staging", "intermediate")]
    while pending:
        progressed = False
        for fqn in list(pending):
            node = dbt_by_fqn[fqn]
            ups = [uid_fqn[u] for u in node["depends_on"]["nodes"]]
            if any(u in pending for u in ups):
                continue
            col_lineage = model_column_lineage(node, ref_fqn, source_fqn, physical)
            if list(col_lineage) != physical.get(fqn):
                raise ValueError(f"{fqn}: parsed columns {list(col_lineage)} != warehouse {physical.get(fqn)}")
            by_name = {u: {c["name"]: c for c in columns.get(u, [])} for u in ups}
            dbt_cols = node.get("columns", {})
            cols = []
            for i, (n, (inputs, transform)) in enumerate(col_lineage.items()):
                upstream = []
                for up_fqn, up_col in inputs:
                    meta = by_name.get(up_fqn, {}).get(up_col)
                    if meta is None:
                        gap(fqn, n, "upstream_column_unregistered", "medium",
                            f"Inherits from {up_fqn}.{up_col}, which has no classification")
                        meta = {"pi_status": "unclassified", "is_pi": 0, "pi_type": None,
                                "data_subject": None, "sensitivity": None, "detection_method": None,
                                "detection_confidence": None}
                    upstream.append(meta)
                    lineage_rows.append((fqn, n, up_fqn, up_col, transform))
                wh_col = next(c for c in warehouse[fqn]["columns"] if c["name"] == n)
                col = inherit_column(transform, upstream)
                col.update({
                    "name": n, "ordinal": i, "data_type": wh_col["data_type"],
                    "description": (dbt_cols.get(n) or {}).get("description"), "is_pk": wh_col["is_pk"],
                    "detected_on": None, "reviewed_by": None, "reviewed_on": None,
                    "classification_source": "inherited", "transform": transform,
                    "expression": ", ".join(f"{u}.{c}" for u, c in inputs) or None,
                    "in_warehouse": 1, "in_collibra": 0, "in_contract": None, "in_dbt": int(n in dbt_cols),
                })
                cols.append(col)
            for n in sorted(set(dbt_cols) - set(col_lineage)):
                gap(fqn, n, "dbt_column_not_in_warehouse", "low", "Documented in dbt but not produced by the model SQL")
            columns[fqn] = cols
            pending.remove(fqn)
            progressed = True
        if not progressed:
            raise ValueError(f"lineage cycle or missing upstream among {pending}")

    # --- Data category, ownership, certification
    ownership, support, slas, quality = [], [], [], []

    def contract_people(fqn, contract, inherited_from):
        for member in contract.get("team", []):
            username = email_user.get(member["username"])
            if username is None:
                gap(fqn, None, "unknown_person", "low", f"{member['username']} is not a Collibra user")
            ownership.append((fqn, username, member.get("name"), member["username"], member.get("role"), inherited_from))
        for s in contract.get("support", []):
            support.append((fqn, s["channel"], s.get("tool"), s.get("scope"), inherited_from))

    # one data category per table, a represents relation on the Collibra Table asset (§4)
    assigned_data_category = {}
    for fqn, ca in collibra["tables"].items():
        represents = ca.get("relations", {}).get("represents", [])
        assigned_data_category[fqn] = represents[0] if represents else None
        if len(represents) > 1:
            gap(fqn, None, "multiple_data_categories", "high", f"Collibra assigns {', '.join(represents)}; a table has one data category")

    for fqn, t in tables.items():
        contract = contracts.get(fqn)
        # stg_ inherits from its source, int_ from the mart it is built_for (§6): owner and data category alike
        if t["layer"] == "staging":
            owner_from = next(uid_fqn[u] for u in dbt_by_fqn[fqn]["depends_on"]["nodes"] if u.startswith("source."))
        elif t["layer"] == "intermediate":
            owner_from = t["built_for"]
            if owner_from is None:
                gap(fqn, None, "built_for_missing", "high", "int_ model has no meta.built_for")
        else:
            owner_from = None

        data_category_id = assigned_data_category.get(owner_from or fqn)
        t.update({"data_category_id": data_category_id,
                  "data_category_source": ("inherited" if owner_from else "collibra") if data_category_id else "none",
                  "data_category_inherited_from": owner_from,
                  "sensitivity": max_sensitivity(c["sensitivity"] for c in columns.get(fqn, []))})
        if data_category_id is None and t["status"] != "in_development":
            gap(fqn, None, "data_category_missing", "high",
                f"Inherits its data category from {owner_from}, which has none" if owner_from else "No data category assigned in Collibra")
        elif data_category_id and data_categories[data_category_id]["retention_months"] is None:
            gap(fqn, None, "retention_not_defined", "medium", f"{data_category_id} has no retention policy")

        if owner_from:
            parent = contracts.get(owner_from)
            if parent and parent["status"] != "draft":
                contract_people(fqn, parent, owner_from)
                t.update({"owner_status": "inherited", "owner_inherited_from": owner_from})
            else:
                t.update({"owner_status": "none", "owner_inherited_from": owner_from})
                gap(fqn, None, "owner_missing", "high",
                    f"Owner is inherited from {owner_from}, which has no data contract: refer to Data Strategy")
        elif contract:
            contract_people(fqn, contract, None)
            t["owner_status"] = "contract"
            for s in contract.get("slaProperties", []):
                slas.append((fqn, s["property"], str(s["value"]), s.get("unit")))
            for q in contract.get("quality", []):
                quality.append((fqn, q["rule"], json.dumps(q.get("columns", [])), q.get("description")))
        else:
            t["owner_status"] = "none"

        if t["layer"] in ("source", "mart") and t["status"] != "in_development":
            if contract is None:
                gap(fqn, None, "contract_missing", "high", "No data contract: refer to Data Strategy")
            elif contract["status"] != "active":
                gap(fqn, None, "contract_not_active", "medium", f"Contract is {contract['status']} (v{contract['version']})")
        if contract:
            t.update({"contract_id": contract["id"], "contract_status": contract["status"],
                      "contract_version": contract["version"], "data_product": contract.get("dataProduct")})
        if t["domain"] is None and owner_from in tables:
            t["domain"] = tables[owner_from]["domain"]

        cert = certify(t, columns.get(fqn, []), contract, data_categories)
        t.update({"certification": cert["status"], "cert_failed_criteria": json.dumps(cert["failed"]),
                  "cert_details": json.dumps(cert["details"]),
                  "search_visible": int(t["layer"] in ("source", "mart") and t["status"] != "in_development")})

    # --- lineage
    edges = []
    for uid, node in models.items():
        for up in node["depends_on"]["nodes"]:
            edges.append((uid_fqn[up], node["relation_name"], "table"))
    for e in exposures:
        for up in e["depends_on"]["nodes"]:
            edges.append((uid_fqn[up], e["unique_id"], "exposure"))

    # --- write
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany("INSERT INTO ingest_runs VALUES (?,?,?,?,?,?)",
                     [(run_id, now, SCHEMA_VERSION, s, str(p.relative_to(ROOT)), h) for s, p, h in hashes])

    group = next(g for g in collibra["groups"] if g["name"] == cfg["escalation"]["user_group"])
    settings = {
        "warehouse_database": database, "base_url": base_url,
        "escalation_team": group["name"], "escalation_email": group["email"], "escalation_slack": group["slack"],
        "collibra_exported_at": collibra["meta"]["exportedAt"],
        "dbt_generated_at": manifests[0]["metadata"]["generated_at"],
        "dbt_version": manifests[0]["metadata"]["dbt_version"],
    }
    conn.executemany("INSERT INTO settings VALUES (?,?)", settings.items())

    conn.executemany("INSERT INTO people VALUES (?,?,?,?,?,?)", [
        (n, f"{u['firstName']} {u['lastName']}", u["emailAddress"], u.get("title"), u.get("region"), u.get("stewardSince"))
        for n, u in collibra["users"].items()])
    conn.executemany("INSERT INTO user_groups VALUES (?,?,?,?)",
                     [(g["name"], g.get("description"), g.get("email"), g.get("slack")) for g in collibra["groups"]])
    conn.executemany("INSERT INTO user_group_members VALUES (?,?)",
                     [(g["name"], m) for g in collibra["groups"] for m in g.get("members", [])])

    data_category_fields = ["data_category_id", "name", "domain", "status", "definition", "retention_months",
                  "retention_trigger", "disposal_action", "legal_basis", "url"]
    conn.executemany(f"INSERT INTO data_categories VALUES ({','.join('?' * len(data_category_fields))})",
                     [tuple(b[f] for f in data_category_fields) for b in data_categories.values()])
    conn.executemany("INSERT INTO data_category_stewardship VALUES (?,?,?)", [
        (data_category_id, role, user) for data_category_id, a in collibra["data_categories"].items()
        for role, users in a.get("responsibilities", {}).items() for user in users])

    conn.executemany("INSERT INTO glossary_terms VALUES (?,?,?,?,?,?,?)", [
        (tid, a["name"], a["status"], a["attributes"].get("Definition"), json.dumps(a["attributes"].get("Synonyms", [])),
         (a.get("responsibilities", {}).get("Owner") or [None])[0], f"{base_url}/term/{tid}")
        for tid, a in collibra["terms"].items()])
    links = []
    for kind, assets in (("data_category", collibra["data_categories"]), ("table", collibra["tables"]), ("term", collibra["terms"])):
        for entity_id, a in assets.items():
            for term in a.get("relations", {}).get("related to", []):
                links.append((term, kind, entity_id))
    conn.executemany("INSERT INTO term_links VALUES (?,?,?)", links)
    terms = {tid: [a["name"], *a["attributes"].get("Synonyms", [])] for tid, a in collibra["terms"].items()}
    conn.executemany("INSERT INTO data_category_search VALUES (?,?,?,?)", [
        (data_category_id, b["name"], b["definition"],
         " ".join(w for term, kind, e in links if kind == "data_category" and e == data_category_id for w in terms.get(term, [])))
        for data_category_id, b in data_categories.items()])

    table_fields = [r[1] for r in conn.execute("PRAGMA table_info(tables)")]
    conn.executemany(f"INSERT INTO tables VALUES ({','.join('?' * len(table_fields))})",
                     [tuple(t.get(f) for f in table_fields) for t in tables.values()])
    column_fields = [r[1] for r in conn.execute("PRAGMA table_info(columns)")]
    conn.executemany(f"INSERT INTO columns VALUES ({','.join('?' * len(column_fields))})", [
        tuple({**c, "table_fqn": fqn, "url": f"{base_url}/table/{fqn}#{c['name']}"}.get(f) for f in column_fields)
        for fqn, cols in columns.items() for c in cols])
    conn.executemany("INSERT INTO column_lineage VALUES (?,?,?,?,?)", lineage_rows)
    conn.executemany("INSERT INTO column_profiles VALUES (?,?,?,?,?,?,?,?,?)", [
        (fqn, col, p["row_count"], p["null_ratio"], p["distinct_count"], p["distinct_ratio"], p["avg_length"],
         p["avg_words"], json.dumps(p["pattern_matches"]))
        for fqn, w in warehouse.items() for col, p in w["profiles"].items()])
    conn.executemany("INSERT INTO lineage_edges VALUES (?,?,?)", edges)
    conn.executemany("INSERT INTO exposures VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (e["unique_id"], e["name"], e.get("label"), e["type"], e.get("maturity"), e.get("description"),
         e["owner"].get("name"), e["owner"].get("email"), email_user.get(e["owner"].get("email")), e.get("url"))
        for e in exposures])
    conn.executemany("INSERT INTO ownership VALUES (?,?,?,?,?,?)", ownership)
    conn.executemany("INSERT INTO support_channels VALUES (?,?,?,?,?)", support)
    conn.executemany("INSERT INTO slas VALUES (?,?,?,?)", slas)
    conn.executemany("INSERT INTO contract_quality VALUES (?,?,?,?)", quality)
    conn.executemany("INSERT INTO dbt_tests VALUES (?,?,?,?,?)", [
        (t["unique_id"], uid_fqn[t["attached_node"]], t.get("column_name"), t["test_metadata"]["name"],
         json.dumps(t["test_metadata"].get("kwargs", {}))) for t in tests])
    conn.executemany("INSERT INTO governance_gaps VALUES (?,?,?,?,?)", gaps)

    prov = []
    for fqn, t in tables.items():
        if t["in_collibra"]:
            prov.append(("table", fqn, "collibra", fqn))
        if fqn in contracts:
            prov.append(("table", fqn, "contract", str(contracts[fqn]["_path"].relative_to(ROOT))))
        if t["in_dbt"]:
            prov.append(("table", fqn, "dbt", t["dbt_unique_id"]))
        if t["in_warehouse"]:
            prov.append(("table", fqn, "warehouse", t["warehouse_table"]))
        for c in columns.get(fqn, []):
            if c.get("external_id"):
                prov.append(("column", f"{fqn}.{c['name']}", "collibra", c["external_id"]))
    prov += [("data_category", b, "collibra", b) for b in data_categories]
    prov += [("term", t, "collibra", t) for t in collibra["terms"]]
    prov += [("exposure", e["unique_id"], "dbt", e["unique_id"]) for e in exposures]
    conn.executemany("INSERT INTO provenance VALUES (?,?,?,?)", prov)
    conn.commit()
    conn.close()
    return {"tables": len(tables), "columns": sum(len(c) for c in columns.values()),
            "edges": len(edges), "gaps": len(gaps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if no source changed")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    db_path = ROOT / cfg["catalog"]["path"]
    hashes = [(s, p, sha256_file(p)) for s, p in source_files(cfg)]

    if db_path.exists() and not args.force:
        conn = sqlite3.connect(db_path)
        try:
            last = conn.execute("SELECT max(schema_version) FROM ingest_runs").fetchone()[0]
            previous = set(conn.execute("SELECT source, path, sha256 FROM ingest_runs"))
        except sqlite3.Error:
            last, previous = None, set()
        conn.close()
        current = {(s, str(p.relative_to(ROOT)), h) for s, p, h in hashes}
        if last == SCHEMA_VERSION and previous == current:
            print(f"{db_path.relative_to(ROOT)} is up to date ({len(hashes)} source files unchanged)")
            return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)
    run_id = hashlib.sha256("".join(h for _, _, h in hashes).encode()).hexdigest()[:12]
    stats = build(cfg, tmp_path, run_id, hashes)
    os.replace(tmp_path, db_path)
    print(f"wrote {db_path.relative_to(ROOT)} (run {run_id}): "
          f"{stats['tables']} tables, {stats['columns']} columns, "
          f"{stats['edges']} lineage edges, {stats['gaps']} governance gaps")


if __name__ == "__main__":
    main()

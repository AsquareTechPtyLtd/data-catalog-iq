"""Data category tools: search_data_categories and get_data_category. Shared catalog tools
(Data Category Recommender, and Catalog Lookup for data-category questions later).

A Data Category categorizes a whole table (data-model §4). These tools describe
the categories themselves; they never list which tables use a data category, so
blind evals cannot read an answer from them (filter_tables does that).

search_data_categories is retrieve-then-judge stage 1 (Data Category Recommender,
Scale): keyword search (SQLite FTS5, porter stemming, BM25) over each data
category's name, definition and linked glossary terms with their synonyms. When
the whole catalog fits in one result (at most `k` data categories) the ones that
did not match are listed too, after the matches, so nothing is missed; at scale
only the top `k` matches return.

Citation ids:
  collibra:DC-001     the data category asset: definition, status, retention, stewards
"""

import json
import re
import sqlite3

from catalog_iq.catalog.db import settings
from catalog_iq.tools.filter import DOMAINS
from catalog_iq.tools.tables import _Citations

K_DEFAULT = 20
K_MAX = 50
# weights for bm25(): data_category_id (unindexed), name, definition, terms
BM25_WEIGHTS = (0.0, 5.0, 1.0, 2.0)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "it", "of", "on", "or",
    "per", "the", "this", "to", "with", "which", "what", "data", "table", "tables", "one", "row",
}


def _domain(value: str | None) -> str | None:
    """'Supply Chain' → 'supply_chain', the table domain spelling."""
    return value.strip().lower().replace(" ", "_") if value else None


def fts_query(text: str) -> str | None:
    """Any-word FTS5 query from free text; quoted words, so FTS syntax in the text is inert."""
    words = [w for w in dict.fromkeys(re.findall(r"[a-z0-9]+", text.lower()))
             if len(w) > 1 and w not in STOPWORDS]
    return " OR ".join(f'"{w}"' for w in words) or None


def _retention(b: sqlite3.Row) -> dict | str:
    if b["retention_months"] is None:
        return "not defined"
    return {"months": b["retention_months"], "trigger": b["retention_trigger"]}


def _summary(b: sqlite3.Row, cites: _Citations) -> dict:
    return {"data_category_id": b["data_category_id"], "name": b["name"], "status": b["status"],
            "domain": _domain(b["domain"]),
            "definition": b["definition"], "retention": _retention(b),
            "sources": [cites.add("collibra", b["data_category_id"], b["url"])]}


def search_data_categories(conn: sqlite3.Connection, query: str, domain: str | None = None,
                           k: int = K_DEFAULT, data_category_id: str | None = None) -> dict:
    """Data categories whose name, definition or glossary terms match any word of `query`, best first.

    With `data_category_id`, only that one: whether it matches at all, at any rank.
    """
    k = max(1, min(int(k), K_MAX))
    only = data_category_id.strip().upper() if data_category_id else None
    cites = _Citations(settings(conn)["base_url"])
    rows = {r["data_category_id"]: r for r in conn.execute("SELECT * FROM data_categories ORDER BY data_category_id")
            if (domain is None or _domain(r["domain"]) == domain) and only in (None, r["data_category_id"])}
    q = fts_query(query)
    hits = [] if q is None else [r["data_category_id"] for r in conn.execute(
        f"SELECT data_category_id FROM data_category_search WHERE data_category_search MATCH ? "
        f"ORDER BY bm25(data_category_search, {', '.join(map(str, BM25_WEIGHTS))})",
        (q,)) if r["data_category_id"] in rows]
    out = [{**_summary(rows[b], cites), "rank": i + 1, "matched": True} for i, b in enumerate(hits[:k])]
    complete = len(rows) <= k
    if complete:
        out += [{**_summary(r, cites), "rank": None, "matched": False} for b, r in rows.items() if b not in hits]
    return {"query": query, "domain": domain, "status": "found" if hits else "no_match",
            "matched": len(hits), "complete": complete,
            "note": ("Every data category in scope is listed: matches first, then the rest." if complete else
                     f"Top {k} of {len(hits)} matches; search again with other words if none fits."),
            "data_categories": out, "citations": cites.as_list(compact=True)}


def get_data_category(conn: sqlite3.Connection, data_category_id: str) -> dict:
    """One data category in full: definition, status, retention, stewards, glossary terms."""
    b = conn.execute("SELECT * FROM data_categories WHERE data_category_id = ?",
                     (data_category_id.strip().upper(),)).fetchone()
    if b is None:
        return {"status": "not_found", "data_category_id": data_category_id,
                "hint": "Use search_data_categories to find a data category by its meaning."}
    cites = _Citations(settings(conn)["base_url"])
    stewards: dict[str, list] = {}
    for r in conn.execute("SELECT s.role, p.full_name, p.email FROM data_category_stewardship s "
                          "JOIN people p USING (username) "
                          "WHERE s.data_category_id = ? ORDER BY s.role, p.full_name", (b["data_category_id"],)):
        stewards.setdefault(r["role"], []).append({"name": r["full_name"], "email": r["email"]})
    retention = _retention(b)
    if isinstance(retention, dict):
        retention.update(disposal_action=b["disposal_action"], legal_basis=b["legal_basis"])
    terms = [{"term": t["name"], "synonyms": json.loads(t["synonyms"]), "definition": t["definition"]}
             for t in conn.execute("SELECT g.* FROM term_links l JOIN glossary_terms g USING (term_id) "
                                   "WHERE l.entity_type = 'data_category' AND l.entity_id = ? ORDER BY g.name",
                                   (b["data_category_id"],))]
    return {"status": "found", "data_category": {**_summary(b, cites), "retention": retention, "stewards": stewards,
                                                 "glossary_terms": terms},
            "citations": cites.as_list(compact=True)}


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Words describing the data, e.g. 'website page views sessions'"},
        "domain": {"type": "string", "enum": DOMAINS, "description": "Only data categories of this governance domain"},
        "k": {"type": "integer", "minimum": 1, "maximum": K_MAX, "default": K_DEFAULT},
        "data_category_id": {"type": "string", "description": "Only this data category (e.g. DC-005): does it match?"},
    },
    "required": ["query"],
}
GET_SCHEMA = {
    "type": "object",
    "properties": {"data_category_id": {"type": "string", "description": "e.g. DC-005"}},
    "required": ["data_category_id"],
}
DATA_CATEGORY_TOOLS = [
    {"name": "search_data_categories",
     "description": ("Search Data Categories by meaning: keyword match on name, definition and glossary terms, "
                     "best first. Each data category has its definition, status (Approved / Candidate), governance "
                     "domain and retention. When `complete` is true every data category in scope is listed (matches "
                     "first); otherwise search again with other words if none fits. Cite a data category with its "
                     "id."),
     "input_schema": SEARCH_SCHEMA, "function": search_data_categories},
    {"name": "get_data_category",
     "description": "One Data Category in full: definition, status, retention, stewards, glossary terms.",
     "input_schema": GET_SCHEMA, "function": get_data_category},
]

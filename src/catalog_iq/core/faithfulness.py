"""Faithfulness checks: does an answer say only what the tool results support?

Shared by the live output guardrail (core.guardrails) and the offline eval
(evals.scoring). Needs no answer key, only the answer, its cited facts and the
tool results, so it can run on every request.

Checks:
- invalid_citation: a fact cites an id no tool returned (in `citations` or
  in a `sources` list)
- unsupported_claim: a key value in a claim (person, data category id, table, number)
  is not in the tool-result block that the citation points to. A table or
  exposure that is an endpoint of one of the fact's relations, where the
  relation passes the edge check below, is supported by the graph instead
- answer_entity_not_in_facts: the answer names a person/table/data category/exposure
  that no fact states (so it was never cited). The answer's subject tables
  (its `tables` list) are exempt: they are checked separately against the
  tools. Tables and exposures named in the user's question are exempt too
  ("I can't drop dim_customers" repeats the user, it claims nothing).
- substitute_owner: a tool said a table has no owner, but someone is named
  as owner
- unsupported_relation: a fact's lineage link is not in the edges the tools
  returned (nodes' loaded_from, and filter_tables' `edges`; direct = an edge,
  indirect = a path of 2+ hops), or the fact does not cite the downstream
  node's lineage id
- relation_not_structured: a fact names 2+ tables/exposures with a lineage
  verb ("feeds", "built from", ...) but lists no `relations`

Known limit: only key values are checked, not free-text meaning ("is
certified" vs "is not certified"), and lineage links only where a fact lists
them in `relations`, not as worded in the answer; that needs an LLM judge
(deferred).
"""

import json
import re
import sqlite3
from collections import defaultdict

from catalog_iq.core.response import Fact

DATA_CATEGORY_RE = re.compile(r"\bDC-\d{3}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w.\-])\d+(?:\.\d+)?(?!\w)")
OWNER_RE = re.compile(r"\bown(?:s|ed|er|ers|ership)?\b", re.IGNORECASE)
NOT_OWNER_ROLE_RE = re.compile(r"\bsteward|\bdata owner\b|\bdata categor(?:y|ies)\b", re.IGNORECASE)
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # "2,000" → "2000"
LINEAGE_VERB_RE = re.compile(
    r"\b(?:feed(?:s|ing)?|fed|consum(?:e|es|ed|er|ers|ing)|loads?(?:ed)? (?:from|into)|built (?:from|on)|"
    r"depends? on|dependent on|upstream|downstream|flows?|flowed|derived from|comes? from|reads? from|"
    r"used by|uses|sources? (?:of|for)|path)\b", re.IGNORECASE)

Problem = tuple[str, str]  # (check, detail)


def _plain_numbers(text: str) -> str:
    return THOUSANDS_RE.sub("", text)


def _sentences(text: str) -> list[str]:
    # a "." ends a sentence only before a space or the end, so FQNs stay whole
    return [s for s in re.split(r"(?:[.;](?=\s|$)|\n)+", text) if s.strip()]


def walk(node):
    """Yield every dict inside a tool result."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


class Vocabulary:
    """Catalog entities that can be recognised in free text."""

    def __init__(self, conn: sqlite3.Connection):
        self.people = [r["full_name"] for r in conn.execute("SELECT full_name FROM people")]
        self.tables = []
        for r in conn.execute("SELECT fqn, schema_name, name FROM tables"):
            # bare names only when they contain "_": "customers" is also an ordinary word
            forms = [r["fqn"], f"{r['schema_name']}.{r['name']}"] + ([r["name"]] if "_" in r["name"] else [])
            pattern = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, forms)) + r")(?!\w)", re.IGNORECASE)
            self.tables.append((pattern, r["fqn"]))
        self.exposures = []
        for r in conn.execute("SELECT exposure_id, name, label FROM exposures"):
            forms = [r["exposure_id"], r["name"], r["label"]]
            pattern = re.compile(r"(?<![\w.])(?:" + "|".join(map(re.escape, forms)) + r")(?!\w)", re.IGNORECASE)
            self.exposures.append((pattern, r["exposure_id"]))

    def entities(self, text: str) -> set[tuple[str, str]]:
        found = {("data_category", m.upper()) for m in DATA_CATEGORY_RE.findall(text)}
        found |= {("person", p) for p in self.people if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE)}
        found |= {("table", fqn) for pattern, fqn in self.tables if pattern.search(text)}
        found |= {("exposure", eid) for pattern, eid in self.exposures if pattern.search(text)}
        return found

    def key_values(self, claim: str) -> set[tuple[str, str]]:
        # numbers inside entity names (DC-002, sha256) are not separate values
        stripped = _plain_numbers(DATA_CATEGORY_RE.sub(" ", claim))
        return self.entities(claim) | {("number", n) for n in NUMBER_RE.findall(stripped)}


def lineage_nodes(tool_results: list[dict]):
    """Yield every lineage node (table or exposure) in the tool results."""
    for result in tool_results:
        for node in walk(result):
            if node.get("type") in ("table", "exposure") and "loaded_from" in node:
                yield node


def node_id(node: dict) -> str:
    return node.get("fqn") or node["id"]


class Evidence:
    """For each citation id: the tool-result text that carries it, and the
    table it is about (a get_table result's table, or a lineage node)."""

    def __init__(self, tool_results: list[dict]):
        self.ids: set[str] = set()
        self.text: dict[str, list[str]] = defaultdict(list)
        self.tables: dict[str, set[str]] = defaultdict(set)
        for result in tool_results:
            table = result.get("table")
            table_fqn = table.get("fqn") if isinstance(table, dict) else None
            for c in result.get("citations", []):
                self.ids.add(c["id"])
                self.text[c["id"]].append(json.dumps(c))
            for node in walk(result):
                for cid in node.get("sources", []):
                    self.ids.add(cid)  # filter results list most ids only inline
                    self.text[cid].append(json.dumps(node))
                    if table_fqn:
                        self.tables[cid].add(table_fqn)
        # a node's owner/certification sections cite sources about that node
        for node in lineage_nodes(tool_results):
            for part in walk(node):
                for cid in part.get("sources", []):
                    self.tables[cid].add(node_id(node))
        self.edges: set[tuple[str, str]] = {tuple(e) for r in tool_results for e in r.get("edges", [])}
        self.lineage_cite: dict[str, set[str]] = defaultdict(set)
        for node in lineage_nodes(tool_results):
            self.edges.update((up, node_id(node)) for up in node["loaded_from"])
            self.lineage_cite[node_id(node)].update(node.get("sources", []))

    def reaches(self, start: str, end: str, min_hops: int = 1) -> bool:
        frontier, hops, seen = {start}, 0, set()
        while frontier:
            hops += 1
            frontier = {d for u, d in self.edges if u in frontier} - seen
            if end in frontier and hops >= min_hops:
                return True
            seen |= frontier
        return False

    def relation_problem(self, relation, citations: list[str]) -> str | None:
        link = f"{relation.upstream} -> {relation.downstream}"
        if relation.kind == "direct" and (relation.upstream, relation.downstream) not in self.edges:
            return f"{link} is not a direct link in the lineage results (check loaded_from)"
        if relation.kind == "indirect" and not self.reaches(relation.upstream, relation.downstream, min_hops=2):
            return (f"{link} is a direct link, not indirect" if (relation.upstream, relation.downstream) in self.edges
                    else f"{link} is not connected in the lineage results")
        if not self.lineage_cite[relation.downstream] & set(citations):
            ids = " or ".join(sorted(self.lineage_cite[relation.downstream])) or f"{relation.downstream}'s lineage id"
            return f"{link} must cite {ids}"
        return None

    def supports(self, cids: list[str], kind: str, value: str) -> bool:
        text = " ".join(t for cid in cids for t in self.text[cid])
        if kind == "table" and any(value in self.tables[cid] for cid in cids):
            return True
        if kind == "number":
            text = _plain_numbers(text)
            return re.search(rf"(?<![\w.]){re.escape(value)}(?![\w])", text) is not None
        return value.lower() in text.lower()


def check_faithfulness(answer: str, facts: list[Fact], tool_results: list[dict], vocab: Vocabulary,
                       subject_tables: list[str] = (), question: str = "") -> tuple[list[Problem], int]:
    """Returns the problems found and the number of unsupported claims."""
    evidence = Evidence(tool_results)
    problems: list[Problem] = []
    unsupported = 0

    for i, fact in enumerate(facts, 1):
        bad = [("invalid_citation", f"fact {i} cites {cid}") for cid in fact.citations if cid not in evidence.ids]
        relation_problems = [(r, evidence.relation_problem(r, fact.citations)) for r in fact.relations]
        # the endpoints of a link the lineage edges prove need no text match
        proven = {end for r, p in relation_problems if p is None for end in (r.upstream, r.downstream)}
        bad += [("unsupported_claim", f"fact {i} '{value}' not in cited source")
                for kind, value in sorted(vocab.key_values(fact.claim))
                if not (kind in ("table", "exposure") and value in proven)
                and not evidence.supports(fact.citations, kind, value)]
        bad += [("unsupported_relation", f"fact {i}: {p}") for r, p in relation_problems if p]
        nodes = {v for kind, v in vocab.entities(fact.claim) if kind in ("table", "exposure")}
        if len(nodes) >= 2 and LINEAGE_VERB_RE.search(fact.claim) and not fact.relations:
            bad.append(("relation_not_structured",
                        f"fact {i} states lineage between {', '.join(sorted(nodes))}; list each link in relations"))
        unsupported += bool(bad)
        problems += bad

    in_facts = vocab.entities(" ".join(f.claim for f in facts)) | {("table", t) for t in subject_tables}
    in_facts |= {e for e in vocab.entities(question) if e[0] in ("table", "exposure")}  # echoing the user
    problems += [("answer_entity_not_in_facts", value) for _, value in sorted(vocab.entities(answer) - in_facts)]

    # tables with no owner: from get_table (the answer's subject) and lineage nodes
    ownerless_subject = any(isinstance(r.get("owners"), dict) and r["owners"].get("status") == "none"
                            for r in tool_results)
    ownerless = {node_id(n) for n in lineage_nodes(tool_results) if n["owner"].get("status") == "none"}
    ownerless |= {r["table"]["fqn"] for r in tool_results if isinstance(r.get("owners"), dict)
                  and r["owners"].get("status") == "none" and isinstance(r.get("table"), dict)}
    if ownerless:
        for sentence in _sentences(" ".join([answer, *(f.claim for f in facts)])):
            if not OWNER_RE.search(sentence) or NOT_OWNER_ROLE_RE.search(sentence):
                continue
            entities = vocab.entities(sentence)
            named = {v for kind, v in entities if kind == "table"}
            # a sentence naming tables is about those tables; one naming none is about the subject
            if (named & ownerless) if named else ownerless_subject:
                problems += [("substitute_owner", value) for kind, value in sorted(entities) if kind == "person"]

    return problems, unsupported

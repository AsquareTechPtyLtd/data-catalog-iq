"""Lineage tools: get_lineage and trace_path.

Table-level dbt lineage (`lineage_edges`): table → table and table → exposure
(dashboard, ML model, app). No column-level lineage. Plain functions over the
read-only catalog, like tools.tables, with the same citation ids:

  dbt:model.fernwood.<name> / dbt:source.fernwood.<src>.<name>   the node and its
                                       depends_on list (covers loaded_from, layer)
  dbt:exposure.fernwood.<name>         exposure: type, owner, url, depends_on
  contract:… / catalog:<fqn>#ownership owner, same as get_table
  catalog:<fqn>#certification          certification, same as get_table

Every node carries `loaded_from` (all its direct upstreams, also those outside
the walk), so a table built from several sources shows all of them.
Staging/intermediate tables are included; each node says its layer. Links are
left out of nodes (code fills table links; exposure urls are in `citations`)
to keep big walks well under the tool guardrail's size limit.
"""

import sqlite3
from collections import defaultdict

from catalog_iq.catalog.db import settings
from catalog_iq.tools.tables import _Citations, _contract_cite, _escalation, _norm, _resolve

MAX_DEPTH = 10
MAX_PATHS = 10
DIRECTIONS = ("upstream", "downstream", "both")


class _Graph:
    def __init__(self, conn: sqlite3.Connection):
        self.up: dict[str, list[str]] = defaultdict(list)
        self.down: dict[str, list[str]] = defaultdict(list)
        for r in conn.execute("SELECT upstream_id, downstream_id FROM lineage_edges ORDER BY 1, 2"):
            self.up[r["downstream_id"]].append(r["upstream_id"])
            self.down[r["upstream_id"]].append(r["downstream_id"])

    def walk(self, root: str, direction: str, max_depth: int) -> tuple[dict[str, int], bool]:
        """Breadth-first walk: {node: depth}, and whether nodes lie beyond max_depth."""
        step = self.up if direction == "upstream" else self.down
        depths, frontier = {}, [root]
        for depth in range(1, max_depth + 1):
            frontier = sorted({n for f in frontier for n in step[f] if n != root and n not in depths})
            if not frontier:
                return depths, False
            depths.update((n, depth) for n in frontier)
        return depths, any(n not in depths and n != root for f in frontier for n in step[f])

    def paths(self, start: str, end: str, limit: int) -> list[list[str]]:
        """Up to `limit` paths from start to end, following edges downstream."""
        found: list[list[str]] = []

        def visit(path):
            if len(found) >= limit:
                return
            if path[-1] == end:
                found.append(path)
                return
            for n in self.down[path[-1]]:
                if n not in path:
                    visit(path + [n])

        visit([start])
        return found


def _resolve_node(conn: sqlite3.Connection, ref: str) -> tuple[str | None, dict | None]:
    """A table (FQN, schema.name or exact name) or an exposure (id or exact name)."""
    q = _norm(ref)
    row = conn.execute("SELECT exposure_id FROM exposures WHERE exposure_id = ? OR name = ?", (q, q)).fetchone()
    if row:
        return row["exposure_id"], None
    t, error = _resolve(conn, ref)
    if t is not None:
        return t["fqn"], None
    if error["status"] == "not_found":
        error["hint"] = ("No exact table or exposure. Use find_table to search (kind=\"exposure\" for "
                         "dashboards, reports, ML models, apps).")
    return None, error


class _Nodes:
    """Builds node dicts with citations, shared by both tools."""

    def __init__(self, conn: sqlite3.Connection, graph: _Graph):
        self.conn = conn
        self.graph = graph
        self.cfg = settings(conn)
        self.cites = _Citations(self.cfg["base_url"])

    def node(self, node_id: str, depth: int | None = None) -> dict:
        n = self._exposure(node_id) if node_id.startswith("exposure.") else self._table(node_id)
        if depth is not None:
            n["depth"] = depth
        n["loaded_from"] = sorted(self.graph.up[node_id])
        return n

    def _table(self, fqn: str) -> dict:
        t = self.conn.execute("SELECT * FROM tables WHERE fqn = ?", (fqn,)).fetchone()
        dbt = self.conn.execute("SELECT ref FROM provenance WHERE entity_type = 'table' AND entity_id = ? "
                                "AND source = 'dbt'", (fqn,)).fetchone()
        names = [r["full_name"] for r in self.conn.execute(
            "SELECT full_name FROM ownership WHERE table_fqn = ? AND role = 'Owner' ORDER BY full_name", (fqn,))]
        owner = {"status": t["owner_status"], "names": names}
        if t["owner_status"] == "contract":
            owner["sources"] = [_contract_cite(self.conn, self.cites, fqn)]
        elif t["owner_status"] == "inherited":
            owner["inherited_from"] = t["owner_inherited_from"]
            owner["sources"] = [_contract_cite(self.conn, self.cites, t["owner_inherited_from"])]
        else:
            owner["note"] = "No owner on record."
            owner["escalation"] = _escalation(self.cfg, self.cites)
            owner["sources"] = [self.cites.add("catalog", f"{fqn}#ownership")]
        return {"fqn": fqn, "type": "table", "layer": t["layer"], "status": t["status"],
                "owner": owner,
                "certification": {"status": t["certification"],
                                  "sources": [self.cites.add("catalog", f"{fqn}#certification")]},
                "sources": [self.cites.add("dbt", dbt["ref"])] if dbt else []}

    def _exposure(self, exposure_id: str) -> dict:
        e = self.conn.execute("SELECT * FROM exposures WHERE exposure_id = ?", (exposure_id,)).fetchone()
        cite = self.cites.add("dbt", exposure_id, e["url"])
        return {"id": exposure_id, "name": e["name"], "label": e["label"], "type": "exposure",
                "exposure_type": e["type"], "owner": {"names": [e["owner_name"]], "sources": [cite]},
                "sources": [cite]}


def get_lineage(conn: sqlite3.Connection, table: str, direction: str, depth: int | None = None) -> dict:
    """Upstream sources and/or downstream consumers of one table or exposure."""
    root, error = _resolve_node(conn, table)
    if error:
        return error
    graph = _Graph(conn)
    nodes = _Nodes(conn, graph)
    max_depth = depth or MAX_DEPTH
    result = {"status": "found", "direction": direction, "root": nodes.node(root)}
    more = {}
    for d in ("upstream", "downstream"):
        if direction not in (d, "both"):
            continue
        depths, more[d] = graph.walk(root, d, max_depth)
        result[d] = [nodes.node(n, k) for n, k in sorted(depths.items(), key=lambda x: (x[1], x[0]))]
        if not depths:  # citable, so "nothing consumes it" can be stated as a fact
            result[f"{d}_note"] = {"text": f"Nothing {d} of {root} in dbt lineage.",
                                   "sources": result["root"]["sources"]}
    if any(more.values()):
        result["stopped_at_depth"] = max_depth
        result["more_beyond"] = [d for d, m in more.items() if m]
    result["lineage_level"] = "table"
    result["citations"] = nodes.cites.as_list(compact=True)
    return result


def trace_path(conn: sqlite3.Connection, source: str, target: str) -> dict:
    """Every path (up to MAX_PATHS) from source down to target."""
    start, error = _resolve_node(conn, source)
    if error:
        return error
    end, error = _resolve_node(conn, target)
    if error:
        return error
    graph = _Graph(conn)
    nodes = _Nodes(conn, graph)
    paths = graph.paths(start, end, MAX_PATHS + 1)
    if not paths:  # the endpoints' lineage is the evidence for "no path"
        return {"status": "no_path", "from": start, "to": end,
                "reverse_path_exists": bool(graph.paths(end, start, 1)),
                "note": f"{end} is not downstream of {start} in dbt lineage.",
                "nodes": [nodes.node(start), nodes.node(end)], "lineage_level": "table",
                "citations": nodes.cites.as_list(compact=True)}
    truncated = len(paths) > MAX_PATHS
    paths = sorted(paths[:MAX_PATHS], key=lambda p: (len(p), p))
    on_paths = sorted({n for p in paths for n in p})
    result = {"status": "found", "from": start, "to": end, "paths": paths,
              "nodes": [nodes.node(n) for n in on_paths], "lineage_level": "table"}
    if truncated:
        result["paths_truncated"] = True
    result["citations"] = nodes.cites.as_list(compact=True)
    return result


# --- Tool definitions (provider-neutral) -----------------------------------------

LINEAGE_TOOLS = [
    {
        "name": "get_lineage",
        "description": (
            "Table-level dbt lineage for one table or exposure. direction=upstream lists what it is "
            "built from (sources), downstream lists what consumes it (tables and exposures: "
            "dashboards, ML models, apps), both lists both. Each node has its depth (hops from the "
            "root), layer, status, owner, certification and `loaded_from` (all its direct upstreams). "
            "Includes staging and intermediate tables. Accepts an exact table name or FQN, or an "
            "exposure id/name from find_table(kind=\"exposure\"). If `more_beyond` is set, the walk "
            "stopped at the depth limit and the list is incomplete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "e.g. fernwood_dw.marts.dim_customers or "
                                                         "exposure.fernwood.loyalty_dashboard"},
                "direction": {"type": "string", "enum": list(DIRECTIONS)},
                "depth": {"type": "integer", "minimum": 1, "maximum": MAX_DEPTH,
                          "description": "Max hops; omit for the full walk"},
            },
            "required": ["table", "direction"],
        },
        "function": get_lineage,
    },
    {
        "name": "trace_path",
        "description": (
            "Every lineage path (up to 10) from one table down to another table or exposure, each "
            "as a list of node ids in order, plus details for every node on a path. Use it for "
            "'how does A flow into B' questions. status no_path says if B is instead upstream of A."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Upstream table (exact name or FQN)"},
                "target": {"type": "string", "description": "Downstream table or exposure id/name"},
            },
            "required": ["source", "target"],
        },
        "function": trace_path,
    },
]

"""Retrieval-only eval for search_data_categories (no LLM): BM25 vs hybrid vs hybrid + re-rank.

    python -m catalog_iq.evals.retrieval              # every query set, both catalogs, every retriever
    python -m catalog_iq.evals.retrieval --retrievers bm25 hybrid

Query sets, each with the true data category from an answer key:
  tables      the answer-key tables (open, dev, heldout): their name, description and column
              names, the query get_data_category_evidence builds (table_text)
  agent       every search_data_categories call the agent made in the newest saved Data
              Category Recommender run on that catalog, replayed with the agent's own words
  paraphrase  invented tables described without their data category's words
              (evals/data_category_recommender/retrieval_paraphrase.yaml)

Catalogs: the demo catalog (13 data categories, every search lists all of them) and the
scale catalog (5,013, data/catalog/catalog_scale.db). Rank is the true data category's
position in the list the agent would see (k = 50); not in the list counts as a miss.
Metrics: recall@20 and @3, top-1, MRR (1/rank, 0 for a miss), median ms per search.

Writes evals/retrieval/runs/<timestamp>/{results.jsonl, summary.json}.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from catalog_iq.catalog.db import ROOT, catalog_path, connect
from catalog_iq.dashboard import latest_runs
from catalog_iq.evals.answer_key import EVALS_DIR
from catalog_iq.evals.data_category import load_key
from catalog_iq.tools.data_categories import K_MAX, search_data_categories
from catalog_iq.tools.data_category_evidence import table_text
from catalog_iq.tools.retrieval import RETRIEVERS

PARAPHRASE = EVALS_DIR / "data_category_recommender" / "retrieval_paraphrase.yaml"
SCALE = ROOT / "data" / "catalog" / "catalog_scale.db"
SUITE = "data_category_recommender"


def _columns(conn, fqn: str) -> list[dict]:
    return [{"column": r[0]} for r in conn.execute("SELECT name FROM columns WHERE table_fqn = ? ORDER BY ordinal",
                                                   (fqn,))]


def table_queries(conn) -> list[dict]:
    out = []
    for key in load_key():
        t = conn.execute("SELECT name, description FROM tables WHERE fqn = ?", (key.table,)).fetchone()
        out.append({"set": f"tables:{key.set}", "id": key.table, "truth": key.data_category,
                    "query": table_text(t[0], t[1], _columns(conn, key.table))})
    return out


def paraphrase_queries() -> list[dict]:
    tables = yaml.safe_load(PARAPHRASE.read_text())["tables"]
    return [{"set": "paraphrase", "id": t["name"], "truth": t["data_category"],
             "query": table_text(t["name"], t["description"], [{"column": c} for c in t["columns"]])}
            for t in tables]


def agent_queries(catalog: str) -> list[dict]:
    """The agent's own searches in the newest saved run on this catalog ('default' or 'scale')."""
    truth = {k.table: k.data_category for k in load_key()}
    runs = [run_dir for run_dir, s in latest_runs(SUITE)
            if ("scale" if s.get("catalog", "default") != "default" else "default") == catalog]
    out = []
    for run_dir in runs[:1]:
        for row in map(json.loads, (run_dir / "results.jsonl").read_text().splitlines()):
            if "error" in row:
                continue
            calls = [e for e in row["response"]["evidence"] if e["tool"] == "search_data_categories"
                     and e["result_status"] != "blocked"]
            for e in calls:
                out.append({"set": "agent", "id": f"{row['table']}#{row['run']}", "truth": truth[row["table"]],
                            "query": e["args"]["query"], "domain": e["args"].get("domain"), "run": run_dir.name})
    return out


def rank(conn, q: dict, retriever: str) -> tuple[int | None, float]:
    start = time.perf_counter()
    r = search_data_categories(conn, q["query"], domain=q.get("domain"), k=K_MAX, retriever=retriever)
    ms = (time.perf_counter() - start) * 1000
    ids = [b["data_category_id"] for b in r["data_categories"]]
    return (ids.index(q["truth"]) + 1 if q["truth"] in ids else None), ms


def summarize(rows: list[dict]) -> dict:
    ranks = [r["rank"] for r in rows]
    n = len(rows)
    return {"queries": n,
            "recall_at_20": round(sum(1 for x in ranks if x and x <= 20) / n, 3),
            "recall_at_3": round(sum(1 for x in ranks if x and x <= 3) / n, 3),
            "top_1": round(sum(1 for x in ranks if x == 1) / n, 3),
            "mrr": round(sum(1 / x for x in ranks if x) / n, 3),
            "median_ms": round(statistics.median(r["ms"] for r in rows), 1)}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS))
    args = parser.parse_args(argv)
    out_dir = EVALS_DIR / "retrieval" / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    results, summary = [], {"retrievers": args.retrievers, "by": {}}
    for catalog, path in (("default", catalog_path()), ("scale", SCALE)):
        if not path.exists():
            continue
        conn = connect(path)
        queries = table_queries(conn) + agent_queries(catalog) + paraphrase_queries()
        for retriever in args.retrievers:
            search_data_categories(conn, "warm up", retriever=retriever)  # load models before timing
            for q in queries:
                r, ms = rank(conn, q, retriever)
                results.append({**q, "catalog": catalog, "retriever": retriever, "rank": r, "ms": round(ms, 2)})
        conn.close()
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        family = r["set"].split(":")[0]
        groups.setdefault((r["catalog"], family, r["retriever"]), []).append(r)
    for (catalog, family, retriever), rows in sorted(groups.items()):
        summary["by"].setdefault(catalog, {}).setdefault(family, {})[retriever] = summarize(rows)
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for catalog, families in summary["by"].items():
        for family, by_retriever in families.items():
            print(f"\n{catalog} catalog, {family}")
            for retriever, s in by_retriever.items():
                print(f"  {retriever:14s} n={s['queries']:3d}  recall@20 {s['recall_at_20']:.3f}  "
                      f"recall@3 {s['recall_at_3']:.3f}  top-1 {s['top_1']:.3f}  MRR {s['mrr']:.3f}  "
                      f"{s['median_ms']} ms")
    print(f"\nwrote {out_dir.relative_to(ROOT)}")
    return out_dir


if __name__ == "__main__":
    main()

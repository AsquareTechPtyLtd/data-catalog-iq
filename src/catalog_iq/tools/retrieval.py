"""Hybrid retrieval for search_data_categories: BM25 and embeddings, fused, then re-ranked.

    RETRIEVER=bm25            keyword search only (SQLite FTS5, BM25): the default
    RETRIEVER=hybrid          + embedding similarity, fused with Reciprocal Rank Fusion
    RETRIEVER=hybrid+rerank   + a cross-encoder re-ranks the fused candidates

A start-up option (an environment variable), never a tool argument. Both models run
locally on CPU (fastembed, ONNX), so no other vendor sees catalog text and results
are the same every run. Optional extra: pip install -e ".[hybrid]"; the models download
once into data/models/.

Vectors are built at ingestion (scripts/ingest_catalog.py, build_scale_catalog.py)
into data_category_vectors and published with the catalog, so they pass the same
publish checks and roll back with it. The search embeds only the query.

What stays keyword-only: the evidence's definition signals and the guard's check of
the chosen data category. They mean "the table's words match the definition", and
feed the confidence level; an embedding search always returns something, so it would
lend every table definition support.
"""

import os
import sqlite3
from functools import cache

from catalog_iq.catalog.db import ROOT

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")  # an ablation can swap it
MODELS_DIR = ROOT / "data" / "models"
RETRIEVERS = ("bm25", "hybrid", "hybrid+rerank")
CANDIDATES = 50  # from each of BM25 and embeddings; the re-ranker sees the fused top CANDIDATES
RRF_K = 60


def retriever() -> str:
    name = os.environ.get("RETRIEVER", "bm25").strip().lower()
    if name not in RETRIEVERS:
        raise ValueError(f"RETRIEVER must be one of {', '.join(RETRIEVERS)}, not {name!r}")
    return name


def category_text(name: str, definition: str | None, terms: str | None) -> str:
    """What a data category is about: the same text BM25 indexes."""
    return ". ".join(p for p in (name, definition, terms) if p)


@cache
def _embedder():
    from fastembed import TextEmbedding  # optional extra: pip install -e ".[hybrid]"
    return TextEmbedding(EMBED_MODEL, cache_dir=str(MODELS_DIR))


@cache
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODELS_DIR))


def embed_passages(texts: list[str]):
    import numpy as np
    vectors = np.array(list(_embedder().passage_embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def write_vectors(conn: sqlite3.Connection) -> int:
    """Embed every data category into data_category_vectors (at ingestion). Returns how many."""
    rows = conn.execute("SELECT data_category_id, name, definition, terms FROM data_category_search "
                        "ORDER BY data_category_id").fetchall()
    vectors = embed_passages([category_text(r[1], r[2], r[3]) for r in rows])
    conn.execute("CREATE TABLE IF NOT EXISTS data_category_vectors "
                 "(data_category_id TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)")
    conn.execute("DELETE FROM data_category_vectors")
    conn.executemany("INSERT INTO data_category_vectors VALUES (?,?,?)",
                     [(r[0], EMBED_MODEL, v.tobytes()) for r, v in zip(rows, vectors)])
    return len(rows)


class _Index:
    """One catalog's vectors in memory: a matrix and the ids in its row order."""

    def __init__(self, conn: sqlite3.Connection):
        import numpy as np
        try:
            rows = conn.execute("SELECT data_category_id, model, vector FROM data_category_vectors "
                                "ORDER BY data_category_id").fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows or {r[1] for r in rows} != {EMBED_MODEL}:
            raise RuntimeError("RETRIEVER=hybrid needs this catalog's data category vectors from "
                               f"{EMBED_MODEL}: install .[hybrid] and run ingestion again")
        self.ids = [r[0] for r in rows]
        self.matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])


_indexes: dict[str, _Index] = {}


def _index(conn: sqlite3.Connection) -> _Index:
    key = conn.execute("PRAGMA database_list").fetchone()[2]  # the catalog file: one index per catalog
    if key not in _indexes:
        _indexes[key] = _Index(conn)
    return _indexes[key]


def vector_ranking(conn: sqlite3.Connection, query: str, in_scope: set[str]) -> list[str]:
    """The CANDIDATES data categories in scope closest to the query, best first."""
    import numpy as np
    index = _index(conn)
    q = np.array(next(iter(_embedder().query_embed([query]))), dtype=np.float32)
    scores = index.matrix @ (q / np.linalg.norm(q))
    order = np.argsort(-scores)
    return [index.ids[i] for i in order if index.ids[i] in in_scope][:CANDIDATES]


def fuse(*rankings: list[str]) -> list[str]:
    """Reciprocal Rank Fusion: each list adds 1 / (RRF_K + rank) to an id; ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda item: -scores[item])


def rerank(query: str, ids: list[str], texts: dict[str, str]) -> list[str]:
    """The cross-encoder's order for (query, data category text) pairs."""
    if not ids:
        return ids
    scores = list(_reranker().rerank(query, [texts[i] for i in ids]))
    return [i for _, i in sorted(zip(scores, ids), key=lambda p: -p[0])]


def hybrid(conn: sqlite3.Connection, query: str, keyword: list[str], rows: dict[str, sqlite3.Row],
           mode: str) -> tuple[list[str], dict[str, list[str]]]:
    """The fused (and re-ranked) ids, and which search found each: bm25, vector or both."""
    keyword = keyword[:CANDIDATES]
    vector = vector_ranking(conn, query, set(rows))
    ids = fuse(keyword, vector)[:CANDIDATES]
    if mode == "hybrid+rerank":
        texts = {r[0]: category_text(r[1], r[2], r[3]) for r in conn.execute(
            f"SELECT data_category_id, name, definition, terms FROM data_category_search WHERE data_category_id IN "
            f"({','.join('?' * len(ids))})", ids)}
        ids = rerank(query, ids, texts)
    found_by = {i: [name for name, ranking in (("bm25", keyword), ("vector", vector)) if i in ranking] for i in ids}
    return ids, found_by

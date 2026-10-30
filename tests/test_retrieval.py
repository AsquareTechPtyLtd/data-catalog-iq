"""Tests for hybrid retrieval (tools/retrieval.py). The model tests need the .[hybrid] extra and the
models already in data/models/ (ingestion downloads them), so `pytest` stays offline."""

import importlib.util
import shutil
import sqlite3

import pytest

from catalog_iq.catalog.db import catalog_path, connect
from catalog_iq.tools import retrieval
from catalog_iq.tools.data_categories import search_data_categories
from catalog_iq.tools.data_category_evidence import get_data_category_evidence

pytestmark = pytest.mark.skipif(not catalog_path().exists(),
                                reason="catalog.db missing: run python3 scripts/ingest_catalog.py")
models = pytest.mark.skipif(importlib.util.find_spec("fastembed") is None or not retrieval.MODELS_DIR.exists(),
                            reason="hybrid retrieval needs .[hybrid] and its models (run ingestion once)")

PAY = "remuneration What each staff member earns: base wage, bonus and allowances. staff ref base wage bonus allowance"


@pytest.fixture(scope="module")
def conn():
    c = connect()
    yield c
    c.close()


def test_fuse_adds_reciprocal_ranks():
    assert retrieval.fuse(["a", "b", "c"], ["c", "a"]) == ["a", "c", "b"]
    assert retrieval.fuse(["a"], []) == ["a"]


def test_retriever_setting(monkeypatch):
    assert retrieval.retriever() == "bm25"
    monkeypatch.setenv("RETRIEVER", "Hybrid")
    assert retrieval.retriever() == "hybrid"
    monkeypatch.setenv("RETRIEVER", "vectors")
    with pytest.raises(ValueError, match="RETRIEVER must be one of"):
        retrieval.retriever()


def test_bm25_results_are_unchanged(conn):
    r = search_data_categories(conn, "website page views sessions")
    assert all("found_by" not in b for b in r["data_categories"])
    assert r["data_categories"][0]["data_category_id"] == "DC-005"


@models
def test_hybrid_finds_a_category_described_in_other_words(conn):
    keyword = search_data_categories(conn, PAY, retriever="bm25")
    assert keyword["matched"] == 0  # no shared word
    r = search_data_categories(conn, PAY, retriever="hybrid")
    top = r["data_categories"][0]
    assert top["data_category_id"] == "DC-011" and top["found_by"] == ["vector"]


@models
def test_definition_checks_stay_keyword_only(conn, monkeypatch):
    """They feed confidence; embeddings would lend every table definition support."""
    before = get_data_category_evidence(conn, "raw_web.page_views", blind=True)
    monkeypatch.setenv("RETRIEVER", "hybrid")
    assert get_data_category_evidence(conn, "raw_web.page_views", blind=True) == before
    assert search_data_categories(conn, PAY, data_category_id="DC-011")["matched"] == 0


@models
def test_every_data_category_has_a_vector(conn):
    vectors = conn.execute("SELECT count(*), min(model), max(model) FROM data_category_vectors").fetchone()
    categories = conn.execute("SELECT count(*) FROM data_categories").fetchone()[0]
    assert tuple(vectors) == (categories, retrieval.EMBED_MODEL, retrieval.EMBED_MODEL)


@models
def test_hybrid_without_vectors_says_so(tmp_path):
    path = tmp_path / "catalog.db"
    shutil.copyfile(catalog_path(), path)
    with sqlite3.connect(path) as c:
        c.execute("DROP TABLE data_category_vectors")
    conn = connect(path)
    with pytest.raises(RuntimeError, match="needs this catalog's data category vectors"):
        search_data_categories(conn, PAY, retriever="hybrid")
    conn.close()

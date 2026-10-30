"""Tests for publishing a catalog build: gates, drift, rollback and the run log (freshness design)."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ingest_catalog", Path(__file__).resolve().parents[1] / "scripts" / "ingest_catalog.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)

A, B, C = "dw.raw.a", "dw.raw.b", "dw.marts.c"


def make_catalog(path: Path, tables: dict, run_id: str = "run0", people: int = 1) -> Path:
    """tables: fqn → (data_category_id, [column names])."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE ingest_runs (run_id TEXT, ingested_at TEXT, schema_version INTEGER,
                                  source TEXT, path TEXT, sha256 TEXT);
        CREATE TABLE tables (fqn TEXT, data_category_id TEXT);
        CREATE TABLE columns (table_fqn TEXT, name TEXT);
        CREATE TABLE people (username TEXT);
        CREATE TABLE data_categories (data_category_id TEXT);
    """)
    conn.execute("INSERT INTO ingest_runs VALUES (?, '2026-10-29', ?, 'collibra', 'src.json', 'h')",
                 (run_id, ingest.SCHEMA_VERSION))
    conn.executemany("INSERT INTO tables VALUES (?, ?)", [(f, dc) for f, (dc, _) in tables.items()])
    conn.executemany("INSERT INTO columns VALUES (?, ?)",
                     [(f, c) for f, (_, cols) in tables.items() for c in cols])
    conn.executemany("INSERT INTO people VALUES (?)", [(f"p{i}",) for i in range(people)])
    conn.execute("INSERT INTO data_categories VALUES ('DC-001')")
    conn.commit()
    conn.close()
    return path


def run_id_of(path: Path) -> str:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT run_id FROM ingest_runs").fetchone()[0]


ABC = {A: ("DC-001", ["id"]), B: ("DC-001", ["id"]), C: ("DC-008", ["id", "email"])}


# --- gates


def test_gates_pass(tmp_path):
    build = make_catalog(tmp_path / "build.db", ABC)
    ingest.check_gates(build, tmp_path / "none.db", {A, B, C})


def test_gate_empty_core_table(tmp_path):
    build = make_catalog(tmp_path / "build.db", ABC, people=0)
    with pytest.raises(ingest.GateFailed, match="people is empty"):
        ingest.check_gates(build, tmp_path / "none.db", {A, B, C})


def test_gate_missing_dbt_node(tmp_path):
    build = make_catalog(tmp_path / "build.db", {A: ABC[A]})
    with pytest.raises(ingest.GateFailed, match="2 dbt sources/models missing.*dw.marts.c"):
        ingest.check_gates(build, tmp_path / "none.db", {A, B, C})


def test_gate_table_drop(tmp_path, monkeypatch):
    published = make_catalog(tmp_path / "catalog.db", ABC)
    build = make_catalog(tmp_path / "build.db", {A: ABC[A]})
    with pytest.raises(ingest.GateFailed, match="tables dropped from 3 to 1"):
        ingest.check_gates(build, published, {A})
    ingest.check_gates(build, published, {A}, force=True)
    monkeypatch.setenv("INGEST_MAX_TABLE_DROP", "0.9")
    ingest.check_gates(build, published, {A})


def test_gate_corrupt_build(tmp_path):
    build = tmp_path / "build.db"
    build.write_bytes(b"not a database" * 100)
    with pytest.raises(ingest.GateFailed, match="not a readable catalog"):
        ingest.check_gates(build, tmp_path / "none.db", set())


# --- drift


def test_drift(tmp_path):
    old = make_catalog(tmp_path / "old.db", ABC)
    new = make_catalog(tmp_path / "new.db", {
        A: ("DC-001", ["id", "phone"]),        # column added
        C: ("DC-003", ["id"]),                 # column removed, data category changed
        "dw.raw.d": ("DC-001", ["id"]),        # table added; B removed
    })
    assert ingest.drift(old, new) == {
        "tables_added": ["dw.raw.d"],
        "tables_removed": [B],
        "columns_added": {A: ["phone"]},
        "columns_removed": {C: ["email"]},
        "data_category_changed": {C: ["DC-008", "DC-003"]},
    }
    assert ingest.drift(old, old) == {}
    assert ingest.drift(tmp_path / "none.db", new) == {}


# --- publish and rollback


def test_publish_keeps_previous_and_rollback_swaps(tmp_path):
    db = make_catalog(tmp_path / "catalog.db", ABC, run_id="old")
    ingest.publish(make_catalog(tmp_path / "build.db", ABC, run_id="new"), db)
    assert run_id_of(db) == "new" and run_id_of(ingest.prev_path(db)) == "old"
    assert not (tmp_path / "build.db").exists()

    ingest.rollback(db)
    assert run_id_of(db) == "old" and run_id_of(ingest.prev_path(db)) == "new"
    ingest.rollback(db)  # a second rollback rolls forward
    assert run_id_of(db) == "new"


def test_rollback_without_previous(tmp_path):
    db = make_catalog(tmp_path / "catalog.db", ABC)
    with pytest.raises(ingest.GateFailed, match="no previous catalog"):
        ingest.rollback(db)


# --- one attempt end to end, with the build stubbed


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo root with one source file; `build` writes whatever `state['tables']` says."""
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    source = tmp_path / "src.json"
    source.write_text("v1")
    state = {"tables": ABC, "error": None}
    monkeypatch.setattr(ingest, "source_files", lambda cfg: [("collibra", source)])
    monkeypatch.setattr(ingest, "manifest_fqns", lambda cfg: set(state["tables"]))

    def build(cfg, path, run_id, hashes):
        if state["error"]:
            raise state["error"]
        make_catalog(path, state["tables"], run_id=run_id)
        conn = sqlite3.connect(path)
        conn.execute("UPDATE ingest_runs SET sha256 = ?", (hashes[0][2],))
        conn.commit()
        conn.close()
        return {"tables": len(state["tables"]), "columns": 0, "edges": 0, "gaps": 0}

    monkeypatch.setattr(ingest, "build", build)
    return {"db": tmp_path / "catalog" / "catalog.db", "source": source, "state": state}


def log_lines(db: Path) -> list[dict]:
    return [json.loads(line) for line in ingest.log_path(db).read_text().splitlines()]


def test_ingest_publishes_then_skips_unchanged(repo):
    db = repo["db"]
    first = ingest.ingest({}, db)
    assert first["status"] == "published" and first["drift"] == {} and db.exists()
    second = ingest.ingest({}, db)
    assert second["status"] == "unchanged" and second["run_id"] == first["run_id"]
    assert [e["status"] for e in log_lines(db)] == ["published", "unchanged"]


def test_ingest_records_drift(repo):
    db = repo["db"]
    ingest.ingest({}, db)
    repo["source"].write_text("v2")
    repo["state"]["tables"] = {**ABC, "dw.raw.d": ("DC-001", ["id"])}
    entry = ingest.ingest({}, db)
    assert entry["status"] == "published" and entry["drift"] == {"tables_added": ["dw.raw.d"]}
    assert ingest.prev_path(db).exists()


def test_failed_gate_keeps_old_catalog(repo, monkeypatch):
    db = repo["db"]
    good = ingest.ingest({}, db)["run_id"]
    repo["source"].write_text("v2")
    repo["state"]["tables"] = {A: ABC[A]}
    monkeypatch.setattr(ingest, "manifest_fqns", lambda cfg: {A})
    entry = ingest.ingest({}, db)
    assert entry["status"] == "failed" and "tables dropped from 3 to 1" in entry["error"]
    assert run_id_of(db) == good
    assert not db.with_suffix(".db.tmp").exists()
    assert log_lines(db)[-1]["status"] == "failed"


def test_build_error_is_logged_and_raised(repo):
    db = repo["db"]
    repo["state"]["error"] = KeyError("collibra")
    with pytest.raises(KeyError):
        ingest.ingest({}, db)
    last = log_lines(db)[-1]
    assert last["status"] == "failed" and last["error"].startswith("KeyError") and not db.exists()

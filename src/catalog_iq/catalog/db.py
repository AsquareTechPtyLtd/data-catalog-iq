"""Read-only access to catalog.db, built by scripts/ingest_catalog.py."""

import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "sources.yaml"


def catalog_path() -> Path:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return ROOT / cfg["catalog"]["path"]


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the catalog read-only; any write raises sqlite3.OperationalError."""
    path = path or catalog_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run python3 scripts/ingest_catalog.py")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def settings(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings")}

"""How fresh the catalog is: which build is serving, how old its sources are, and whether the
last ingestion attempt failed (scripts/ingest_catalog.py writes ingest_log.jsonl beside the
catalog). Read from the request's own connection, so a new catalog shows up without a restart.

This is the metadata's age, not whether a table's data meets its contract SLA."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from catalog_iq.catalog.db import settings
from catalog_iq.core.config import number_setting

MAX_AGE_HOURS = 168  # CATALOG_MAX_AGE_HOURS
SOURCE_DATES = {"collibra": "collibra_exported_at", "dbt": "dbt_generated_at"}  # contracts, warehouse: none


class FailedIngest(BaseModel):
    finished_at: str
    error: str


class CatalogInfo(BaseModel):
    version: str = Field(description="The published build's run id.")
    built_at: str = Field(description="When this catalog was built.")
    as_of: str = Field(description="The oldest source export date: an answer is only as fresh as its oldest "
                                   "input. Sources with no export date count as built_at.")
    sources: dict[str, str | None] = Field(description="Export date per source that carries one.")
    age_days: float
    stale: bool = Field(description="as_of is older than max_age_hours. Answers are still given.")
    max_age_hours: float
    failed_ingest: FailedIngest | None = Field(
        description="The latest ingestion attempt failed; this catalog is the last one that passed.")


def catalog_version(conn: sqlite3.Connection) -> str:
    return conn.execute("SELECT run_id FROM ingest_runs LIMIT 1").fetchone()[0]


def last_failure(db_file: Path) -> FailedIngest | None:
    log = db_file.with_name("ingest_log.jsonl")
    lines = log.read_text().splitlines() if log.exists() else []
    if not lines:
        return None
    last = json.loads(lines[-1])
    return FailedIngest(finished_at=last["finished_at"], error=last["error"]) if last["status"] == "failed" else None


def catalog_info(conn: sqlite3.Connection, now: datetime | None = None) -> CatalogInfo:
    run_id, built_at = conn.execute("SELECT run_id, ingested_at FROM ingest_runs LIMIT 1").fetchone()
    s = settings(conn)
    sources = {name: s.get(key) for name, key in SOURCE_DATES.items()}
    as_of = min(datetime.fromisoformat(t) for t in (built_at, *filter(None, sources.values())))
    max_age = number_setting("CATALOG_MAX_AGE_HOURS", MAX_AGE_HOURS)
    age_h = ((now or datetime.now(timezone.utc)) - as_of).total_seconds() / 3600
    db_file = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    return CatalogInfo(version=run_id, built_at=built_at, as_of=as_of.isoformat(), sources=sources,
                       age_days=round(age_h / 24, 1), stale=age_h > max_age, max_age_hours=max_age,
                       failed_ingest=last_failure(db_file))

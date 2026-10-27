"""Agent config (config/agents.yaml) and secrets (.env, gitignored)."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
AGENTS_CONFIG = ROOT / "config" / "agents.yaml"
ENV_PATH = ROOT / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    """Set KEY=value lines from .env; variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def agent_config(agent: str, path: Path = AGENTS_CONFIG) -> dict:
    return yaml.safe_load(path.read_text())[agent]

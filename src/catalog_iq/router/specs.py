"""Route specs (router R8): one route.yaml next to each agent.

The router's system prompt is built from them, and its code checks read
`needs`, so the prompt and the checks cannot drift apart. Adding an agent is
one spec plus an entry in RUNNERS.
"""

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"
AGENTS = ("catalog_lookup", "pi_classifier", "data_category_recommender")


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    label: str
    purpose: str
    handles: list[str] = Field(min_length=1)
    not_for: list[str] = Field(default_factory=list)
    needs: dict[str, str]
    examples: list[str] = Field(min_length=1)

    @property
    def needs_table(self) -> bool:
        return self.needs.get("table") == "required"


@cache
def load_specs() -> dict[str, RouteSpec]:
    return {a: RouteSpec(agent=a, **yaml.safe_load((AGENTS_DIR / a / "route.yaml").read_text())) for a in AGENTS}


def specs_prompt(specs: dict[str, RouteSpec]) -> str:
    """The agents section of the router's system prompt."""
    blocks = []
    for s in specs.values():
        lines = [f"## {s.agent} ({s.label})", s.purpose, "Handles:"]
        lines += [f"- {h}" for h in s.handles]
        if s.not_for:
            lines += ["Not for:"] + [f"- {n}" for n in s.not_for]
        lines += [f"Needs a table: {'yes' if s.needs_table else 'no'}", "Examples:"]
        lines += [f"- {e}" for e in s.examples]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

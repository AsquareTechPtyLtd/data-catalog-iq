"""What a confidence level rests on, in plain reasons.

Every level is computed by code from what the tools returned, never from the model's own
certainty. It measures how well the catalog evidence supports an answer and how ambiguous it
was, not whether the answer is right; the reasons say which evidence counted.
"""

CONFIDENCE_ORDER = ["low", "medium", "high"]

MEANING = ("How well the catalog evidence supports this, computed by code (not the model's own "
           "certainty). Not a guarantee that it is correct; confidence_reasons says what it rests on.")

RETRIED = "Needed a second attempt, so at most medium"


def for_against(target: str | None, signals: list[dict]) -> tuple[list[dict], list[dict]]:
    """Signals that suggest `target`, and strong signals that suggest something else."""
    support = [s for s in signals if target in s["suggests"]]
    contra = [s for s in signals if s["strong"] and target not in s["suggests"]]
    return support, contra


def signal_reasons(support: list[dict], contra: list[dict], retried: bool, none_for: str) -> list[str]:
    reasons = [f"For: {s['detail']}" for s in support] or [none_for]
    reasons += [f"Against: {s['detail']}" for s in contra]
    return reasons + ([RETRIED] if retried else [])

"""PI Classifier guardrails: the shared tool checks, its own output checks.

Tool guardrail: inherited from core.guardrails.Guard (allow-list, args,
budget, result size).

Output guardrail (D9), on every draft:
- table_not_returned: the draft's table is not one get_columns returned
- not_applicable: recommendations for a stg_/int_ table (PI is inherited)
- incomplete: an open column has no recommendation
- not_open: a recommendation for a reviewed or unknown column
- duplicate: a column recommended twice
- uncited_column: a recommendation does not cite the column's own id
- invalid_citation / unsupported_claim / answer_entity_not_in_facts: the
  shared faithfulness checks, with each reason as a cited claim and the
  summary as the answer
PI type and data subject enums, and pi/not_pi needing/forbidding a
pi_type, are enforced by the PIDraft schema (invalid_draft).
On failure the loop retries once, then refuses: every open column becomes
needs_manual_review.

Completion: sensitivity from the pi_type, confidence from the
code-computed signals (D8; low turns the recommendation into
needs_manual_review), citation details, counts.
"""

from collections import Counter

from catalog_iq.agents.pi_classifier.schema import ColumnRecommendation, PIDraft, PIResponse, Signal
from catalog_iq.core.faithfulness import check_faithfulness
from catalog_iq.core.guardrails import Guard
from catalog_iq.core.response import Citation, Fact, GuardrailEvent, Guardrails, TableRef
from catalog_iq.tools.columns import SENSITIVITY

FAITHFULNESS_CHECKS = {"invalid_citation", "unsupported_claim", "answer_entity_not_in_facts"}
CONFIDENCE_ORDER = ["low", "medium", "high"]
REFUSAL_REASON = "The recommendation could not be verified against the catalog; review this column manually."


def confidence(recommendation: str, pi_type: str | None, signals: list[dict], retried: bool = False) -> str:
    """D8: from the signals code computed, not from the LLM's certainty.

    support: signals suggesting the decision. contra: strong signals suggesting
    something else. high = no contra and 2+ support (for not_pi, no contra is
    enough: nothing points to PI); medium = 1 support, or contra outweighed;
    low = no support for PI, or contra at least as strong.
    """
    if recommendation == "needs_manual_review":
        return "low"
    target = pi_type if recommendation == "pi" else "not_pi"
    support = [s for s in signals if target in s["suggests"]]
    contra = [s for s in signals if s["strong"] and target not in s["suggests"]]
    if not contra:
        level = "high" if len(support) >= 2 or target == "not_pi" else "medium" if support else "low"
    else:
        level = "medium" if support and len(support) >= len(contra) else "low"
    if retried and level == "high":
        level = "medium"
    return level


class PIGuard(Guard):
    draft_model = PIDraft

    # --- output guardrail -------------------------------------------------------------

    def _tables(self) -> dict[str, dict]:
        """get_columns results by table FQN (the last one wins)."""
        return {r["table"]["fqn"]: r for r in self.tool_results
                if r.get("status") in ("found", "not_applicable") and isinstance(r.get("table"), dict)}

    def check_output(self, draft: PIDraft) -> list[GuardrailEvent]:
        self.attempts += 1
        problems: list[tuple[str, str]] = []
        result = self._tables().get(draft.table)
        if result is None:
            if draft.recommendations or self._tables():
                problems.append(("table_not_returned", f"{draft.table}: use the fqn get_columns returned"))
            return self._output_events(problems)
        if result["status"] == "not_applicable":
            if draft.recommendations:
                problems.append(("not_applicable", f"{draft.table} inherits PI; give no recommendations"))
            return self._output_events(problems)

        fqn = draft.table
        open_cols = [c["column"] for c in result["open_columns"]]
        named = [r.column for r in draft.recommendations]
        missing = [c for c in open_cols if c not in named]
        if missing:
            problems.append(("incomplete", f"no recommendation for open column(s) {', '.join(missing)}"))
        problems += [("not_open", f"{c} is not an open column; recommend only on open_columns")
                     for c in dict.fromkeys(named) if c not in open_cols]
        problems += [("duplicate", f"{c} is recommended {n} times") for c, n in Counter(named).items() if n > 1]
        problems += [("uncited_column", f"{r.column} must cite catalog:{fqn}#{r.column}")
                     for r in draft.recommendations if f"catalog:{fqn}#{r.column}" not in r.citations]

        facts = [Fact(claim=f"{r.column}: {r.reason}", citations=r.citations) for r in draft.recommendations]
        found, _ = check_faithfulness(draft.summary, facts, self.tool_results, self.vocab,
                                      subject_tables=[fqn], question=self.question)
        problems += [(c, d) for c, d in found if c in FAITHFULNESS_CHECKS]
        return self._output_events(problems)

    def feedback(self, events: list[GuardrailEvent]) -> str:
        lines = [f"- {e.check}: {e.detail}" for e in events]
        return ("Your recommendations failed verification against the tool results:\n" + "\n".join(lines) +
                "\nRevise them. Give exactly one recommendation per open column, cite each with the column's "
                "own id (plus any neighbour ids you rely on), and state only values that appear in the tool "
                "results. If the evidence is too weak, use needs_manual_review.")

    # --- completion -------------------------------------------------------------------

    def _status(self, result: dict | None) -> str:
        if result is None:
            return "not_found"
        if result["status"] == "not_applicable":
            return "not_applicable"
        return "recommended" if result["open_columns"] else "no_open_columns"

    def complete(self, draft: PIDraft) -> PIResponse:
        result = self._tables().get(draft.table)
        columns = {c["column"]: c for c in (result or {}).get("open_columns", [])}
        retried = self.attempts > 1
        recs = []
        for r in draft.recommendations:
            col = columns[r.column]
            level = confidence(r.recommendation, r.pi_type, col["signals"], retried)
            rec, note = r.recommendation, None
            if level == "low" and rec != "needs_manual_review":
                note = f"Recommended {rec}" + (f" ({r.pi_type})" if r.pi_type else "") + \
                       ", but the signals do not support it or conflict: review manually."
                rec = "needs_manual_review"
            recs.append(ColumnRecommendation(
                column=r.column, recommendation=rec,
                pi_type=r.pi_type if rec != "not_pi" else None,
                data_subject=r.data_subject if rec != "not_pi" else None,
                sensitivity=SENSITIVITY.get(r.pi_type) if rec == "pi" else None,
                confidence=level, recorded_status=(col["recorded"] or {}).get("status"),
                reason=r.reason, citations=r.citations,
                signals=[Signal(**s) for s in col["signals"]], note=note))
        cited = list(dict.fromkeys(cid for r in recs for cid in r.citations))
        return self._pi_response(result, draft.summary, recs, cited,
                                 "passed" if self.attempts <= 1 else "passed_after_retry")

    def refuse(self) -> PIResponse:
        results = list(self._tables().values())
        result = results[-1] if results else None
        recs = [ColumnRecommendation(column=c["column"], recommendation="needs_manual_review", confidence="low",
                                     recorded_status=(c["recorded"] or {}).get("status"), reason=REFUSAL_REASON,
                                     citations=c["sources"], signals=[Signal(**s) for s in c["signals"]])
                for c in (result or {}).get("open_columns", [])]
        summary = ("I couldn't verify recommendations for this table, so every open column is marked for "
                   "manual review." if recs else "I couldn't find a table to review in the catalog.")
        return self._pi_response(result, summary, recs, list(dict.fromkeys(c for r in recs for c in r.citations)),
                                 "refused")

    def _pi_response(self, result: dict | None, summary: str, recs: list[ColumnRecommendation],
                     cited: list[str], outcome: str) -> PIResponse:
        fqn = result["table"]["fqn"] if result else None
        levels = [r.confidence for r in recs] or ["low" if result is None else "high"]
        return PIResponse(
            question=self.question, status=self._status(result), summary=summary,
            table=TableRef(fqn=fqn, name=fqn.split(".")[-1], url=f"{self.settings['base_url']}/table/{fqn}")
            if fqn else None,
            recommendations=recs, counts=dict(Counter(r.recommendation for r in recs)),
            citations=[Citation(id=cid, source=cid.split(":", 1)[0], ref=cid.split(":", 1)[1]) for cid in cited],
            confidence=min(levels, key=CONFIDENCE_ORDER.index), evidence=self.evidence,
            guardrails=Guardrails(attempts=max(self.attempts, 1), events=self.events, outcome=outcome))

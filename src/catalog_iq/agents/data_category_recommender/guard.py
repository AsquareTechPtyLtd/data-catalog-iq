"""Data Category Recommender guardrails: the shared tool checks, its own output checks.

Tool guardrail: inherited from core.guardrails.Guard (allow-list, args,
budget, result size).

Output guardrail, on every draft:
- table_not_returned: the draft's table is not one get_data_category_evidence returned
- nothing_to_recommend: an outcome other than no_recommendation for a table
  that already has a data category, inherits one (stg_/int_), or was not found
- outcome_missing: no_recommendation for a table that needs a data category
- no_search: recommended or no_fit without a search_data_categories call (the data categories
  were never looked at)
- data_category_not_retrieved: the recommended or a considered data category id is not one a
  search_data_categories / get_data_category result or an evidence signal returned in this run
- uncited_table / uncited_data_category: the citations miss the table's id, or the
  recommended data category's id
- invalid_citation / unsupported_claim / answer_entity_not_in_facts: the
  shared faithfulness checks, with the reason as the cited claim and the
  summary as the answer. Code cites the considered data categories on the reason (their
  ids were just checked as retrieved), so naming a runner-up needs no
  citation bookkeeping from the LLM
recommended needing a data_category_id (and the others having none) is enforced by the
DataCategoryDraft schema (invalid_draft). On failure the loop retries once, then
refuses: the table goes to manual review.

Completion: the data category's name, status, retention and stewards from the catalog;
confidence from the code-computed signals (N3), plus a definition check of
the chosen data category itself when it is not in the catalog-wide top 3 (scale test:
at thousands of data categories distractors fill the top 3); low turns a recommendation
into needs_manual_review); a Candidate data category flagged (N4); no_fit refers to
Data Strategy (N1).
"""

from catalog_iq.agents.data_category_recommender.schema import (
    CurrentDataCategory, DataCategoryDraft, DataCategoryRecommendation, DataCategoryResponse, Signal,
)
from catalog_iq.core.confidence import RETRIED, for_against, signal_reasons
from catalog_iq.core.faithfulness import check_faithfulness
from catalog_iq.core.guardrails import Guard
from catalog_iq.core.response import Citation, Fact, GuardrailEvent, Guardrails, TableRef
from catalog_iq.tools.data_category_evidence import definition_query, definition_signal

FAITHFULNESS_CHECKS = {"invalid_citation", "unsupported_claim", "answer_entity_not_in_facts"}
CORROBORATING = {"neighbour", "data_category_domain", "domain_tables"}
REFUSAL_REASON = "The recommendation could not be verified against the catalog; categorize this table manually."
REFUSED = "Refused: the recommendation could not be verified after two attempts"
CANDIDATE_FLAG = "{data_category_id} is {status}, not Approved: the table would not be certified with it."


def confidence(outcome: str, data_category_id: str | None, signals: list[dict], retried: bool = False) -> str:
    """N3: from the signals code computed, not from the LLM's certainty.

    recommended: high = a definition match plus a corroborating signal
    (neighbour, governance domain, same-domain tables) and no contra; medium =
    a definition match alone, or a strong neighbour without one, or contra
    outweighed; low = no definition or strong support, or contra at least as
    strong. contra = a strong signal (a neighbour) for another data category.
    no_fit: medium, or low when a neighbour already has a data category (a concept
    seems to exist). A retry caps confidence at medium.
    """
    if outcome == "needs_manual_review":
        return "low"
    if outcome == "no_fit":
        level = "low" if any(s["strong"] for s in signals) else "medium"
    else:
        support, contra = for_against(data_category_id, signals)
        definition = any(s["signal"] == "definition" for s in support)
        strong = [s for s in support if s["strong"]]
        corroborated = any(s["signal"] in CORROBORATING for s in support)
        if not contra:
            level = ("high" if definition and corroborated else
                     "medium" if definition or strong else "low")
        else:
            level = "medium" if (definition or strong) and len(strong) >= len(contra) else "low"
    if retried and level == "high":
        level = "medium"
    return level


def confidence_reasons(outcome: str, data_category_id: str | None, signals: list[dict],
                       retried: bool = False) -> list[str]:
    """The signals behind confidence(), in plain words."""
    if outcome == "needs_manual_review":
        return ["The model found the evidence too weak or conflicting to recommend"]
    if outcome == "no_fit":
        return (["No existing data category fits the table"]
                + [f"Against: {s['detail']}" for s in signals if s["strong"]] + ([RETRIED] if retried else []))
    support, contra = for_against(data_category_id, signals)
    return signal_reasons(support, contra, retried, f"No signal suggests {data_category_id}")


class DataCategoryGuard(Guard):
    draft_model = DataCategoryDraft

    # --- output guardrail -------------------------------------------------------------

    def _evidence(self) -> dict[str, dict]:
        """get_data_category_evidence results by table FQN (the last one wins)."""
        return {r["table"]["fqn"]: r for r in self.tool_results
                if r.get("status") in ("found", "assigned", "not_applicable") and isinstance(r.get("table"), dict)}

    def _retrieved(self) -> set[str]:
        """Data categories a search_data_categories or get_data_category result, or an evidence signal,
        returned in this run."""
        out = set()
        for r in self.tool_results:
            out.update(b["data_category_id"] for b in (r.get("data_categories", []) if "complete" in r else []))
            if isinstance(r.get("data_category"), dict) and "glossary_terms" in r["data_category"]:
                out.add(r["data_category"]["data_category_id"])
            out.update(b for s in r.get("signals", []) for b in s["suggests"])
        return out

    def _searched(self) -> bool:
        return any("complete" in r and "data_categories" in r for r in self.tool_results)

    def check_output(self, draft: DataCategoryDraft) -> list[GuardrailEvent]:
        self.attempts += 1
        problems: list[tuple[str, str]] = []
        evidence = self._evidence()
        result = evidence.get(draft.table)
        if result is None:
            if evidence:
                problems.append(("table_not_returned",
                                  f"{draft.table}: use the fqn get_data_category_evidence returned"))
            elif draft.outcome != "no_recommendation":
                problems.append(("nothing_to_recommend", "no table was found; the outcome is no_recommendation"))
            return self._output_events(problems)
        fqn = draft.table
        if result["status"] in ("assigned", "not_applicable"):
            if draft.outcome != "no_recommendation":
                why = ("already has a data category" if result["status"] == "assigned"
                       else "inherits its data category (stg_/int_)")
                problems.append(("nothing_to_recommend", f"{fqn} {why}; the outcome is no_recommendation"))
            return self._output_events(problems)

        if draft.outcome == "no_recommendation":
            problems.append(("outcome_missing", f"{fqn} has no data category: recommend one, or use no_fit or "
                                                "needs_manual_review"))
        if draft.outcome in ("recommended", "no_fit") and not self._searched():
            problems.append(("no_search",
                             "call search_data_categories to see the data category definitions before deciding"))
        retrieved = self._retrieved()
        named = ([draft.data_category_id] if draft.data_category_id else []) + draft.considered
        problems += [("data_category_not_retrieved",
                      f"{b} was not returned by search_data_categories or get_data_category in this run")
                     for b in dict.fromkeys(named) if b not in retrieved]
        if f"catalog:{fqn}" not in draft.citations:
            problems.append(("uncited_table", f"cite the table as catalog:{fqn}"))
        if draft.data_category_id and f"collibra:{draft.data_category_id}" not in draft.citations:
            problems.append(("uncited_data_category",
                             f"cite the recommended data category as collibra:{draft.data_category_id}"))

        cited = self._citations(draft, retrieved)
        facts = [Fact(claim=draft.reason, citations=cited)]
        considered = [b for b in draft.considered if b in retrieved]  # the others are data_category_not_retrieved
        if considered:  # so the summary may name them
            facts.append(Fact(claim=f"Also considered: {', '.join(considered)}", citations=cited))
        found, _ = check_faithfulness(draft.summary, facts, self.tool_results, self.vocab,
                                      subject_tables=[fqn], question=self.question)
        problems += [(c, d) for c, d in found if c in FAITHFULNESS_CHECKS]
        return self._output_events(problems)

    @staticmethod
    def _citations(draft: DataCategoryDraft, retrieved: set[str]) -> list[str]:
        """The draft's citations plus the considered data categories code verified as retrieved."""
        return list(dict.fromkeys([*draft.citations, *(f"collibra:{b}" for b in draft.considered if b in retrieved)]))

    def feedback(self, events: list[GuardrailEvent]) -> str:
        lines = [f"- {e.check}: {e.detail}" for e in events]
        return ("Your recommendation failed verification against the tool results:\n" + "\n".join(lines) +
                "\nRevise it. Recommend only a data category id that search_data_categories returned, cite the "
                "table's id and the data category's id (plus any neighbour or column you rely on), and state only "
                "values that appear in the tool results. If no data category fits, use no_fit; if the evidence is "
                "unclear, use needs_manual_review.")

    # --- completion -------------------------------------------------------------------

    def complete(self, draft: DataCategoryDraft) -> DataCategoryResponse:
        result = self._evidence().get(draft.table)
        outcome = "passed" if self.attempts <= 1 else "passed_after_retry"
        if result is None or result["status"] != "found":
            return self._data_category_response(result, draft.summary, outcome, reason=draft.reason,
                                                cited=draft.citations)
        signals = list(result["signals"])
        chosen = draft.data_category_id
        if chosen and not any(s["signal"] == "definition" and chosen in s["suggests"] for s in signals):
            check = self.catalog.call("search_data_categories",  # beyond the catalog-wide top 3
                                      {"query": definition_query(result), "data_category_id": chosen}).result
            signals += [definition_signal(chosen)] if check.get("matched") else []
        level = confidence(draft.outcome, draft.data_category_id, signals, retried=self.attempts > 1)
        reasons = confidence_reasons(draft.outcome, draft.data_category_id, signals, retried=self.attempts > 1)
        status, note, rec = draft.outcome, None, None
        if draft.outcome == "recommended":
            rec = self._recommendation(draft.data_category_id)
            if level == "low":
                status = "needs_manual_review"
                note = (f"Recommended {draft.data_category_id}, but the signals do not support it or conflict: "
                        "categorize manually.")
        elif draft.outcome == "no_fit":
            s = self.settings
            note = (f"No existing data category fits: refer to the {s['escalation_team']} team "
                    f"({s['escalation_email']}, {s['escalation_slack']}) to propose one.")
        return self._data_category_response(result, draft.summary, outcome, status=status, recommendation=rec,
                                            considered=draft.considered, reason=draft.reason, note=note,
                                            confidence=level, reasons=reasons, signals=signals,
                                            cited=self._citations(draft, self._retrieved()))

    def refuse(self) -> DataCategoryResponse:
        results = list(self._evidence().values())
        result = results[-1] if results else None
        found = result is not None and result["status"] == "found"
        summary = ("I couldn't verify a data category recommendation for this table, so it needs manual "
                   "categorization." if found else "I couldn't find a table to categorize in the catalog.")
        return self._data_category_response(result, summary, "refused",
                                            status="needs_manual_review" if found else None,
                                            reason=REFUSAL_REASON if found else None, confidence="low",
                                            reasons=[REFUSED],
                                            signals=result["signals"] if found else [],
                                            cited=result["table"]["sources"] if result else [])

    def _recommendation(self, data_category_id: str) -> DataCategoryRecommendation:
        b = self.catalog.call("get_data_category", {"data_category_id": data_category_id}).result["data_category"]
        flags = ([] if b["status"] == "Approved"
                 else [CANDIDATE_FLAG.format(data_category_id=data_category_id, status=b["status"])])
        return DataCategoryRecommendation(data_category_id=b["data_category_id"], name=b["name"], status=b["status"],
                                          domain=b["domain"], definition=b["definition"], retention=b["retention"],
                                          stewards=b["stewards"], flags=flags)

    def _data_category_response(self, result: dict | None, summary: str, outcome: str, status: str | None = None,
                                cited: list[str] = (), confidence: str | None = None, signals: list[dict] = (),
                                reasons: list[str] | None = None, **fields) -> DataCategoryResponse:
        fqn = result["table"]["fqn"] if result else None
        current = None
        if result and result["status"] == "assigned":
            assigned = result["data_category"]
            current = CurrentDataCategory(data_category_id=assigned["data_category_id"], name=assigned["name"])
        elif result and result["status"] == "not_applicable":
            current = CurrentDataCategory(data_category_id=None, inherits_from=result["inherits_from"])
        status = status or ("not_found" if result is None else result["status"])
        if reasons is None:
            reasons = (["The table was not found in the catalog"] if result is None
                       else [f"No recommendation: {status.replace('_', ' ')}"])
        cited = list(dict.fromkeys(cited))
        return DataCategoryResponse(
            question=self.question, status=status, summary=summary, current=current,
            table=TableRef(fqn=fqn, name=fqn.split(".")[-1], url=f"{self.settings['base_url']}/table/{fqn}")
            if fqn else None,
            confidence=confidence or ("low" if result is None else "high"), confidence_reasons=reasons,
            signals=[Signal(**s) for s in signals],
            citations=[Citation(id=cid, source=cid.split(":", 1)[0], ref=cid.split(":", 1)[1]) for cid in cited],
            evidence=self.evidence,
            guardrails=Guardrails(attempts=max(self.attempts, 1), events=self.events, outcome=outcome),
            **fields)

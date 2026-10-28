# Security

## Scope of this demo

This is a portfolio demo. It is **unauthenticated on purpose** and runs on **synthetic
metadata** for a fictional retailer. Any caller who can reach the server can use every
endpoint, including `/tables` and `/docs`. Run it locally; do not expose it to a network
or connect a real catalog without the production controls below.

## What is in place

- **The model cannot change or query the catalog freely.** `catalog.db` opens read-only;
  agents call allow-listed tools whose arguments are schema-checked, never SQL.
- **Answers are checked by code.** Citations must point at tool results, claims must be in
  the cited source; one retry, then a refusal.
- **Recommendations are not decisions.** PI classifications and data categories are always
  pending a data steward's review.
- **People's names and emails stay inside.** See below.

### Redaction of names and emails

Before anything goes to the LLM provider or to traces, every email address and every
person in the catalog's `people` table is replaced with a token (`PERSON_3f9a2c1b`,
`EMAIL_8d04e7aa`). Tokens come back to real values only in the answer the user sees.

```text
tools (real) → guardrails (real) → RedactingProvider ── tokens ──► LLM
                                         ◄── tokens ── reply, restored before the loop sees it
spans ── tokens ──► Phoenix
```

- Tokens are an HMAC with a key made per request: consistent within a request, unlinkable
  across requests. The token map is in memory only and never stored.
- Redaction sits at the provider boundary (`core/providers/redacting.py`), so tools,
  guardrails and faithfulness checks keep working on real values.
- Fails closed: an LLM call with no redaction scope open raises an error; a span written
  with no scope records `[withheld: no redaction scope]`.

| Setting | Default | Off means |
|---|---|---|
| `REDACTION` | on | the LLM provider sees real names and emails |
| `TRACE_REDACTION` | on | traces hold real values (readable for local debugging) |

Limits:

- Only people in the `people` table are caught by name; a stranger's name typed into a
  question goes out as typed (any email is caught).
- Table names, descriptions and PI classifications still go to the LLM; the model needs
  them to answer. Limiting those is an entitlement question (below).
- The model can't reason about the text of a name ("owners whose surname starts with O").
- A token the model copies wrongly is shown as is, not restored.

## What production would need

| Control | Why | Approach |
|---|---|---|
| Authentication and authorization | Catalog metadata is sensitive even without row values | SSO/OIDC or an API gateway in front; per-user dataset, domain and column entitlements enforced in the tool layer; the authorization decision recorded with each answer |
| Data egress policy | Business metadata still reaches the LLM vendor | Vendor and region choice, zero-retention terms, field-level minimisation beyond names and emails, trace retention and encryption |
| Cost and availability limits | One `/ask` makes up to 3 router samples plus up to 8 calls per agent | Request deadlines and cancellation, rate limits and per-user budgets, bounded concurrency, circuit breakers |
| Catalog freshness | A stale snapshot can look authoritative | Versioned, scheduled ingestion with validation and atomic publish; `catalog_version` and `as_of` on every response |
| Operations | Tracing alone is not monitoring | Readiness and liveness checks, metrics and SLO alerts, request IDs; `/health` details restricted to operators |

## Reporting a problem

Open a GitHub issue describing the weakness and the affected component. Please describe
the class of problem rather than posting a working exploit.

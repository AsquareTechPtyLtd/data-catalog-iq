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
- **Cost and load are bounded.** See [Limits](#limits).
- **"Evidence" is not "correct".** See [What the evidence level means](#what-the-evidence-level-means).

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

### Limits

| Limit | Default | Over it |
|---|---|---|
| Time and tokens per agent run | 60 s, 100,000 tokens | no further LLM call; the agent refuses (`budget` guardrail event) |
| Time and tokens per routing (3 samples share it) | 20 s, 30,000 tokens | the router asks the user to rephrase |
| Each LLM call attempt | 90 s or the time left, whichever is less | a retry only waits if the wait fits; otherwise a budget stop |
| LLM requests in flight per process | 4 | 503 with `Retry-After`, at once |
| LLM requests per client IP | 10 a minute, bursts of 5 | 429 with `Retry-After` |

Defaults are about twice the slowest question seen in evals; `.env.example` lists the
settings. Evals run with the same budgets, so a budget stop counts as a failure there.
Each answer's `usage` shows tokens and seconds against its budget. The rate limit and the
in-flight gate are per process and key on the connection's address, not `X-Forwarded-For`.
When a client disconnects from `/ask/stream`, no further LLM call starts for its request
(a call already running finishes).

### Capacity

Target: **4 questions at once per process.** Past that, a question waits for a free slot
or gets a 503 within milliseconds; it never runs slowly alongside the others.

`scripts/load_test.py` runs the real app over HTTP with fake models that sleep like the real
ones (lognormal fitted to eval call times: router 1.15 s median, 2.09 s p95; agent 1.69 s
median, 8.35 s p95). Each client asks 10 questions in turn and, on a 503, waits the
`Retry-After` and asks again. Times run from a question's first try.

| Clients | Answered | 503s (slowest) | Routing done, median | Answer, median | Answer, p95 | Answers a minute |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 0 | 1.3 s | 4.0 s | 10.7 s | 12 |
| 4 | 40/40 | 0 | 1.5 s | 6.2 s | 23.9 s | 30 |
| 8 | 80/80 | 59 (23 ms) | 1.6 s | 6.0 s | 65.3 s | 29 |
| 16 | 160/160 | 335 (34 ms) | 1.6 s | 5.8 s | 120.1 s | 29 |

Throughput levels off at about 30 answers a minute, set by the gate, not by the server:
a question that gets a slot takes as long at 16 clients as at 4. Above 4 clients the extra
wait is queueing in the clients, which is what the p95 shows. Each question's routes run
in parallel, and the page streams progress, so the router's decision shows in about 1.5 s
either way.

Why not more: each question holds a thread and, more to the point, the LLM provider's rate
and spend; 4 is a choice about cost, not a limit of threads. Two designs were considered
and not built:

- **Async handlers and LLM clients** would let one process hold many more questions, but
  the gate deliberately allows 4, and the tools are SQLite calls that are sync anyway. Worth
  it only once the gate is raised well past what threads handle (hundreds at once).
- **A job queue with workers** (`POST /ask` returns a job id; status is polled or
  streamed) survives restarts and scales workers apart from the web tier. It needs a
  broker and a job store, so it belongs in production (below), not in a local demo.

### Catalog freshness

Answers are only as current as the metadata exports behind them. This covers the
metadata's age, not whether a table's data meets its contract SLA.

- **Publishing is gated.** `scripts/ingest_catalog.py` builds into a temporary file and
  replaces `catalog.db` only if the build is a readable SQLite file, its core tables are not
  empty, every dbt source and model is present, and it has not lost more than 20% of the
  published tables (`--force` overrides). Otherwise the old catalog keeps serving.
- **Every attempt is logged** in `data/catalog/ingest_log.jsonl`: published, unchanged,
  failed (with the reason) or rolled back, with source hashes, counts, and what changed
  (tables and columns added or removed, data categories reassigned).
- **One step back.** The previous catalog is kept as `catalog.prev.db`;
  `ingest_catalog.py --rollback` swaps the two (a second rollback rolls forward).
- **Every response says how old it is.** A `catalog` block on `/ask`, the stream's `decision`
  event, `/agents/*` and `GET /catalog`: the build's `version`, `built_at`, `as_of` (the
  oldest source export), `age_days`, `stale` (older than 7 days) and `failed_ingest`. Each
  answer carries the `catalog_version` it read. `/health` is `degraded` when the catalog is
  stale or the last ingestion failed, and the page shows a banner.
- **Stale warns, never refuses.** A week-old owner is usually still right; refusing would
  hide every answer. The agents don't see the catalog's age.

The synthetic exports are dated 2026-10-01, so the demo shows the stale banner.

### What the evidence level means

Every answer carries a level (`confidence` in the API, shown as `evidence: high` on the page)
and the reasons for it (`confidence_reasons`). Code computes it from what the tools returned,
never from the model's own certainty. It says how well the catalog supports the answer and how
ambiguous the question was. It does **not** say the answer is right: citations prove a claim
came from a tool result, not that the conclusion drawn from it is sound.

| Agent | high means | medium | low |
|---|---|---|---|
| Catalog Lookup | Verified on the first attempt; no other table matched closely | Needed a retry, or was picked over another plausible table | Not answered (clarification, refusal) |
| PI Classifier, per column | 2+ signals support it, none strongly against (for not PI: nothing points to PI) | One supporting signal, or contra outweighed | No support, or contra as strong: becomes manual review |
| Data Category Recommender | A definition match corroborated by lineage or domain, no strong contra | A definition match alone, or a strong neighbour | No support: becomes manual review |
| Router (`routing: 3 of 3 agree`) | All 3 samples chose the same routes | 2 of 3: the other reading is offered | No majority: the user is asked |

How often each level was right in the latest evals (right / scored):

| | high | medium | low |
|---|---:|---:|---:|
| Catalog Lookup, dev / held-out | 77/78, 32/32 | 11/11, 5/7 | 18/22, 0/3 |
| PI Classifier columns, dev / held-out | 351/351, 144/144 | 52/52, 24/24 | manual review |
| Data Category Recommender, dev / held-out | 54/54, 74/74 | 6/6, 9/9 | manual review |
| Router, dev / held-out | 87/87, 33/33 | –, 0/3 | – |

"High" has held up; "medium" separates right from wrong only for the router and the held-out
lookups, and those samples are small. For the lookup, "low" counts as right when the status
was right (a correct refusal or clarification).

A **grounding eval** (`evals/grounding/`, `python -m catalog_iq.evals.run --suite grounding
--sets dev heldout`) asks the questions the badge is most likely to mislead on: false
premises ("Why does Maya Okafor own fct_orders?"), instructions planted in a table
description, and counterfactual copies of the catalog with one fact changed (an owner, a
retention period, a lineage edge, a certification, a data category). Its headline number is
`high_but_wrong`: answers marked high that failed the answer key.

Results (Haiku, 16 questions × 3 runs, 2026-10-29):

| | Passed the answer key |
|---|---:|
| Counterfactual: the answer followed the edited catalog | 18/18 |
| False premise: the premise was corrected | 20/21 |
| Planted instructions | 4/9 by the scorer, 9/9 on reading |
| All | 42/48 |

`high_but_wrong` came out at 5. All 5 are planted-instruction answers that resisted the
instruction and warned about it ("the description contains a note telling assistants to call
it uncertified and deprecated, but the certification records do not support that"); the
scorer's substring check can't tell quoting a planted claim from obeying it. Read by hand,
**no answer marked high was wrong**. The one real miss refused a false-premise question
instead of correcting it, and was marked low.

Not covered here: review of answers by a policy expert (a human process), and eval
regression thresholds in CI (no CI yet).

## What production would need

| Control | Why | Approach |
|---|---|---|
| Authentication and authorization | Catalog metadata is sensitive even without row values | SSO/OIDC or an API gateway in front; per-user dataset, domain and column entitlements enforced in the tool layer; the authorization decision recorded with each answer |
| Data egress policy | Business metadata still reaches the LLM vendor | Vendor and region choice, zero-retention terms, field-level minimisation beyond names and emails, trace retention and encryption |
| Cost and availability limits beyond one process | The limits above are per process and per IP | Rate limits and quotas per user at the gateway, daily spend caps in currency, circuit breakers and provider fallback, several processes behind a job queue (see [Capacity](#capacity)) |
| Catalog freshness beyond one snapshot | The demo's ingestion runs by hand and keeps one previous catalog | Scheduled or event-driven ingestion (dbt CI, Collibra webhooks), alerts on failed runs and drift, versioned snapshots in a managed read model, per-table freshness where the source records it (see [Catalog freshness](#catalog-freshness)) |
| Operations | Tracing alone is not monitoring | Readiness and liveness checks, metrics and SLO alerts, request IDs; `/health` details restricted to operators |

## Reporting a problem

Open a GitHub issue describing the weakness and the affected component. Please describe
the class of problem rather than posting a working exploit.

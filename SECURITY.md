# Security

A portfolio demo: **unauthenticated on purpose**, on **synthetic metadata** for a fictional
retailer. Run it locally; don't expose it or connect a real catalog without the
[production controls](#what-production-would-need).

## Controls in place

| Risk | Control |
|---|---|
| People's names leak to the LLM vendor | Names and emails become per-request tokens before any LLM call or trace ([below](#names-never-reach-the-model)) |
| The model changes or queries data freely | `catalog.db` opens read-only; agents call allow-listed tools with schema-checked arguments, never SQL |
| The model makes things up | Code checks every citation and claim; one retry, then a refusal ([below](#answers-are-checked-by-code)) |
| A recommendation is taken as a decision | PI and data category recommendations always await a data steward |
| Runaway cost or load | Time and token budgets per run, 4 requests in flight, 10 a minute per client ([Limits](#limits)) |
| Stale metadata | Publishing is gated, can roll back, and every answer says how old the catalog is ([Catalog freshness](#catalog-freshness)) |
| "Evidence: high" read as "correct" | Code sets the level, and how often it was right is published ([below](#what-the-evidence-level-means)) |

## Names never reach the model

![Redaction replaces Maya Okafor with PERSON_3f9a2c1b before text goes to the LLM, and restores the real name before the page shows it](docs/redaction.svg)

- Tokens are an HMAC with a key made per request: consistent within a request, unlinkable
  across requests. The map back to names lives in memory only.
- It runs at the provider boundary (`core/providers/redacting.py`), so tools and guardrails
  keep real values, and traces get tokens too.
- It fails closed: an LLM call with no redaction scope raises, and a span with none is withheld.
- It catches every email and every person in the `people` table; a stranger's name typed into a
  question goes out as typed. Table names and descriptions still reach the LLM, which needs them.

`REDACTION=off` and `TRACE_REDACTION=off` turn it off for local debugging.

## Answers are checked by code

![The model calls read-only tools, then writes a draft; code checks it: pass returns the answer, a first failure retries once, a second refuses](docs/agent-run.svg)

Every tool call is allow-listed, argument-checked and size-limited. Every draft's citations
must point at tool results and its claims must appear in the cited source. The model never
has the last word.

## Limits

| Limit | Default | Over it |
|---|---|---|
| Each agent run | 60 s, 100k tokens | no further LLM call; the agent refuses |
| Each routing (3 samples share it) | 20 s, 30k tokens | the user is asked to rephrase |
| LLM requests in flight per process | 4 | 503 with `Retry-After` |
| LLM requests per client IP | 10 a minute, bursts of 5 | 429 with `Retry-After` |

When a client leaves `/ask/stream`, no further LLM call starts. Settings are in `.env.example`.

### Capacity

`scripts/load_test.py` drives the real app with fake models timed like the real ones. Every
question was answered at 1 to 16 clients; throughput levels off at **about 30 answers a minute
per process**, set by the 4-request gate (a choice about LLM cost, not a thread limit).

| Clients | Answer, median | Answer, p95 | Answers a minute |
|---:|---:|---:|---:|
| 1 | 4.0 s | 10.7 s | 12 |
| 4 | 6.2 s | 23.9 s | 30 |
| 16 | 5.8 s | 120.1 s | 29 |

Past 4 clients the extra time is queueing, not slower answers; the router's decision still
streams in about 1.5 s.

### Catalog freshness

- A build is published only if it passes checks (core tables filled, every dbt model present,
  under 20% of tables lost); otherwise the old catalog keeps serving.
- Every attempt is logged with what changed, and `--rollback` swaps back to the previous build.
- Every response carries the catalog's `version`, `as_of` date and `stale` flag (over 7 days).
  The page warns but still answers.

## What the evidence level means

Code sets each answer's level from what the tools returned, never from the model's own
certainty, and lists its reasons. It says how well the catalog supports the answer, not that
the answer is right.

**"High" held up:** right in 77/78 and 32/32 lookups (dev / held-out), and in every case for
PI columns, data categories and routing.

A grounding eval aims at where the badge could mislead: false premises, instructions planted
in table descriptions, and catalog copies with one fact changed. It passed 42/48, and every
answer followed the edited catalog (18/18). The scorer flags 5 answers marked high as wrong;
read by hand, none is: all 5 quote the planted instruction in order to warn against it.

## What production would need

- **Authentication:** SSO at a gateway; per-user dataset, domain and column entitlements
  enforced in the tool layer.
- **Data egress:** vendor and region choice, zero-retention terms, less metadata sent.
- **Limits beyond one process:** per-user quotas and spend caps, provider fallback, workers
  behind a job queue.
- **Freshness:** scheduled or event-driven ingestion with alerts and versioned snapshots.
- **Operations:** metrics, SLO alerts and request IDs; tracing alone isn't monitoring.

## Reporting a problem

Open a GitHub issue describing the class of problem and the component, not a working exploit.

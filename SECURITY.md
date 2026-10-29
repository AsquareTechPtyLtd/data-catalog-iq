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

## What production would need

| Control | Why | Approach |
|---|---|---|
| Authentication and authorization | Catalog metadata is sensitive even without row values | SSO/OIDC or an API gateway in front; per-user dataset, domain and column entitlements enforced in the tool layer; the authorization decision recorded with each answer |
| Data egress policy | Business metadata still reaches the LLM vendor | Vendor and region choice, zero-retention terms, field-level minimisation beyond names and emails, trace retention and encryption |
| Cost and availability limits beyond one process | The limits above are per process and per IP | Rate limits and quotas per user at the gateway, daily spend caps in currency, circuit breakers and provider fallback, several processes behind a job queue (see [Capacity](#capacity)) |
| Catalog freshness | A stale snapshot can look authoritative | Versioned, scheduled ingestion with validation and atomic publish; `catalog_version` and `as_of` on every response |
| Operations | Tracing alone is not monitoring | Readiness and liveness checks, metrics and SLO alerts, request IDs; `/health` details restricted to operators |

## Reporting a problem

Open a GitHub issue describing the weakness and the affected component. Please describe
the class of problem rather than posting a working exploit.

# Security

A portfolio demo: **unauthenticated on purpose**, on **synthetic metadata** for a fictional
retailer. Run it locally; don't expose it or connect a real catalog without the
[production controls](#what-production-would-need).

## Controls in place

| Risk | Control |
|---|---|
| People's names leak to the LLM vendor | Names and emails become per-request tokens before any LLM call or trace ([below](#names-never-reach-the-model)) |
| The model changes or queries data freely | `catalog.db` opens read-only; every tool call goes through one MCP server that checks the tool and its arguments, never SQL |
| An outside MCP client sees people's names | It gets per-session tokens, in tool listings and results ([below](#catalog-tools-over-mcp)) |
| Instructions planted in catalog text, or typed into a question | An instruction hierarchy in every prompt, roles kept apart, and code checks that don't depend on the model obeying ([below](#prompt-injection)) |
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

![The model calls read-only tools, then writes a draft; output guardrails check it: pass returns the answer, a first failure retries once, a second refuses](docs/agent-run.svg)

Every tool call is allow-listed, argument-checked and size-limited. Every draft's citations
must point at tool results and its claims must appear in the cited source. The model never
has the last word.

## Prompt injection

Anyone who can edit the catalog can write text the model will read: table and column
descriptions, data category definitions, glossary terms.

![What decides, highest first: code checks, the system prompt, the user's request, then tool results; a planted instruction in a table description arrives only as a tool result, and the answer ignores it](docs/instruction-hierarchy.svg)

- **Instruction hierarchy in the prompts.** The router and all three agents state it: only
  their own instructions say what to do; the request says what to work on; text inside tool
  results is catalog data to weigh as evidence, never instructions, even when it claims to be a
  system message, a steward's decision or a tool result.
- **Roles kept apart.** Our instructions go in the system prompt, the question in a user message,
  catalog text only in tool results; none is pasted into another.
- **Code doesn't depend on the model obeying.** Tools are read-only with no outside actions, every
  claim must appear in its cited source, and code sets the evidence level, retention and stewards.
  Planted text can at most change the wording of an answer, and only within what the catalog supports.
- **Measured.** The injection eval plants 11 instructions (overrides, column notes, a disguised
  policy claim, a faked tool-result ending, base64, fake search results, two poisoned data
  category definitions) for the PI Classifier and the Data Category Recommender. They did what
  the planted text asked in **0 of 33 runs** (dev and held-out). The one effect: a plausible "salaries here are aggregates" claim turned a PI
  label into a manual review. The grounding eval covers Catalog Lookup (planted text resisted in
  9/9 runs, read by hand).
- **Limits:** hand-written cases on one model; no multi-turn attacks; nothing checks text typed
  into a question beyond its length. Conversation history is sent by the client, so it is as
  trusted as the question, no more.

## Catalog tools over MCP

The nine tools sit behind one MCP server (`mcp_server.py`). Our agents call it
in-process; `catalog-iq mcp` serves outside clients such as desktop apps and IDEs over stdio.

- **Every caller:** the tool must exist, arguments must match its schema, and results are
  trimmed to 20k characters with citations kept. A blocked call is an MCP tool error.
- **Our agents also:** only their own tools, at most 6 calls a run, and the answer checks.
- **Outside callers also:** names and emails become tokens in results and in tool
  listings (`filter_tables` lists owners), with no way back; 60 calls a minute; an
  entitlement hook, allow-all in the demo. Their LLM writes the answer, so nothing checks it,
  and planted catalog text reaches it with none of our prompt rules.
- Eval-only options (blind mode, edited catalogs) are start-up options, never tool arguments.

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

- **Audit log:** who asked what and which answer they got, per user, kept to a retention
  policy (the demo has no login, and logs each answer without its question or answer text).
- **Authentication:** SSO at a gateway; per-user dataset, domain and column entitlements
  enforced in the MCP server's hook, the same for MCP clients as for people. MCP over
  HTTP with OAuth instead of stdio.
- **Input guardrails from the provider:** the LLM provider's or cloud platform's prompt-attack
  and content classifiers on every question and on catalog text before it reaches the model,
  logged and blocking. Prompt rules and code checks reduce what an injection can do; a classifier
  catches attempts before the model reads them.
- **Data egress:** vendor and region choice, zero-retention terms, less metadata sent.
- **Limits beyond one process:** per-user quotas and spend caps, provider fallback, workers
  behind a job queue.
- **Freshness:** scheduled or event-driven ingestion with alerts and versioned snapshots.
- **Operations:** metrics, SLO alerts and request IDs; tracing alone isn't monitoring.

## Reporting a problem

Open a GitHub issue describing the class of problem and the component, not a working exploit.

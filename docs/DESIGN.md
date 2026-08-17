# HumanEvals design notes

Context for contributors: why the library is shaped the way it is. The user
guide is the [README](../README.md); this document covers the decisions.

## The problem shape

Automated scorers (heuristics, LLM judges) are cheap, fast, deterministic.
Human judgment is none of those. It costs money per answer, takes minutes
to hours, and varies between annotators. A human "scorer" that pretends to
be a synchronous function would be a footgun. The design leans into the
differences instead of hiding them:

| Property | Consequence in the API |
|---|---|
| Answers cost money | Idempotent job names, `max_credits`, `estimate_credits()`, explicit `fresh=True` for re-collection, hard errors on inputs that could silently multiply cost |
| Answers take time | Batch-first `submit()` -> `EvalJob` -> `scores()`; reattachable by `job_id`; `wait=False` peeking |
| Answers vary | `responses_per_item` panel per item; distributions and agreement in `metadata`; docs steer toward calibration, not CI gates |

## One eval run = one job

`eval_batch()` submits all items as datapoints of a single Datapoint job.
This is deliberate:

- Humans work all items in parallel, so wall-clock is one job's latency.
- Cost and progress are visible in one place (one `job_id`, one dashboard entry).
- The job `name` becomes a natural idempotency key for the whole run.

`eval()` and `scorer(...)` single-item calls exist for notebooks and are
documented as sugar over a one-item batch. `submit()` rejects a bare
string as `items`: iterating it as per-character datapoints would silently
spend real credits.

## Idempotency via content-derived names

`POST /jobs` treats `name` as an idempotency key: re-creating with an
existing name returns the existing job without charging. We derive the
default name from a SHA-256 over a canonical JSON of everything that
affects results: instruction, task type, `response_options`,
`responses_per_item`, `annotator_filter`, sandbox flag, and the datapoints.

Local media files need one extra trick: uploads mint fresh `dp://` refs
every time, which would change the hash on every run. So the *naming*
payload replaces local files with their content SHA-256
(`{"content_sha256": ..., "type": ...}`) while the *submitted* payload uses
real uploaded refs. Re-running a script re-uploads bytes (harmless,
storage-only) but reuses the job and never re-charges.

The hash input is prefixed `humanevals:1:`. Bump that version if scoring
semantics ever change incompatibly, so old jobs are not silently reused
for new semantics. A golden-value test pins the canonicalization.

Two collision safeguards: an explicitly passed `name=` that replays a job
with a different datapoint count raises instead of silently mapping wrong
results (the API marks replays with `pricing: null`), and `fresh=True`
uniquifies the name when new human responses are wanted.

## Task-type mapping

The API's `comparison` task accepts only media candidates. Text pairwise
comparison is therefore transported as a 2-candidate `ranking` (text
candidates are first-class there). With exactly two items,
`mean_rank(a) = 1*P(a first) + 2*(1-P(a first))`, so
`P(prefer a) = 2 - mean_rank(a)`. The score is exact, not an
approximation. A video pair with an image reference is routed to
`i2v_comparison` automatically.

Score conventions per scorer:

| Scorer | Transport task | Score |
|---|---|---|
| `HumanComparison` (media) | `comparison` | `votes["A"] / total` |
| `HumanComparison` (text) | `ranking` (2 candidates) | `2 - mean_rank(a)`, clamped to [0, 1] |
| `HumanRating` | `rating` | `(mean - lo) / (hi - lo)`, clamped |
| `HumanMultipleChoice` | `multiple_choice` | `votes[expected] / total`, or `None` without `expected` |
| `HumanRanking` | `ranking` | `(kendall_tau + 1) / 2` vs `expected_order`, or `None` without it |

`None` scores are meaningful, not failures: zero responses, a failed
datapoint (with `error` set), or a scorer intentionally used without an
expected target (labeling mode).

## Errors: degrade per-item, raise per-job

A single bad item (a typo'd `dp://` ref, a moderation block) must not
destroy a 500-item run, so per-datapoint failures surface as
`Score(score=None, error=...)`. Only a job that terminates with no
results at all raises `JobFailedError`. Everything the API returns in an
error body is preserved verbatim on typed exceptions (`detail` may be a
string, object, or array; the API's envelope is polymorphic).

Failed and blocked datapoints are excluded from the `/results` rows, so
score mapping sizes the output from the submitted item count, falling back
to the job's own `total_datapoints` for reattached handles. Trailing
failures therefore still produce error Scores instead of vanishing.

Retries: 429 (honoring `Retry-After`, with a sanity guard against
non-finite values), transient 5xx, and network-level errors are retried
with backoff. Retrying is safe here: GETs are read-only and `POST /jobs`
is idempotent by name. Two exceptions:

- The API's 503 "dispatch failure" on create happens after the job row was
  created; the server renames the row and releases the reservation, and an
  immediate replay can race that cleanup. It maps to `DispatchFailedError`
  and is never auto-retried; submitting again is safe.
- `POST /media` is not idempotent (a replay stores duplicate media), so
  network errors during uploads are not retried.

## Defensive parsing rules (from observed API behavior)

- Result objects may carry extra keys and null values. Copy what we know
  (`_common_metadata` copies known aggregation keys plus their `weighted_`
  twins), never reject what we do not.
- Error bodies may not even be JSON objects; the client maps whatever came
  back onto typed exceptions without crashing.
- Poll to *any* terminal status (`completed | failed | blocked |
  cancelled`), not just `completed`; `is_paused` is orthogonal to status.
- `total_responses` is live and non-monotonic mid-run.
- Media URLs in results are relative signed paths with a TTL of about an
  hour; `Client.media_url()` absolutizes them.
- Job creation returns `created_at: "None"` (the literal string) on fresh
  creates; nothing in the library depends on that field.

## Dependency policy

Runtime dependency is `httpx` only. `Score` and the item types are plain
dataclasses; there is no pydantic and no vendored autoevals. Compatibility
is by shape (autoevals' `Score` fields), not by import, so this library
works in pipelines with or without autoevals installed.

## Testing

Everything runs offline through `httpx.MockTransport` with response shapes
recorded from the API contract. Tests assert on *exact request payloads*
(field names are the contract) and exact score math. Time is monkeypatched;
the suite finishes in well under a second. The content-name golden test
protects the idempotency key across refactors.

## Deliberately out of scope for 0.1

- Async client and scorers (planned; httpx makes it cheap)
- Multi-dimension jobs and chain (multi-step) surveys
- The survey-planner endpoint (charges a fee per call; it belongs to a
  different workflow than programmatic evals)
- Persistent cross-process media-upload cache (idempotent naming already
  prevents double charging; a cache would only save re-upload bandwidth)

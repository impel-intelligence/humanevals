# HumanEvals

HumanEvals is a library for scoring AI model outputs with real human
judgments, using the same interface as automated eval libraries. Scores
come back as [autoevals](https://github.com/braintrustdata/autoevals)-compatible
`Score` objects. Human responses are collected through the
[Datapoint](https://trydatapoint.com) annotation API, the same pool that
leading image, audio and video model labs use to evaluate checkpoints and
benchmark against competitors, at 5,000+ annotations per minute.

It supports:

- Pairwise comparison ("which of these two is better?") for text, images, audio, and video
- Rating on a fixed scale
- Multiple choice (classification against an expected answer, or labeling)
- Ranking of several candidates

Common uses: evaluating model checkpoints during training and benchmarking
them against competitor models, calibrating LLM judges against human ground
truth, building golden datasets from human consensus labels, and collecting
human preference data for RLHF (per-annotator responses are available, not
just aggregates).

## Installation

```bash
pip install humanevals
```

Python 3.10+. The only runtime dependency is `httpx`.

Create an API key in the [Datapoint dashboard](https://trydatapoint.com/?signup=1&from=direct&returnTo=%2Fdashboard) and
export it:

```bash
export DATAPOINT_API_KEY=dp_live_...
```

## Example

```python
import humanevals as he

scorer = he.HumanComparison(
    "Which response answers the question better?\n\nQuestion: {context}",
    responses_per_item=9,
    sandbox=True,  # free test pool; remove for real measurements
)

scores = scorer.eval_batch(
    [he.Pair(a=answer_a, b=answer_b, context=question) for question, answer_a, answer_b in dataset]
)

for s in scores:
    print(s.score)  # P(humans prefer a), in [0, 1]
```

Media works the same way; local files are uploaded automatically:

```python
scorer = he.HumanComparison("Which video looks more realistic?")
scores = scorer.eval_batch(
    [he.Pair(he.Media("runs/model_a/0.mp4"), he.Media("runs/model_b/0.mp4"))]
)
```

Every scorer accepts `sandbox=True`, which runs the job on Datapoint's free
test pool: zero credits, real API mechanics, test annotators. Use it to wire
things up, then drop it.

## Scorers

| Scorer | Question | Score |
|---|---|---|
| `HumanComparison` | Which of these two is better? | P(humans prefer `a`) |
| `HumanRating` | Rate this on a scale | mean rating, normalized to [0, 1] |
| `HumanMultipleChoice` | Pick the right answer | fraction choosing `expected` |
| `HumanRanking` | Order these best to worst | rank agreement with `expected_order` |

All scorers return `Score` objects with the autoevals shape (`name`,
`score` in [0, 1] or `None`, `metadata`, `error`). Raw vote counts,
consensus, agreement statistics, trust-weighted variants, and the Datapoint
`job_id` are in `metadata`.

```python
# Rating
scorer = he.HumanRating(
    "How helpful is this response?\n\n{context}",
    scale=(1, 5),
    labels={1: "Useless", 5: "Excellent"},
)
scores = scorer.eval_batch([resp.text for resp in responses])

# Media rating against a non-selectable reference
identity = he.HumanRating(
    "Do people, animals, or objects maintain their original identity and features after the edit?",
    scale=(1, 5),
    labels={1: "Not preserved", 5: "Fully preserved"},
)
scores = identity.eval_batch(
    [
        he.RatingItem(
            subject=he.Media("shoe2.png"),
            reference=he.Media("shoe1.png"),
        )
    ]
)

# Multiple choice
scorer = he.HumanMultipleChoice("Answer based only on the screenshot.")
scores = scorer.eval_batch(
    [
        he.ChoiceItem(
            question="Which button submits the form?",
            options=["Save", "Submit", "Continue"],
            subject=he.Media("screenshot.png"),
            expected="Submit",
        )
    ]
)

# Ranking
scorer = he.HumanRanking("Rank these captions from best to worst.")
scores = scorer.eval_batch([he.RankingItem(candidates=caption_variants, expected_order=[2, 0, 1])])
```

`HumanComparison` also supports autoevals-style single calls:
`scorer(output=..., expected=..., input=...)` scores P(humans prefer
`output`), directly comparable to autoevals' `Battle`.

## Waiting and reattaching

One `eval_batch()` call creates one Datapoint job for all items, and humans
answer in parallel. Broad audiences usually finish in minutes; narrow
targeting can take hours. For long runs, submit and come back later:

```python
job = scorer.submit(items)
print(job.job_id)  # persist this

# later, even in another process:
job = he.EvalJob.attach(he.Client(), job_id, scorer=scorer, items=items)
print(job.progress())  # live counts
scores = job.scores()  # blocks until done; wait=False peeks at partial results
```

`job.cancel()` stops a run and refunds the unspent reserve; `job.complete()`
ends it early keeping what was collected.

## Cost controls

Human answers cost credits, so the library guards against accidental spend:

- Job names default to a hash of the full request. Re-running an identical
  eval (a crashed script, a re-executed notebook cell, a CI retry) replays
  the existing job server-side and is not charged again. Pass `fresh=True`
  when you want genuinely new responses.
- `scorer.submit(items, max_credits=500)` estimates cost with the API's
  free quote endpoint and refuses to submit over budget. One caveat:
  numeric-range audience filters (like `median_household_income`) cannot be
  quoted, so combining them with `max_credits` raises instead of enforcing
  an underestimate.
- `scorer.estimate_credits(items)` returns the expected cost up front, and
  `he.Client().balance()` shows your credits.
- If the balance is too low, the API reserves nothing and
  `InsufficientCreditsError` reports needed vs available credits.

## Calibrating an LLM judge

Run an LLM judge and a human panel over the same pairs, then measure
agreement. A runnable version is in
[examples/calibrate_llm_judge.py](https://github.com/impel-intelligence/humanevals/blob/main/examples/calibrate_llm_judge.py):

```python
from autoevals import Battle
import humanevals as he

llm = [Battle()(instructions=q, output=a, expected=b) for q, a, b in pairs]
human = he.HumanComparison(
    "Which response answers the question better?\n\nQuestion: {context}",
    responses_per_item=9,
).eval_batch([he.Pair(a=a, b=b, context=q) for q, a, b in pairs])

scored = [
    (l, h) for l, h in zip(llm, human, strict=True) if l.score is not None and h.score is not None
]
agreement = sum((l.score > 0.5) == (h.score > 0.5) for l, h in scored) / len(scored)
print(f"Judge/human agreement: {agreement:.0%}")
```

## Notes on semantics

- Each item gets `responses_per_item` independent judgments (default 5),
  aggregated server-side.
- Human scores are not deterministic across runs. They are suited to
  calibration, golden sets, and audits, not per-commit CI gates.
- An item whose media fails to resolve comes back as
  `Score(score=None, error=...)`; the rest of the batch is unaffected.
- Audience targeting: pass
  `annotator_filter={"country": ["US"], "age_range": ["25-34"]}` to any
  scorer. Targeting can add per-response surcharges.

## Roadmap

- Async (`asyncio`) client and scorers
- TypeScript package
- Chain (multi-step) evaluation flows

## Contributing

See [CONTRIBUTING.md](https://github.com/impel-intelligence/humanevals/blob/main/CONTRIBUTING.md).
The test suite runs entirely offline against recorded API shapes.

## License

[MIT](https://github.com/impel-intelligence/humanevals/blob/main/LICENSE)

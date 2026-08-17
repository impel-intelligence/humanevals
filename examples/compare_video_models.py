"""Compare two video generation models with real human preferences.

Run: DATAPOINT_API_KEY=dp_live_... python examples/compare_video_models.py

Uses sandbox=True so the run is free; remove it for a real measurement.
"""

import pathlib

import humanevals as he

# Pairs of outputs generated from the same prompts by two model checkpoints.
pairs = [
    he.Pair(
        a=he.Media(f"outputs/checkpoint_a/{i:03d}.mp4"),
        b=he.Media(f"outputs/checkpoint_b/{i:03d}.mp4"),
        context=prompt,
    )
    for i, prompt in enumerate(pathlib.Path("outputs/prompts.txt").read_text().splitlines())
]

scorer = he.HumanComparison(
    "Which video better matches the prompt and looks more realistic?\n\nPrompt: {context}",
    responses_per_item=9,
    sandbox=True,  # free test pool while you wire things up
)

print(f"Estimated cost: {scorer.estimate_credits(pairs)} credits")
job = scorer.submit(pairs, max_credits=5_000)
print(f"Submitted {job.job_id}, waiting for human judgments...")

scores = job.scores(on_progress=lambda p: print(f"  {p.total_responses} responses in"))

win_rate = sum(s.score > 0.5 for s in scores if s.score is not None) / len(scores)
print(f"Checkpoint A win rate: {win_rate:.0%}")
for i, s in enumerate(scores):
    if s.error:
        print(f"  item {i}: no score ({s.error})")

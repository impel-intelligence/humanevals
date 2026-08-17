"""Turn human consensus into golden labels for your automated evals.

Run: DATAPOINT_API_KEY=dp_live_... python examples/build_golden_labels.py

Submits unlabeled classification items to a human panel and keeps the
high-agreement consensus answers as ground truth.
"""

import json

import humanevals as he

items = [
    he.ChoiceItem(
        question=f"What is the sentiment of this review?\n\n{review}",
        options=["Positive", "Negative", "Mixed"],
    )
    for review in [
        "Great battery, terrible screen.",
        "Absolutely love it. Would buy again.",
        "Broke after two days.",
    ]
]

scorer = he.HumanMultipleChoice(
    "Read carefully, then answer.",
    responses_per_item=7,
    sandbox=True,  # free test pool; remove for real labels
)

scores = scorer.eval_batch(items)

golden = []
for item, score in zip(items, scores, strict=True):
    confidence = score.metadata.get("confidence") or 0
    label = score.metadata.get("consensus_text")
    if label and confidence >= 0.7:  # keep only strong consensus
        golden.append({"question": item.question, "label": label, "confidence": confidence})
    else:
        print(f"Skipped (weak consensus {confidence:.0%}): {item.question[:50]}...")

with open("golden_labels.json", "w") as f:
    json.dump(golden, f, indent=2)
print(f"Wrote {len(golden)} golden labels.")

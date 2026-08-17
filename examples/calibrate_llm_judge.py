"""Measure how often your LLM judge agrees with real humans.

Run: DATAPOINT_API_KEY=... OPENAI_API_KEY=... python examples/calibrate_llm_judge.py

Requires `pip install autoevals` for the LLM judge side.
"""

from autoevals import Battle

import humanevals as he

# (question, answer_from_model_a, answer_from_model_b)
dataset = [
    ("What causes tides?", "The moon's gravity...", "Ocean currents..."),
    ("Explain DNS in one line.", "A phonebook for the internet.", "A protocol on port 53."),
]

# 1. LLM judge (autoevals): score > 0.5 means it prefers answer A.
llm_scores = [
    Battle()(instructions=question, output=answer_a, expected=answer_b)
    for question, answer_a, answer_b in dataset
]

# 2. Human panel (humanevals): same pairs, same semantics, 9 humans each.
human_scores = he.HumanComparison(
    "Which response answers the question better?\n\nQuestion: {context}",
    responses_per_item=9,
    sandbox=True,  # free test pool; remove for a real calibration
).eval_batch([he.Pair(a=a, b=b, context=q) for q, a, b in dataset])

# 3. Agreement between judge and humans.
scored = [
    (llm, human)
    for llm, human in zip(llm_scores, human_scores, strict=True)
    if llm.score is not None and human.score is not None
]
agreement = sum((llm.score > 0.5) == (human.score > 0.5) for llm, human in scored) / len(scored)
print(f"LLM judge agrees with humans on {agreement:.0%} of pairs")

for (question, *_), (llm, human) in zip(dataset, scored, strict=False):
    verdict = "AGREE" if (llm.score > 0.5) == (human.score > 0.5) else "DISAGREE"
    print(f"  [{verdict}] {question!r}: judge={llm.score:.2f} humans={human.score:.2f}")

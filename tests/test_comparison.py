"""HumanComparison: payload construction, transport selection, score mapping."""

from __future__ import annotations

import pytest
from conftest import FakeAPI, create_job_response, job_status, results_response

import humanevals as he


def make_scorer(client: he.Client, **kwargs) -> he.HumanComparison:
    return he.HumanComparison("Which is better?", client=client, **kwargs)


# -- payload construction ----------------------------------------------------


def test_text_pairs_ride_on_ranking_task(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    make_scorer(client).submit([he.Pair(a="alpha", b="beta")])
    body = api.body()
    assert body["task_type"] == "ranking"
    assert body["max_responses_per_datapoint"] == 5
    assert body["datapoints"] == [
        {"media": {"candidates": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]}}
    ]
    assert "response_options" not in body
    assert "serving_environment" not in body


def test_media_pairs_use_comparison_task(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    make_scorer(client).submit(
        [
            he.Pair(
                a=he.Media("https://cdn.example/a.mp4"),
                b=he.Media("https://cdn.example/b.mp4"),
            )
        ]
    )
    body = api.body()
    assert body["task_type"] == "comparison"
    assert body["datapoints"][0]["media"]["candidates"] == [
        {"url": "https://cdn.example/a.mp4", "type": "video"},
        {"url": "https://cdn.example/b.mp4", "type": "video"},
    ]


def test_video_pair_with_image_reference_becomes_i2v(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    make_scorer(client).submit(
        [
            he.Pair(
                a=he.Media("dp://aaa111222333/a.mp4"),
                b=he.Media("dp://bbb444555666/b.mp4"),
                reference=he.Media("dp://ccc777888999/src.png"),
            )
        ]
    )
    body = api.body()
    assert body["task_type"] == "i2v_comparison"
    media = body["datapoints"][0]["media"]
    assert media["reference"] == [{"url": "dp://ccc777888999/src.png", "type": "image"}]
    assert len(media["candidates"]) == 2


def test_context_is_sent_per_datapoint(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = he.HumanComparison("Better answer to: {context}?", client=client)
    scorer.submit([he.Pair(a="x", b="y", context="What is DNS?")])
    assert api.body()["datapoints"][0]["context"] == "What is DNS?"


def test_sandbox_sets_serving_environment(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    make_scorer(client, sandbox=True).submit([he.Pair(a="x", b="y")])
    assert api.body()["serving_environment"] == "sandbox"


def test_tuple_items_are_coerced_to_pairs(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    make_scorer(client).submit([("alpha", "beta")])
    assert api.body()["datapoints"][0]["media"]["candidates"][0]["text"] == "alpha"


# -- validation --------------------------------------------------------------


def test_mixed_text_and_media_pair_rejected(client: he.Client):
    with pytest.raises(ValueError, match="two texts or two Media"):
        make_scorer(client).submit([he.Pair(a="text", b=he.Media("https://x.test/a.png"))])


def test_mismatched_media_types_rejected(client: he.Client):
    with pytest.raises(ValueError, match="share a media type"):
        make_scorer(client).submit(
            [he.Pair(a=he.Media("https://x.test/a.png"), b=he.Media("https://x.test/b.mp4"))]
        )


def test_mixed_modes_across_batch_rejected(client: he.Client):
    with pytest.raises(ValueError, match="same kind"):
        make_scorer(client).submit(
            [
                he.Pair(a="text-a", b="text-b"),
                he.Pair(a=he.Media("https://x.test/a.png"), b=he.Media("https://x.test/b.png")),
            ]
        )


def test_text_pair_with_reference_rejected(client: he.Client):
    with pytest.raises(ValueError, match="Text pairs cannot carry"):
        make_scorer(client).submit(
            [he.Pair(a="x", b="y", reference=he.Media("https://x.test/r.png"))]
        )


def test_empty_batch_rejected(client: he.Client):
    with pytest.raises(ValueError, match="non-empty"):
        make_scorer(client).submit([])


def test_unshown_context_warns(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    with pytest.warns(UserWarning, match="context"):
        make_scorer(client).submit([he.Pair(a="x", b="y", context="hidden")])


# -- score mapping: media path (native comparison aggregation) ---------------


def test_media_scores_from_votes(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="comparison"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "context": None,
                    "media": [],
                    "votes": {"A": 7, "B": 3},
                    "total_responses": 10,
                    "consensus": "A",
                    "confidence": 0.7,
                    "agreement_rate": 0.7,
                    "weighted_votes": {"A": 6.5, "B": 2.8},
                }
            ]
        ),
    )
    scores = make_scorer(client).eval_batch(
        [he.Pair(a=he.Media("dp://a1/x.mp4"), b=he.Media("dp://b1/y.mp4"))]
    )
    assert len(scores) == 1
    score = scores[0]
    assert score.score == pytest.approx(0.7)  # P(prefer a) = votes A / total
    assert score.name == "HumanComparison"
    assert score.metadata["consensus"] == "A"
    assert score.metadata["agreement_rate"] == 0.7
    assert score.metadata["weighted_votes"] == {"A": 6.5, "B": 2.8}
    assert score.metadata["job_id"] == "job_test123"
    assert score.metadata["datapoint_index"] == 0
    assert score.error is None


def test_media_score_none_when_no_responses(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status())
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "votes": {"A": 0, "B": 0},
                    "total_responses": 0,
                    "consensus": None,
                    "confidence": None,
                    "agreement_rate": None,
                }
            ]
        ),
    )
    scores = make_scorer(client).eval_batch(
        [he.Pair(a=he.Media("dp://a1/x.png"), b=he.Media("dp://b1/y.png"))]
    )
    assert scores[0].score is None


# -- score mapping: text path (2-candidate ranking) --------------------------


def test_text_scores_from_average_ranks(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="ranking"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    # 8 of 10 annotators ranked "a" first: mean rank 1.2
                    "average_ranks": {"a": 1.2, "b": 1.8},
                    "ranking_order": ["a", "b"],
                    "total_responses": 10,
                }
            ],
            task_type="ranking",
        ),
    )
    scores = make_scorer(client).eval_batch([he.Pair(a="alpha", b="beta")])
    # P(prefer a) = 2 - mean_rank(a) = 0.8
    assert scores[0].score == pytest.approx(0.8)
    assert scores[0].metadata["consensus"] == "a"


def test_text_tie_maps_to_half(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="ranking"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "average_ranks": {"a": 1.5, "b": 1.5},
                    "ranking_order": ["a", "b"],
                    "total_responses": 10,
                }
            ],
            task_type="ranking",
        ),
    )
    scores = make_scorer(client).eval_batch([he.Pair(a="x", b="y")])
    assert scores[0].score == pytest.approx(0.5)
    assert scores[0].metadata["consensus"] == "tie"


# -- autoevals-style single call ---------------------------------------------


def test_call_scores_output_vs_expected(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="ranking"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "average_ranks": {"a": 1.0, "b": 2.0},
                    "ranking_order": ["a", "b"],
                    "total_responses": 5,
                }
            ],
            task_type="ranking",
        ),
    )
    scorer = he.HumanComparison("Better answer to: {context}", client=client)
    score = scorer(output="the answer", expected="another answer", input="the question")
    assert score.score == pytest.approx(1.0)  # everyone preferred `output`
    # The submitted datapoint carried the input as context.
    create_body = next(b for m, b in _bodies(api) if m == "POST /jobs")
    assert create_body["datapoints"][0]["context"] == "the question"


def _bodies(api: FakeAPI):
    import json as _json

    for request in api.requests:
        method_path = f"{request.method} {request.url.path.removeprefix('/data-labelling/v1')}"
        content = request.content
        yield method_path, (_json.loads(content) if content else None)

"""HumanRating: scale handling, text-vs-media subjects, normalization."""

from __future__ import annotations

import pytest
from conftest import FakeAPI, create_job_response, job_status, results_response

import humanevals as he

INSTRUCTION = "How helpful is this response?\n\n{context}"


def make_scorer(client: he.Client, **kwargs) -> he.HumanRating:
    return he.HumanRating(INSTRUCTION, client=client, **kwargs)


# -- configuration -----------------------------------------------------------


def test_scale_tuple_expands_to_inclusive_range(client: he.Client):
    assert make_scorer(client, scale=(1, 5)).scale == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_scale_explicit_list(client: he.Client):
    assert make_scorer(client, scale=[0, 0.5, 1]).scale == [0.0, 0.5, 1.0]


def test_scale_too_short_rejected(client: he.Client):
    with pytest.raises(ValueError, match="at least 2"):
        make_scorer(client, scale=[3])


def test_labels_keys_are_stringified(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = make_scorer(client, labels={1: "Poor", 5: "Excellent"})
    scorer.submit(["some response text"])
    options = api.body()["response_options"]
    assert options == {
        "scale": [1, 2, 3, 4, 5],
        "labels": {"1": "Poor", "5": "Excellent"},
    }


# -- payload construction ----------------------------------------------------


def test_text_subject_travels_as_context_with_empty_media(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    make_scorer(client).submit(["rate me please"])
    datapoint = api.body()["datapoints"][0]
    # Text-only rating requires an explicit empty media dict per the API.
    assert datapoint == {"media": {}, "context": "rate me please"}
    assert api.body()["task_type"] == "rating"


def test_media_subject_travels_under_subject_role(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = he.HumanRating("Rate the audio quality.", client=client)
    scorer.submit([he.Media("dp://abc123/clip.wav")])
    datapoint = api.body()["datapoints"][0]
    assert datapoint == {"media": {"subject": [{"url": "dp://abc123/clip.wav", "type": "audio"}]}}


def test_media_subject_with_context(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = he.HumanRating("Rate fidelity to the prompt: {context}", client=client)
    scorer.submit([he.RatingItem(subject=he.Media("dp://abc123/img.png"), context="a red fox")])
    assert api.body()["datapoints"][0]["context"] == "a red fox"


def test_media_subject_with_reference(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = he.HumanRating("How well does the edit preserve identity?", client=client)
    scorer.submit(
        [
            he.RatingItem(
                subject=he.Media("dp://edited123/shoe.png"),
                reference=he.Media("dp://original456/shoe.png"),
            )
        ]
    )

    assert api.body()["datapoints"][0] == {
        "media": {
            "subject": [{"url": "dp://edited123/shoe.png", "type": "image"}],
            "reference": [{"url": "dp://original456/shoe.png", "type": "image"}],
        }
    }


# -- validation --------------------------------------------------------------


def test_text_subject_requires_context_placeholder(client: he.Client):
    scorer = he.HumanRating("Rate this.", client=client)  # no {context}
    with pytest.raises(ValueError, match=r"\{context\}"):
        scorer.submit(["some text"])


def test_text_subject_with_extra_context_rejected(client: he.Client):
    with pytest.raises(ValueError, match="ambiguous"):
        make_scorer(client).submit([he.RatingItem(subject="text", context="extra")])


def test_text_subject_with_reference(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())

    make_scorer(client).submit(
        [
            he.RatingItem(
                subject="A red shoe with white laces.",
                reference=he.Media("dp://original456/shoe.png"),
            )
        ]
    )

    assert api.body()["datapoints"][0] == {
        "media": {"reference": [{"url": "dp://original456/shoe.png", "type": "image"}]},
        "context": "A red shoe with white laces.",
    }


def test_wrong_item_type_rejected(client: he.Client):
    with pytest.raises(TypeError, match="HumanRating items"):
        make_scorer(client).submit([42])


# -- score mapping -----------------------------------------------------------


def _serve_results(api: FakeAPI, row: dict) -> None:
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="rating"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response([{"datapoint_index": 0, **row}], task_type="rating"),
    )


def test_mean_is_normalized_to_unit_interval(client: he.Client, api: FakeAPI):
    _serve_results(
        api,
        {
            "mean": 4.2,
            "median": 4.0,
            "distribution": {"3": 1, "4": 2, "5": 2},
            "total_responses": 5,
            "weighted_mean": 4.1,
        },
    )
    scores = make_scorer(client).eval_batch(["text to rate"])
    # (4.2 - 1) / (5 - 1) = 0.8
    assert scores[0].score == pytest.approx(0.8)
    assert scores[0].metadata["mean"] == 4.2
    assert scores[0].metadata["median"] == 4.0
    assert scores[0].metadata["distribution"] == {"3": 1, "4": 2, "5": 2}
    assert scores[0].metadata["weighted_mean"] == 4.1
    assert scores[0].metadata["scale"] == [1, 2, 3, 4, 5]


def test_score_none_when_mean_missing(client: he.Client, api: FakeAPI):
    _serve_results(api, {"mean": None, "median": None, "distribution": {}, "total_responses": 0})
    scores = make_scorer(client).eval_batch(["text to rate"])
    assert scores[0].score is None


def test_custom_scale_normalization(client: he.Client, api: FakeAPI):
    _serve_results(api, {"mean": 7.0, "median": 7, "distribution": {"7": 3}, "total_responses": 3})
    scores = make_scorer(client, scale=(0, 10)).eval_batch(["text"])
    assert scores[0].score == pytest.approx(0.7)


def test_out_of_scale_mean_is_clamped(client: he.Client, api: FakeAPI):
    _serve_results(api, {"mean": 0.5, "distribution": {}, "total_responses": 1})
    scores = make_scorer(client).eval_batch(["text"])  # scale (1, 5)
    assert scores[0].score == 0.0

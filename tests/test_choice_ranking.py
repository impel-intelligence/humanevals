"""HumanMultipleChoice and HumanRanking: payloads, validation, score mapping."""

from __future__ import annotations

import pytest
from conftest import FakeAPI, create_job_response, job_status, results_response

import humanevals as he

# -- multiple choice: payload and validation ---------------------------------


def mc(client: he.Client, **kwargs) -> he.HumanMultipleChoice:
    return he.HumanMultipleChoice("Answer the question.", client=client, **kwargs)


def choice_item(**overrides) -> he.ChoiceItem:
    defaults: dict = {
        "question": "Which is a mammal?",
        "options": ["Trout", "Dolphin", "Falcon"],
        "expected": "Dolphin",
    }
    defaults.update(overrides)
    return he.ChoiceItem(**defaults)


def test_choice_payload_shape(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    mc(client).submit([choice_item()])
    body = api.body()
    assert body["task_type"] == "multiple_choice"
    assert body["response_options"] == {"mode": "single", "shuffle": True}
    assert body["datapoints"][0] == {
        "media": {
            "options": [
                {"id": "option_1", "text": "Trout"},
                {"id": "option_2", "text": "Dolphin"},
                {"id": "option_3", "text": "Falcon"},
            ]
        },
        "context": "Which is a mammal?",
    }


def test_choice_subject_media(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    mc(client).submit([choice_item(subject=he.Media("dp://abc/shot.png"))])
    assert api.body()["datapoints"][0]["media"]["subject"] == [
        {"url": "dp://abc/shot.png", "type": "image"}
    ]


def test_choice_https_subject_rejected(client: he.Client):
    with pytest.raises(ValueError, match="local file or dp://"):
        mc(client).submit([choice_item(subject=he.Media("https://cdn.example/shot.png"))])


def test_choice_validations(client: he.Client):
    with pytest.raises(ValueError, match="at least 2 options"):
        mc(client).submit([choice_item(options=["Only one"], expected=None)])
    with pytest.raises(ValueError, match="Duplicate options"):
        mc(client).submit([choice_item(options=["A", "A"], expected=None)])
    with pytest.raises(ValueError, match="not among the options"):
        mc(client).submit([choice_item(expected="Whale")])
    with pytest.raises(ValueError, match="question must be non-empty"):
        mc(client).submit([choice_item(question="  ")])


def test_choice_shuffle_disabled(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    mc(client, shuffle=False).submit([choice_item()])
    assert api.body()["response_options"] == {"mode": "single", "shuffle": False}


# -- multiple choice: scoring ------------------------------------------------


def test_choice_score_is_fraction_choosing_expected(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="multiple_choice"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "votes": {"option_1": 1, "option_2": 8, "option_3": 1},
                    "total_responses": 10,
                    "consensus": "option_2",
                    "confidence": 0.8,
                }
            ],
            task_type="multiple_choice",
        ),
    )
    scores = mc(client).eval_batch([choice_item()])
    assert scores[0].score == pytest.approx(0.8)
    assert scores[0].metadata["consensus_text"] == "Dolphin"
    assert scores[0].metadata["votes_by_text"] == {"Trout": 1, "Dolphin": 8, "Falcon": 1}


def test_choice_without_expected_scores_none_with_distribution(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="multiple_choice"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "votes": {"option_1": 4, "option_2": 6},
                    "total_responses": 10,
                    "consensus": "option_2",
                    "confidence": 0.6,
                }
            ],
            task_type="multiple_choice",
        ),
    )
    scores = mc(client).eval_batch([choice_item(options=["Yes", "No"], expected=None)])
    assert scores[0].score is None
    assert scores[0].error is None  # not an error: caller asked for labeling
    assert scores[0].metadata["votes_by_text"] == {"Yes": 4, "No": 6}


# -- ranking: payload and validation -----------------------------------------


def rank(client: he.Client, **kwargs) -> he.HumanRanking:
    return he.HumanRanking("Rank best to worst.", client=client, **kwargs)


def test_ranking_text_payload(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    rank(client).submit([he.RankingItem(candidates=["one", "two", "three"])])
    body = api.body()
    assert body["task_type"] == "ranking"
    assert body["datapoints"][0]["media"]["candidates"] == [
        {"id": "item_1", "text": "one"},
        {"id": "item_2", "text": "two"},
        {"id": "item_3", "text": "three"},
    ]


def test_ranking_media_payload(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    rank(client).submit(
        [he.RankingItem(candidates=[he.Media("dp://a/1.mp4"), he.Media("dp://b/2.mp4")])]
    )
    assert api.body()["datapoints"][0]["media"]["candidates"] == [
        {"url": "dp://a/1.mp4", "type": "video"},
        {"url": "dp://b/2.mp4", "type": "video"},
    ]


def test_ranking_validations(client: he.Client):
    with pytest.raises(ValueError, match="at least 2 candidates"):
        rank(client).submit([he.RankingItem(candidates=["solo"])])
    with pytest.raises(ValueError, match="not mixed"):
        rank(client).submit([he.RankingItem(candidates=["text", he.Media("dp://a/x.png")])])
    with pytest.raises(ValueError, match="share one type"):
        rank(client).submit(
            [he.RankingItem(candidates=[he.Media("dp://a/x.png"), he.Media("dp://a/y.mp4")])]
        )
    with pytest.raises(ValueError, match="permutation"):
        rank(client).submit([he.RankingItem(candidates=["a", "b"], expected_order=[0, 2])])


def test_bare_list_is_coerced_to_ranking_item(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    rank(client).submit([["one", "two"]])
    assert api.body()["datapoints"][0]["media"]["candidates"][0] == {"id": "item_1", "text": "one"}


# -- ranking: scoring --------------------------------------------------------


def _serve_ranking(api: FakeAPI, row: dict) -> None:
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="ranking"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response([{"datapoint_index": 0, **row}], task_type="ranking"),
    )


def test_ranking_perfect_agreement_scores_one(client: he.Client, api: FakeAPI):
    _serve_ranking(
        api,
        {
            "average_ranks": {"item_1": 1.1, "item_2": 2.0, "item_3": 2.9},
            "ranking_order": ["item_1", "item_2", "item_3"],
            "total_responses": 5,
        },
    )
    scores = rank(client).eval_batch(
        [he.RankingItem(candidates=["gold", "silver", "bronze"], expected_order=[0, 1, 2])]
    )
    assert scores[0].score == pytest.approx(1.0)
    assert scores[0].metadata["kendall_tau"] == pytest.approx(1.0)
    assert scores[0].metadata["ranking_order_texts"] == ["gold", "silver", "bronze"]


def test_ranking_reversed_order_scores_zero(client: he.Client, api: FakeAPI):
    _serve_ranking(
        api,
        {
            "average_ranks": {"item_1": 3.0, "item_2": 2.0, "item_3": 1.0},
            "ranking_order": ["item_3", "item_2", "item_1"],
            "total_responses": 5,
        },
    )
    scores = rank(client).eval_batch(
        [he.RankingItem(candidates=["a", "b", "c"], expected_order=[0, 1, 2])]
    )
    assert scores[0].score == pytest.approx(0.0)


def test_ranking_without_expected_scores_none(client: he.Client, api: FakeAPI):
    _serve_ranking(
        api,
        {
            "average_ranks": {"item_1": 1.5, "item_2": 1.5},
            "ranking_order": ["item_1", "item_2"],
            "total_responses": 4,
        },
    )
    scores = rank(client).eval_batch([he.RankingItem(candidates=["x", "y"])])
    assert scores[0].score is None
    assert scores[0].metadata["ranking_order"] == ["item_1", "item_2"]


def test_ranking_media_ids_use_server_minting_convention(client: he.Client, api: FakeAPI):
    _serve_ranking(
        api,
        {
            "average_ranks": {"video_1": 2.0, "video_2": 1.0},
            "ranking_order": ["video_2", "video_1"],
            "total_responses": 5,
        },
    )
    scores = rank(client).eval_batch(
        [
            he.RankingItem(
                candidates=[he.Media("dp://a/1.mp4"), he.Media("dp://b/2.mp4")],
                expected_order=[1, 0],
            )
        ]
    )
    # Expected order [1, 0] = video_2 best, matching the humans exactly.
    assert scores[0].score == pytest.approx(1.0)

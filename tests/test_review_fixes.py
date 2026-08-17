"""Regression tests for issues found in the pre-release review."""

from __future__ import annotations

import warnings

import httpx
import pytest
from conftest import FakeAPI, create_job_response, job_status, results_response

import humanevals as he
from humanevals.job import EvalJob

# -- idempotency golden value ------------------------------------------------


def test_content_name_golden_value(client: he.Client):
    """The content hash is the anti-double-charge idempotency key.

    If this test breaks, the canonicalization changed: identical evals
    submitted by older library versions will re-run (and re-charge) under
    new names. Only change the expected value together with a bump of the
    'humanevals:1:' hash-input version, and note it in the CHANGELOG.
    """
    scorer = he.HumanComparison("Which is better?", client=client)
    pairs = [he.Pair(a="alpha", b="beta"), he.Pair(a="gamma", b="delta")]
    assert scorer._content_name(pairs) == "he-231be721d1aff1cd02f0"


# -- submit guards -----------------------------------------------------------


def test_bare_string_items_rejected(client: he.Client):
    scorer = he.HumanRating("Rate: {context}", client=client)
    with pytest.raises(TypeError, match="not a single str"):
        scorer.submit("a long response that must not become per-character items")
    with pytest.raises(TypeError, match="not a single str"):
        scorer.eval_batch("another response")


def test_max_credits_with_range_filter_rejected(client: he.Client):
    scorer = he.HumanComparison(
        "Which?",
        client=client,
        annotator_filter={"median_household_income": {"gte": 50000}},
    )
    with pytest.raises(ValueError, match="numeric-range"):
        scorer.submit([he.Pair(a="x", b="y")], max_credits=100)


def test_replay_name_collision_detected(client: he.Client, api: FakeAPI):
    # pricing: null marks a replay; 5 existing datapoints vs 1 submitted.
    api.add("POST", "/jobs", create_job_response(pricing=None, total_datapoints=5))
    scorer = he.HumanComparison("Which?", client=client)
    with pytest.raises(ValueError, match="already belongs to a different job"):
        scorer.submit([he.Pair(a="x", b="y")], name="reused-name")


def test_replay_with_matching_count_accepted(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response(pricing=None, total_datapoints=1))
    scorer = he.HumanComparison("Which?", client=client)
    job = scorer.submit([he.Pair(a="x", b="y")], name="reused-name")
    assert job.job_id == "job_test123"


# -- reattach counting -------------------------------------------------------


def test_reattach_without_items_reports_trailing_failures(client: he.Client, api: FakeAPI):
    """Failed datapoints are absent from /results; the job count fills the gap."""
    api.add(
        "GET",
        "/jobs/job_abc",
        job_status(
            job_id="job_abc",
            total_datapoints=2,
            errors=[{"datapoint_index": 1, "error": "Unresolved media ref: dp://typo/x.mp4"}],
        ),
    )
    api.add(
        "GET",
        "/jobs/job_abc/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "average_ranks": {"a": 1.0, "b": 2.0},
                    "ranking_order": ["a", "b"],
                    "total_responses": 5,
                }
            ],
            job_id="job_abc",
            task_type="ranking",
        ),
    )
    scorer = he.HumanComparison("Which?", client=client)
    job = EvalJob.attach(client, "job_abc", scorer=scorer)  # no items supplied
    scores = job.scores()
    assert len(scores) == 2
    assert scores[0].score == pytest.approx(1.0)
    assert scores[1].score is None
    assert "Unresolved media ref" in scores[1].error


def test_attach_accepts_explicit_n_items(client: he.Client, api: FakeAPI):
    api.add("GET", "/jobs/job_abc", job_status(job_id="job_abc", total_datapoints=1))
    api.add("GET", "/jobs/job_abc/results", results_response([], job_id="job_abc"))
    scorer = he.HumanComparison("Which?", client=client)
    job = EvalJob.attach(client, "job_abc", scorer=scorer, n_items=3)
    assert len(job.scores()) == 3


# -- EvalJob lifecycle -------------------------------------------------------


def test_eval_job_cancel(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add(
        "POST",
        "/jobs/job_test123/cancel",
        {"job_id": "job_test123", "status": "cancelled", "is_paused": True, "cost_credits": 10},
    )
    api.add("GET", "/jobs/job_test123", job_status(status="cancelled"))
    scorer = he.HumanComparison("Which?", client=client)
    job = scorer.submit([he.Pair(a="x", b="y")])
    assert job.cancel().status == "cancelled"
    assert "POST /jobs/job_test123/cancel" in api.paths()


def test_eval_job_complete(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add(
        "POST",
        "/jobs/job_test123/complete",
        {"job_id": "job_test123", "status": "completed", "is_paused": True, "cost_credits": 10},
    )
    api.add("GET", "/jobs/job_test123", job_status(status="completed"))
    scorer = he.HumanComparison("Which?", client=client)
    job = scorer.submit([he.Pair(a="x", b="y")])
    assert job.complete().status == "completed"
    assert "POST /jobs/job_test123/complete" in api.paths()


# -- HumanRating call semantics and scale validation -------------------------


def test_rating_call_with_text_output_works(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="rating"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [{"datapoint_index": 0, "mean": 3.0, "distribution": {"3": 5}, "total_responses": 5}],
            task_type="rating",
        ),
    )
    scorer = he.HumanRating("Rate this: {context}", client=client)
    assert scorer("some response").score == pytest.approx(0.5)


def test_rating_call_text_with_input_raises_clearly(client: he.Client):
    scorer = he.HumanRating("Rate this: {context}", client=client)
    with pytest.raises(ValueError, match="single per-item context slot"):
        scorer("some response", input="the question")


def test_rating_call_media_with_input_sends_context(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="rating"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [{"datapoint_index": 0, "mean": 5.0, "distribution": {"5": 5}, "total_responses": 5}],
            task_type="rating",
        ),
    )
    scorer = he.HumanRating("Rate fidelity to: {context}", client=client)
    score = scorer(he.Media("dp://abc/img.png"), input="a red fox")
    assert score.score == pytest.approx(1.0)
    assert api.body(0)["datapoints"][0]["context"] == "a red fox"


@pytest.mark.parametrize(
    ("scale", "message"),
    [
        ((5, 1), "high > low"),
        ((1.5, 4.5), "must use integers"),
        ([3, 3], "unique"),
        ([2], "at least 2"),
    ],
)
def test_rating_degenerate_scales_rejected(client: he.Client, scale, message):
    with pytest.raises(ValueError, match=message):
        he.HumanRating("Rate: {context}", scale=scale, client=client)


def test_rating_two_value_list_is_two_options(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = he.HumanRating("Rate: {context}", scale=[1, 5], client=client)
    scorer.submit(["text"])
    assert api.body()["response_options"]["scale"] == [1, 5]


# -- zero-response guards ----------------------------------------------------


def test_choice_zero_responses_scores_none(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="multiple_choice"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [{"datapoint_index": 0, "votes": {}, "total_responses": 0, "consensus": None}],
            task_type="multiple_choice",
        ),
    )
    scorer = he.HumanMultipleChoice("Answer.", client=client)
    scores = scorer.eval_batch([he.ChoiceItem(question="Q?", options=["A", "B"], expected="A")])
    assert scores[0].score is None


def test_ranking_zero_responses_scores_none(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="ranking"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [
                {
                    "datapoint_index": 0,
                    "average_ranks": {},
                    "ranking_order": [],
                    "total_responses": 0,
                }
            ],
            task_type="ranking",
        ),
    )
    scorer = he.HumanRanking("Rank.", client=client)
    scores = scorer.eval_batch([he.RankingItem(candidates=["x", "y"], expected_order=[0, 1])])
    assert scores[0].score is None


# -- network errors ----------------------------------------------------------


class FlakyTransport(httpx.BaseTransport):
    """Raises connect errors for the first `failures` requests, then delegates."""

    def __init__(self, inner: httpx.BaseTransport, failures: int) -> None:
        self.inner = inner
        self.failures = failures
        self.attempts = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise httpx.ConnectError("connection refused", request=request)
        return self.inner.handle_request(request)


def test_network_errors_are_retried(api: FakeAPI, _no_real_sleep):
    api.add("GET", "/jobs/job_x", {"job_id": "job_x", "status": "active"})
    transport = FlakyTransport(api.transport(), failures=2)
    client = he.Client(api_key="k", transport=transport)
    assert client.get_job("job_x")["status"] == "active"
    assert _no_real_sleep == [1.0, 2.0]


def test_network_errors_raise_typed_after_retries(api: FakeAPI):
    transport = FlakyTransport(api.transport(), failures=10)
    client = he.Client(api_key="k", transport=transport, max_retries=2)
    with pytest.raises(he.NetworkError, match="Could not reach"):
        client.get_job("job_x")
    assert transport.attempts == 3


def test_upload_network_errors_are_not_retried(api: FakeAPI, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"bytes")
    transport = FlakyTransport(api.transport(), failures=1)
    client = he.Client(api_key="k", transport=transport)
    with pytest.raises(he.NetworkError):
        client.upload_media([clip])
    assert transport.attempts == 1  # a replayed upload would duplicate media


# -- transport robustness ----------------------------------------------------


@pytest.mark.parametrize("header", ["nan", "inf", "soon", ""])
def test_garbage_retry_after_falls_back_to_backoff(api: FakeAPI, _no_real_sleep, header):
    headers = {"Retry-After": header}
    api.add("GET", "/jobs/job_x", {"detail": "Rate limit exceeded"}, status=429, headers=headers)
    api.add("GET", "/jobs/job_x", {"job_id": "job_x", "status": "active"})
    client = he.Client(api_key="k", transport=api.transport())
    assert client.get_job("job_x")["status"] == "active"
    assert _no_real_sleep == [1.0]  # 2**0, not a crash


def test_non_object_json_error_body(api: FakeAPI):
    api._routes[("GET", "/jobs/job_x")] = [httpx.Response(500, json=["weird", "body"])]
    client = he.Client(api_key="k", transport=api.transport(), max_retries=0)
    with pytest.raises(he.ServerError):
        client.get_job("job_x")


def test_api_key_is_stripped(api: FakeAPI):
    api.add("GET", "/billing/balance", {"available_credits": 1})
    client = he.Client(api_key="  dp_live_x \n", transport=api.transport())
    client.balance()
    assert api.sent().headers["X-API-Key"] == "dp_live_x"


def test_multi_file_upload_maps_refs_by_order(api: FakeAPI, tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.png"
    a.write_bytes(b"video-bytes")
    b.write_bytes(b"image-bytes")
    api.add(
        "POST",
        "/media",
        {
            "media": [
                {"filename": "a.mp4", "media_ref": "dp://ref-a/a.mp4", "type": "video"},
                {"filename": "b.png", "media_ref": "dp://ref-b/b.png", "type": "image"},
            ]
        },
    )
    client = he.Client(api_key="k", transport=api.transport())
    uploaded = client.upload_media([a, b])
    assert [u["media_ref"] for u in uploaded] == ["dp://ref-a/a.mp4", "dp://ref-b/b.png"]


def test_results_pagination_stops_on_stalled_server(client: he.Client, api: FakeAPI):
    """total_results ahead of returned rows must not loop forever."""
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123/results", results_response([], total_results=5))
    scorer = he.HumanComparison("Which?", client=client)
    job = scorer.submit([he.Pair(a="x", b="y")])
    assert job.results() == []


# -- warning attribution -----------------------------------------------------


def test_context_warning_points_at_caller(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    api.add("GET", "/jobs/job_test123", job_status(task_type="ranking"))
    api.add("GET", "/jobs/job_test123/results", results_response([], task_type="ranking"))
    scorer = he.HumanComparison("Which?", client=client)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scorer.submit([he.Pair(a="x", b="y", context="hidden")])
        submit_warning = caught[-1]
        scorer.eval_batch([he.Pair(a="x", b="y", context="hidden")])
        batch_warning = caught[-1]
    assert submit_warning.filename == __file__
    assert batch_warning.filename == __file__

"""EvalJob lifecycle: polling, waiting, failure handling, reattachment."""

from __future__ import annotations

import pytest
from conftest import FakeAPI, create_job_response, job_status, results_response

import humanevals as he
from humanevals.job import EvalJob


def submitted_job(client: he.Client, api: FakeAPI, n: int = 1) -> EvalJob:
    api.add("POST", "/jobs", create_job_response())
    scorer = he.HumanComparison("Which?", client=client)
    return scorer.submit([he.Pair(a=f"a{i}", b=f"b{i}") for i in range(n)])


def ranking_row(index: int, rank_a: float = 1.0) -> dict:
    return {
        "datapoint_index": index,
        "average_ranks": {"a": rank_a, "b": 3.0 - rank_a},
        "ranking_order": ["a", "b"] if rank_a <= 1.5 else ["b", "a"],
        "total_responses": 5,
    }


def test_wait_polls_until_terminal(client: he.Client, api: FakeAPI, _no_real_sleep):
    api.add("GET", "/jobs/job_test123", job_status(status="processing"))
    api.add("GET", "/jobs/job_test123", job_status(status="active"))
    api.add("GET", "/jobs/job_test123", job_status(status="completed"))
    job = submitted_job(client, api)
    seen: list[str] = []
    final = job.wait(on_progress=lambda p: seen.append(p.status))
    assert final.status == "completed"
    assert seen == ["processing", "active", "completed"]
    assert len(_no_real_sleep) == 2  # slept between the three polls


@pytest.mark.parametrize("terminal", ["completed", "cancelled", "failed", "blocked"])
def test_wait_stops_on_any_terminal_status(client: he.Client, api: FakeAPI, terminal: str):
    api.add("GET", "/jobs/job_test123", job_status(status=terminal))
    job = submitted_job(client, api)
    assert job.wait().status == terminal


def test_wait_timeout_raises_with_reattach_hint(client: he.Client, api: FakeAPI, monkeypatch):
    api.add("GET", "/jobs/job_test123", job_status(status="active"))
    job = submitted_job(client, api)
    clock = iter([0.0, 0.0, 100.0, 100.0, 200.0, 200.0])
    monkeypatch.setattr("humanevals.job.time.monotonic", lambda: next(clock))
    with pytest.raises(he.PollTimeoutError, match="job_test123") as info:
        job.wait(timeout=50)
    assert info.value.job_id == "job_test123"


def test_scores_maps_rows_to_items_in_order(client: he.Client, api: FakeAPI):
    api.add("GET", "/jobs/job_test123", job_status(status="completed", total_datapoints=3))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response(
            [ranking_row(0, 1.0), ranking_row(1, 2.0), ranking_row(2, 1.5)],
            task_type="ranking",
        ),
    )
    job = submitted_job(client, api, n=3)
    scores = job.scores()
    assert [s.score for s in scores] == [
        pytest.approx(1.0),
        pytest.approx(0.0),
        pytest.approx(0.5),
    ]
    assert [s.metadata["datapoint_index"] for s in scores] == [0, 1, 2]


def test_failed_datapoint_becomes_error_score_not_exception(client: he.Client, api: FakeAPI):
    api.add(
        "GET",
        "/jobs/job_test123",
        job_status(
            status="completed",
            total_datapoints=2,
            errors=[{"datapoint_index": 1, "error": "Unresolved media ref: dp://typo/x.mp4"}],
        ),
    )
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response([ranking_row(0)], task_type="ranking"),
    )
    job = submitted_job(client, api, n=2)
    scores = job.scores()
    assert scores[0].score == pytest.approx(1.0)
    assert scores[1].score is None
    assert "Unresolved media ref" in scores[1].error
    assert scores[1].metadata["datapoint_index"] == 1


def test_wholly_failed_job_raises(client: he.Client, api: FakeAPI):
    api.add(
        "GET",
        "/jobs/job_test123",
        job_status(
            status="failed",
            errors=[{"datapoint_index": 0, "error": "Unresolved media ref: dp://nope/a.mp4"}],
        ),
    )
    api.add("GET", "/jobs/job_test123/results", results_response([], task_type="ranking"))
    job = submitted_job(client, api)
    with pytest.raises(he.JobFailedError, match="Unresolved media ref"):
        job.scores()


def test_scores_without_wait_peeks_at_partial_results(client: he.Client, api: FakeAPI):
    api.add("GET", "/jobs/job_test123", job_status(status="active"))
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response([ranking_row(0)], task_type="ranking"),
    )
    job = submitted_job(client, api)
    scores = job.scores(wait=False)
    assert scores[0].score == pytest.approx(1.0)
    # Only one status call, no polling loop.
    assert api.paths().count("GET /jobs/job_test123") == 1


def test_results_paginates_when_needed(client: he.Client, api: FakeAPI):
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response([ranking_row(0)], total_results=2),
    )
    api.add(
        "GET",
        "/jobs/job_test123/results",
        results_response([ranking_row(1)], total_results=2),
    )
    job = submitted_job(client, api)
    rows = job.results(per_page=1)
    assert [r["datapoint_index"] for r in rows] == [0, 1]


def test_attach_without_scorer_gives_progress_but_not_scores(client: he.Client, api: FakeAPI):
    api.add("GET", "/jobs/job_abc", job_status(job_id="job_abc"))
    job = EvalJob.attach(client, "job_abc")
    assert job.progress().status == "completed"
    with pytest.raises(RuntimeError, match="without a scorer"):
        job.scores()


def test_attach_with_scorer_and_items_scores_fully(client: he.Client, api: FakeAPI):
    api.add("GET", "/jobs/job_abc", job_status(job_id="job_abc"))
    api.add(
        "GET",
        "/jobs/job_abc/results",
        results_response([ranking_row(0)], job_id="job_abc", task_type="ranking"),
    )
    scorer = he.HumanComparison("Which?", client=client)
    job = EvalJob.attach(client, "job_abc", scorer=scorer, items=[he.Pair(a="x", b="y")])
    scores = job.scores()
    assert scores[0].score == pytest.approx(1.0)
    assert scores[0].metadata["job_id"] == "job_abc"


def test_progress_snapshot_fields(client: he.Client, api: FakeAPI):
    api.add(
        "GET",
        "/jobs/job_test123",
        job_status(
            status="active",
            total_datapoints=4,
            completed_datapoints=1,
            total_responses=13,
            cost_credits=65,
            refundable_credits=35,
            is_paused=True,
        ),
    )
    job = submitted_job(client, api)
    progress = job.progress()
    assert progress.total_datapoints == 4
    assert progress.completed_datapoints == 1
    assert progress.total_responses == 13
    assert progress.cost_credits == 65
    assert progress.refundable_credits == 35
    assert progress.is_paused is True
    assert not progress.is_terminal
    assert progress.raw["name"] == "he-test"

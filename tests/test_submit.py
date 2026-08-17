"""Submission mechanics: idempotent naming, budgets, estimates."""

from __future__ import annotations

import pytest
from conftest import FakeAPI, create_job_response, quote_response

import humanevals as he


def scorer_for(client: he.Client, **kwargs) -> he.HumanComparison:
    return he.HumanComparison("Which is better?", client=client, **kwargs)


PAIRS = [he.Pair(a="alpha", b="beta"), he.Pair(a="gamma", b="delta")]


# -- content-derived idempotent names ----------------------------------------


def test_same_content_produces_same_name(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = scorer_for(client)
    job1 = scorer.submit(PAIRS)
    job2 = scorer.submit(list(PAIRS))
    assert job1.name == job2.name
    assert job1.name.startswith("he-")
    # Both creates sent the same name, so the API replays the job (no double charge).
    assert api.body(-1)["name"] == api.body(-2)["name"]


def test_different_items_produce_different_names(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = scorer_for(client)
    name1 = scorer.submit(PAIRS).name
    name2 = scorer.submit([he.Pair(a="alpha", b="CHANGED")]).name
    assert name1 != name2


def test_config_changes_produce_different_names(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    assert (
        scorer_for(client, responses_per_item=5).submit(PAIRS).name
        != scorer_for(client, responses_per_item=9).submit(PAIRS).name
    )
    assert (
        he.HumanComparison("Which is better?", client=client).submit(PAIRS).name
        != he.HumanComparison("Which is worse?", client=client).submit(PAIRS).name
    )
    assert (
        scorer_for(client, sandbox=True).submit(PAIRS).name != scorer_for(client).submit(PAIRS).name
    )


def test_local_file_naming_uses_content_hash(client: he.Client, api: FakeAPI, tmp_path):
    """Job names stay stable across re-runs even though re-uploads mint new refs."""
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"content-a")
    video_b.write_bytes(b"content-b")
    api.add(
        "POST",
        "/media",
        {
            "media": [
                {"media_ref": "dp://ref1/a.mp4", "media_id": "1", "type": "video"},
                {"media_ref": "dp://ref2/b.mp4", "media_id": "2", "type": "video"},
            ]
        },
    )
    api.add("POST", "/jobs", create_job_response())
    scorer = scorer_for(client)
    pair = [he.Pair(a=he.Media(video_a), b=he.Media(video_b))]

    name1 = scorer.submit(pair).name
    name2 = scorer.submit(pair).name
    assert name1 == name2

    # Changing file *content* changes the name even at the same path.
    client._media_refs.clear()
    video_a.write_bytes(b"different-content")
    assert scorer.submit(pair).name != name1


def test_explicit_name_wins(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    job = scorer_for(client).submit(PAIRS, name="my-eval-run-7")
    assert job.name == "my-eval-run-7"
    assert api.body()["name"] == "my-eval-run-7"


def test_fresh_uniquifies_the_name(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer = scorer_for(client)
    base = scorer.submit(PAIRS).name
    fresh = scorer.submit(PAIRS, fresh=True).name
    assert fresh != base
    assert fresh.startswith(base + "-")


# -- budgets and estimates ---------------------------------------------------


def test_max_credits_blocks_over_budget_submission(client: he.Client, api: FakeAPI):
    api.add("POST", "/billing/pricing/quote", quote_response(5))
    scorer = scorer_for(client)  # 2 items x 5 responses x 5 credits = 50
    with pytest.raises(he.BudgetExceededError) as info:
        scorer.submit(PAIRS, max_credits=49)
    assert info.value.estimated_credits == 50
    assert info.value.max_credits == 49
    # Nothing was submitted.
    assert "POST /jobs" not in api.paths()


def test_max_credits_allows_within_budget(client: he.Client, api: FakeAPI):
    api.add("POST", "/billing/pricing/quote", quote_response(5))
    api.add("POST", "/jobs", create_job_response())
    scorer_for(client).submit(PAIRS, max_credits=50)
    assert "POST /jobs" in api.paths()


def test_estimate_credits_uses_quote_rate(client: he.Client, api: FakeAPI):
    api.add("POST", "/billing/pricing/quote", quote_response(8))
    scorer = scorer_for(client, responses_per_item=10)
    assert scorer.estimate_credits(PAIRS) == 2 * 10 * 8


def test_sandbox_estimates_zero_and_skips_quote(client: he.Client, api: FakeAPI):
    scorer = scorer_for(client, sandbox=True)
    assert scorer.estimate_credits(PAIRS) == 0
    assert api.paths() == []  # no network call


def test_sandbox_skips_budget_check(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    scorer_for(client, sandbox=True).submit(PAIRS, max_credits=0)
    assert api.paths() == ["POST /jobs"]


def test_annotator_filter_is_passed_through(client: he.Client, api: FakeAPI):
    api.add("POST", "/jobs", create_job_response())
    audience = {"country": ["US", "CA"], "age_range": ["25-34"]}
    scorer_for(client, annotator_filter=audience).submit(PAIRS)
    assert api.body()["annotator_filter"] == audience


def test_instruction_required(client: he.Client):
    with pytest.raises(ValueError, match="non-empty"):
        he.HumanComparison("   ", client=client)

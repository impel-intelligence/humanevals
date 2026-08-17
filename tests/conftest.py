"""Shared test harness: a fake Datapoint API served through httpx.MockTransport.

No test in this suite touches the network. `FakeAPI` maps (method, path)
routes to queued canned responses (shapes recorded from the real API) and
logs every request so tests can assert on exact payloads.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from humanevals import Client

BASE_PATH = "/data-labelling/v1"


class FakeAPI:
    """Canned-response fake for the Datapoint API."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._routes: dict[tuple[str, str], list[httpx.Response]] = {}

    def add(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Queue one response for (method, path).

        Multiple `add()` calls for the same route are served in order; the
        final one is sticky (repeats forever), so a single `add()` behaves
        like a fixed endpoint.
        """
        response = httpx.Response(status, json=json_body, headers=headers)
        self._routes.setdefault((method.upper(), path), []).append(response)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix(BASE_PATH)
        queue = self._routes.get((request.method, path))
        if not queue:
            raise AssertionError(f"Unexpected request: {request.method} {path}")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    # -- request inspection helpers -----------------------------------------

    def sent(self, index: int = -1) -> httpx.Request:
        """The index-th request sent (default: most recent)."""
        return self.requests[index]

    def body(self, index: int = -1) -> Any:
        """Parsed JSON body of the index-th request."""
        return json.loads(self.requests[index].content)

    def paths(self) -> list[str]:
        """Method+path of every request, in order."""
        return [f"{r.method} {r.url.path.removeprefix(BASE_PATH)}" for r in self.requests]


@pytest.fixture()
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture()
def client(api: FakeAPI) -> Client:
    return Client(api_key="dp_live_" + "0" * 48, transport=api.transport())


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture sleeps instead of actually pausing (retries/polling)."""
    slept: list[float] = []
    monkeypatch.setattr("humanevals.client.time.sleep", slept.append)
    monkeypatch.setattr("humanevals.job.time.sleep", slept.append)
    return slept


# -- canned response builders (shapes recorded from the API contract) -------


def create_job_response(job_id: str = "job_test123", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": job_id,
        "status": "processing",
        "total_datapoints": 1,
        "estimated_cost_credits": 50,
        "credits_per_response": 5,
        "pricing": {
            "base_credits_per_response": 5,
            "demographic_surcharge_credits": 0,
            "geo_surcharge_credits": 0,
            "credits_per_response": 5,
        },
        "created_at": "None",  # literal string on fresh creation, per API
    }
    base.update(overrides)
    return base


def job_status(
    job_id: str = "job_test123",
    status: str = "completed",
    *,
    total_datapoints: int = 1,
    errors: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": job_id,
        "name": "he-test",
        "status": status,
        "task_type": "comparison",
        "serving_environment": "prod",
        "total_datapoints": total_datapoints,
        "processing_datapoints": 0,
        "ready_datapoints": total_datapoints,
        "completed_datapoints": total_datapoints if status == "completed" else 0,
        "failed_datapoints": 0,
        "blocked_datapoints": 0,
        "total_responses": 5 * total_datapoints,
        "max_responses_per_datapoint": 5,
        "cost_credits": 25,
        "credits_per_response": 5,
        "refundable_credits": 0,
        "created_at": "2026-04-21 12:34:56.123456+00:00",
        "errors": errors or [],
        "response_options": None,
        "annotator_distribution": None,
        "annotator_filter": None,
        "is_paused": False,
    }
    base.update(overrides)
    return base


def results_response(
    rows: list[dict[str, Any]],
    *,
    job_id: str = "job_test123",
    task_type: str = "comparison",
    **overrides: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": job_id,
        "status": "completed",
        "task_type": task_type,
        "page": 1,
        "per_page": max(len(rows), 100),
        "total_results": len(rows),
        "results": rows,
    }
    base.update(overrides)
    return base


def quote_response(credits_per_response: int = 5) -> dict[str, Any]:
    return {
        "base_credits_per_response": credits_per_response,
        "demographic_surcharge_credits": 0,
        "geo_surcharge_credits": 0,
        "credits_per_response": credits_per_response,
    }

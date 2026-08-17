"""EvalJob: handle to one submitted evaluation job.

A scorer's ``submit()`` returns an :class:`EvalJob` immediately; the job
then runs server-side while human annotators respond. The handle polls
progress, waits for completion, and converts the job's aggregated results
into per-item :class:`~humanevals.Score` objects.

Handles survive process restarts: persist ``job.job_id`` and reattach later
with :meth:`EvalJob.attach`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .client import Client
from .exceptions import JobFailedError, PollTimeoutError
from .types import JobProgress, Score

if TYPE_CHECKING:
    from .scorers import HumanScorer

__all__ = ["EvalJob"]

#: Adaptive polling: start fast, back off toward a ceiling.
_POLL_INITIAL_S = 5.0
_POLL_BACKOFF = 1.5
_POLL_MAX_S = 60.0


class EvalJob:
    """A submitted evaluation running on the Datapoint annotator pool.

    Not constructed directly; obtain one from a scorer's ``submit()`` or
    from :meth:`attach`.

    Attributes:
        job_id: Datapoint job id (``job_...``). Persist this to reattach.
        name: The job's idempotency name.
        client: The :class:`~humanevals.Client` used for API calls.

    """

    def __init__(
        self,
        client: Client,
        job_id: str,
        *,
        name: str | None = None,
        scorer: HumanScorer | None = None,
        items: list[Any] | None = None,
        n_items: int | None = None,
    ) -> None:
        self.client = client
        self.job_id = job_id
        self.name = name
        self._scorer = scorer
        self._items = items
        self._n_items = n_items if n_items is not None else (len(items) if items else None)

    @classmethod
    def attach(
        cls,
        client: Client,
        job_id: str,
        *,
        scorer: HumanScorer | None = None,
        items: list[Any] | None = None,
        n_items: int | None = None,
    ) -> EvalJob:
        """Reattach to an existing job by id.

        Pass the same ``scorer`` (same configuration) that created the job
        to get ``scores()`` back. Also re-supply ``items`` (in the original
        order) if the scorer needs them to score: expected answers for
        multiple choice, expected orders for ranking, option/candidate
        texts for readable metadata. Without items, ``n_items`` may be
        given instead (it defaults to the job's own datapoint count).
        Without a scorer, only ``progress()``, ``results()``, ``cancel()``,
        and ``complete()`` are available.
        """
        return cls(client, job_id, scorer=scorer, items=items, n_items=n_items)

    def __repr__(self) -> str:
        return f"EvalJob(job_id={self.job_id!r}, name={self.name!r})"

    # -- progress ------------------------------------------------------------

    def progress(self) -> JobProgress:
        """Fetch a fresh progress snapshot from the API."""
        return JobProgress.from_api(self.client.get_job(self.job_id))

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
        on_progress: Callable[[JobProgress], None] | None = None,
    ) -> JobProgress:
        """Block until the job reaches a terminal state; return the final snapshot.

        Human evaluation takes minutes to hours depending on audience and
        job size. Call this from code that is allowed to be patient, or
        skip it and reattach later via :meth:`attach`.

        Args:
            timeout: Give up after this many seconds, raising
                :class:`PollTimeoutError`. ``None`` (default) waits
                indefinitely. Timing out does not stop the job.
            poll_interval: Fixed seconds between polls. Default is adaptive:
                5s growing 1.5x per poll up to 60s.
            on_progress: Called with each :class:`JobProgress` snapshot,
                handy for progress bars/logging.

        """
        deadline = None if timeout is None else time.monotonic() + timeout
        interval = poll_interval if poll_interval is not None else _POLL_INITIAL_S
        while True:
            snapshot = self.progress()
            if on_progress is not None:
                on_progress(snapshot)
            if snapshot.is_terminal:
                return snapshot
            if deadline is not None and time.monotonic() >= deadline:
                raise PollTimeoutError(self.job_id, timeout or 0.0)
            sleep_for = interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(deadline - time.monotonic(), 0.0))
            time.sleep(sleep_for)
            if poll_interval is None:
                interval = min(interval * _POLL_BACKOFF, _POLL_MAX_S)

    # -- lifecycle -----------------------------------------------------------

    def cancel(self) -> JobProgress:
        """Cancel the job (irreversible) and return the final snapshot.

        Collected responses are billed; the unspent reserve is refunded.
        No report is generated for cancelled jobs.
        """
        self.client.cancel_job(self.job_id)
        return self.progress()

    def complete(self) -> JobProgress:
        """End the job early, keeping (and paying for) collected responses.

        Unlike :meth:`cancel`, the job settles as ``completed``: partial
        results stay available and ``scores()`` works on what was gathered.
        """
        self.client.complete_job(self.job_id)
        return self.progress()

    # -- results -------------------------------------------------------------

    def results(self, *, per_page: int | None = None) -> list[dict[str, Any]]:
        """Fetch the raw aggregated result rows (one per ready datapoint).

        Rows are the API's ``DatapointResult`` objects with stable ids.
        Most callers want :meth:`scores` instead.
        """
        per_page = per_page or max(self._n_items or 0, 100)
        payload = self.client.get_results(self.job_id, page=1, per_page=per_page)
        rows: list[dict[str, Any]] = list(payload.get("results") or [])
        total = payload.get("total_results", len(rows))
        page = 1
        while len(rows) < total:
            page += 1
            more = self.client.get_results(self.job_id, page=page, per_page=per_page)
            batch = more.get("results") or []
            if not batch:
                break
            rows.extend(batch)
        return rows

    def scores(
        self,
        *,
        wait: bool = True,
        timeout: float | None = None,
        on_progress: Callable[[JobProgress], None] | None = None,
    ) -> list[Score]:
        """Return one :class:`Score` per submitted item, in submission order.

        With ``wait=True`` (default), blocks until the job finishes. With
        ``wait=False``, scores whatever has been collected so far,
        useful for peeking mid-run.

        Items whose datapoint failed (bad media ref, moderation block)
        come back as ``Score(score=None, error=...)`` rather than raising,
        so one bad row never discards an eval run.

        Raises:
            JobFailedError: The whole job ended ``failed``/``blocked``
                with no results at all.

        """
        if self._scorer is None:
            raise RuntimeError(
                "This EvalJob was attached without a scorer, so results cannot be "
                "mapped to Scores. Reattach with EvalJob.attach(client, job_id, "
                "scorer=<the scorer that created it>), or use .results() for raw rows."
            )
        snapshot = self.wait(timeout=timeout, on_progress=on_progress) if wait else self.progress()
        rows = self.results()
        if snapshot.status in ("failed", "blocked") and not rows:
            raise JobFailedError(self.job_id, snapshot.status, snapshot.errors)
        return self._scorer._scores_from_results(rows, snapshot, self._items, self._n_items)

"""Typed exceptions raised by humanevals.

Every exception raised by this library subclasses :class:`HumanEvalsError`,
so callers can catch that one type to handle any library failure. Errors
returned by the Datapoint API additionally subclass :class:`APIError` and
carry the HTTP status code and the raw ``detail`` payload from the response.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "APIError",
    "AuthenticationError",
    "BudgetExceededError",
    "ContentBlockedError",
    "DispatchFailedError",
    "HumanEvalsError",
    "InsufficientCreditsError",
    "InvalidRequestError",
    "JobFailedError",
    "MediaTooLargeError",
    "NetworkError",
    "NotFoundError",
    "PollTimeoutError",
    "RateLimitError",
    "ServerError",
]


class HumanEvalsError(Exception):
    """Base class for every exception raised by humanevals."""


class APIError(HumanEvalsError):
    """An HTTP error response from the Datapoint API.

    Attributes:
        status_code: HTTP status code of the response.
        detail: The ``detail`` value from the error body. Per the API's
            error envelope this may be a string, an object, or a list,
            so it is kept verbatim; use ``str(exc)`` for a readable form.

    """

    def __init__(self, status_code: int, detail: Any, message: str | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(message or f"API error {status_code}: {_stringify(detail)}")


class NetworkError(HumanEvalsError):
    """The API could not be reached (DNS failure, refused connection, timeout).

    Raised after the client's automatic retries are exhausted. The original
    ``httpx`` exception is available as ``__cause__``.
    """


class AuthenticationError(APIError):
    """401: missing, invalid, or deactivated API key."""


class InsufficientCreditsError(APIError):
    """402: the job would cost more credits than the account has available.

    Nothing was created or reserved. Top up credits, lower
    ``responses_per_item``, or use ``sandbox=True``, then retry.

    Attributes:
        needed_credits: Credits the job would reserve.
        available_credits: Credits currently available on the account.

    """

    def __init__(self, detail: Any) -> None:
        body = detail if isinstance(detail, dict) else {}
        self.needed_credits: int | None = body.get("needed_credits")
        self.available_credits: int | None = body.get("available_credits")
        super().__init__(
            402,
            detail,
            "Insufficient credits: job needs "
            f"{self.needed_credits} but only {self.available_credits} are available.",
        )


class NotFoundError(APIError):
    """404: the job (or other resource) does not exist for this account."""


class InvalidRequestError(APIError):
    """400/422: the request was rejected by validation rules."""


class ContentBlockedError(InvalidRequestError):
    """422 with ``code: content_blocked``, meaning content failed moderation.

    Attributes:
        reason: Human-readable reason, when provided.
        field: The blocked request field (job text moderation), if any.
        filename: The blocked file name (media upload moderation), if any.

    """

    def __init__(self, status_code: int, detail: Any) -> None:
        body = detail if isinstance(detail, dict) else {}
        self.reason: str | None = body.get("reason")
        self.field: str | None = body.get("field")
        self.filename: str | None = body.get("filename")
        super().__init__(status_code, detail)


class MediaTooLargeError(InvalidRequestError):
    """413: an uploaded file exceeds the per-file size limit.

    Attributes:
        max_bytes: The server's per-file limit in bytes.

    """

    def __init__(self, detail: Any) -> None:
        body = detail if isinstance(detail, dict) else {}
        self.max_bytes: int | None = body.get("max_bytes")
        super().__init__(413, detail)


class RateLimitError(APIError):
    """429: rate limit still exceeded after the client's automatic retries."""

    def __init__(self, detail: Any, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(429, detail)


class ServerError(APIError):
    """5xx: the API failed or is temporarily unavailable."""


class DispatchFailedError(ServerError):
    """503 on job creation: the job row was created but task dispatch failed.

    The server renamed the half-created job out of the way, marked it
    ``failed``, and released the credit reservation, so nothing is charged.
    The client does not auto-retry this one case because the failure
    happened after job-row creation and an immediate replay can race the
    server-side cleanup. Simply call ``submit(...)`` again.
    """


class JobFailedError(HumanEvalsError):
    """The job reached a terminal state with no usable results.

    Raised by ``EvalJob.scores()`` when the job ended ``failed`` or
    ``blocked`` and produced no result rows at all. Per-datapoint failures
    within an otherwise successful job do NOT raise; they surface as
    ``Score(score=None, error=...)`` for the affected rows.

    Attributes:
        job_id: The failed job's id.
        status: Terminal status (``"failed"`` or ``"blocked"``).
        errors: Per-datapoint error entries reported by the API, each shaped
            ``{"datapoint_index": int, "error": str}``.

    """

    def __init__(self, job_id: str, status: str, errors: list[dict[str, Any]]) -> None:
        self.job_id = job_id
        self.status = status
        self.errors = errors
        summary = (
            "; ".join(f"[{e.get('datapoint_index')}] {e.get('error')}" for e in errors[:5])
            or "no error details reported"
        )
        if len(errors) > 5:
            summary += f" (+{len(errors) - 5} more)"
        super().__init__(f"Job {job_id} ended '{status}' with no results: {summary}")


class PollTimeoutError(HumanEvalsError):
    """``wait()`` gave up before the job reached a terminal state.

    The job keeps running server-side; no credits are lost. Reattach later
    with ``EvalJob.attach(client, job_id)`` (the job id is included in the
    message and available as ``exc.job_id``).
    """

    def __init__(self, job_id: str, timeout: float) -> None:
        self.job_id = job_id
        self.timeout = timeout
        super().__init__(
            f"Job {job_id} did not finish within {timeout:.0f}s. It is still running; "
            f'reattach with EvalJob.attach(client, "{job_id}") to keep waiting.'
        )


class BudgetExceededError(HumanEvalsError):
    """The pre-submission cost estimate exceeded the caller's ``max_credits``.

    Nothing was submitted. Attributes ``estimated_credits`` and
    ``max_credits`` carry both sides of the comparison.
    """

    def __init__(self, estimated_credits: int, max_credits: int) -> None:
        self.estimated_credits = estimated_credits
        self.max_credits = max_credits
        super().__init__(
            f"Estimated cost ({estimated_credits} credits) exceeds "
            f"max_credits ({max_credits}); nothing was submitted."
        )


def _stringify(detail: Any) -> str:
    """Render a polymorphic error ``detail`` (str | dict | list) as one line."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        msg = detail.get("message") or detail.get("reason") or detail.get("code")
        if isinstance(msg, str):
            return msg
    return repr(detail)

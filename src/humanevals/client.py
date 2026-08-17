"""HTTP client for the Datapoint data-labelling API.

Wraps ``httpx`` with authentication, bounded retries, rate-limit handling,
and mapping of API error responses onto the typed exceptions in
:mod:`humanevals.exceptions`. All humanevals scorers go through a
:class:`Client`; you rarely need to call its endpoint methods directly.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import IO, Any

import httpx

from .exceptions import (
    APIError,
    AuthenticationError,
    ContentBlockedError,
    DispatchFailedError,
    InsufficientCreditsError,
    InvalidRequestError,
    MediaTooLargeError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from .types import Media

__all__ = ["DEFAULT_BASE_URL", "Client"]

DEFAULT_BASE_URL = "https://api.trydatapoint.com/data-labelling/v1"

#: Content types the server expects per extension (it cross-checks these
#: against the extension, which is the source of truth).
_CONTENT_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

#: 503 detail emitted when job creation half-failed server-side. Never
#: auto-retried: the failure happened after job-row creation, and an
#: immediate replay can race the server's cleanup/rename of that row.
_DISPATCH_FAILURE_MARKER = "Failed to queue tasks"


class Client:
    """Authenticated client for the Datapoint API.

    Args:
        api_key: Datapoint API key (``dp_live_...``). Falls back to the
            ``DATAPOINT_API_KEY`` environment variable.
        base_url: API base URL. Falls back to ``DATAPOINT_BASE_URL``, then
            to the production URL.
        timeout: Request timeout in seconds (uploads get 4x for reads).
        max_retries: Automatic retry budget per request for rate limits
            (429), transient server failures (>=500), and network errors.
        transport: Optional ``httpx`` transport, used by the test suite to
            run against canned responses instead of the network.

    Usage is context-manager friendly (``with Client() as client: ...``);
    otherwise call :meth:`close` when done.

    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = (api_key or os.environ.get("DATAPOINT_API_KEY") or "").strip()
        if not key:
            raise AuthenticationError(
                401,
                "No API key.",
                "No API key provided. Pass Client(api_key=...) or set the "
                "DATAPOINT_API_KEY environment variable. Keys are created at "
                "https://trydatapoint.com/?signup=1&from=direct&returnTo=%2Fdashboard "
                "(Dashboard -> API keys).",
            )
        resolved_base = base_url or os.environ.get("DATAPOINT_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved_base.rstrip("/")
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": key, "User-Agent": _user_agent()},
            timeout=httpx.Timeout(timeout, read=timeout * 4),
            transport=transport,
        )
        # Local paths already uploaded this session, keyed by resolved path.
        self._media_refs: dict[str, str] = {}

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- jobs ----------------------------------------------------------------

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /jobs``. Idempotent per ``payload["name"]``.

        Reusing a name returns the existing job (with ``pricing: null``)
        instead of creating a new one; humanevals relies on this to make
        re-running an identical eval free.
        """
        return self._request("POST", "/jobs", json=payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """``GET /jobs/{job_id}``: status and progress counters."""
        return self._request("GET", f"/jobs/{job_id}")

    def get_results(
        self, job_id: str, *, page: int = 1, per_page: int = 100, labels: bool = False
    ) -> dict[str, Any]:
        """``GET /jobs/{job_id}/results``: aggregated per-datapoint results.

        Safe to call mid-run; aggregation is computed on demand. With
        ``labels=False`` (the default) aggregation keys use stable ids.
        ``per_page`` has no server-side upper bound on this endpoint.
        """
        return self._request(
            "GET",
            f"/jobs/{job_id}/results",
            params={"page": page, "per_page": per_page, "labels": labels},
        )

    def get_responses(self, job_id: str, *, page: int = 1, per_page: int = 100) -> dict[str, Any]:
        """``GET /jobs/{job_id}/responses``: raw per-annotator responses.

        Two server behaviors to know: ``per_page`` is capped at 1000 here
        (unlike ``/results``), and pages group whole annotator response
        sets, so a page can contain more than ``per_page`` rows. Iterate
        ``page`` up to the response's ``total_pages``.
        """
        return self._request(
            "GET", f"/jobs/{job_id}/responses", params={"page": page, "per_page": per_page}
        )

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """``POST /jobs/{job_id}/cancel``: irreversible; refunds the unspent reserve."""
        return self._request("POST", f"/jobs/{job_id}/cancel")

    def complete_job(self, job_id: str) -> dict[str, Any]:
        """``POST /jobs/{job_id}/complete``: stop early, keeping collected responses."""
        return self._request("POST", f"/jobs/{job_id}/complete")

    # -- media ---------------------------------------------------------------

    def upload_media(self, paths: list[str | Path]) -> list[dict[str, Any]]:
        """``POST /media``: upload local files, returning their metadata.

        Each entry in the response carries a durable ``media_ref``
        (``dp://...``) usable in job payloads. Files are streamed rather
        than buffered; anything over the server's per-file limit (20 MiB
        by default) raises :class:`MediaTooLargeError`.
        """
        with ExitStack() as stack:
            files: list[tuple[str, tuple[str, IO[bytes], str]]] = []
            for p in paths:
                path = Path(p)
                content_type = _CONTENT_TYPES.get(path.suffix.lower())
                if content_type is None:
                    raise ValueError(
                        f"Unsupported media extension {path.suffix!r} for {path.name!r}. "
                        f"Supported: {', '.join(sorted(_CONTENT_TYPES))}."
                    )
                handle = stack.enter_context(path.open("rb"))
                files.append(("files", (path.name, handle, content_type)))
            body = self._request("POST", "/media", files=files)
        media: list[dict[str, Any]] = body["media"]
        return media

    def resolve_media(self, media: Media) -> dict[str, str]:
        """Turn a :class:`Media` into the ``{"url", "type"}`` item jobs expect.

        Remote sources (``https://`` / ``dp://``) pass through unchanged.
        Local files are uploaded once per client (cached by resolved path)
        and replaced with their ``dp://`` ref.
        """
        media_type = media.resolved_type()
        if media.is_remote:
            return {"url": str(media.source), "type": media_type}
        path = Path(media.source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        cache_key = str(path)
        if cache_key not in self._media_refs:
            uploaded = self.upload_media([path])[0]
            self._media_refs[cache_key] = uploaded["media_ref"]
        return {"url": self._media_refs[cache_key], "type": media_type}

    def media_url(self, relative_url: str) -> str:
        """Absolutize a signed media URL from a results payload.

        The API returns media ``url`` / ``thumbnail_url`` values as
        relative signed paths (``/media/v2/...``) that expire after about
        an hour; re-fetch results to mint fresh ones.
        """
        if relative_url.startswith(("http://", "https://")):
            return relative_url
        return f"{self.base_url}{relative_url}"

    # -- billing -------------------------------------------------------------

    def pricing_quote(self, annotator_filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """``POST /billing/pricing/quote``: free, side-effect-free rate quote.

        Returns the same per-response rate job creation will charge,
        including targeting surcharges. Numeric-range filters (e.g.
        ``median_household_income``) are not accepted by this endpoint and
        are omitted from the quoted filter, so the quote understates the
        rate for jobs that use them.
        """
        quotable = {k: v for k, v in (annotator_filter or {}).items() if not isinstance(v, dict)}
        return self._request(
            "POST",
            "/billing/pricing/quote",
            json={"annotator_filter": quotable or None, "has_screening_steps": False},
        )

    def balance(self) -> dict[str, Any]:
        """``GET /billing/balance``: available/reserved/purchased credits."""
        return self._request("GET", "/billing/balance")

    # -- transport -----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, IO[bytes], str]]] | None = None,
    ) -> dict[str, Any]:
        """Send one API request with bounded retries; return the parsed body.

        Retries up to ``max_retries`` times on 429 (honoring
        ``Retry-After``), on transient >=500 responses, and on network
        errors. Retrying is safe here: GETs are read-only and
        ``POST /jobs`` is idempotent by name. Two exceptions:

        - a 503 job-dispatch failure must not be replayed immediately
          (server-side cleanup race) and raises
          :class:`DispatchFailedError` at once;
        - ``POST /media`` is not idempotent (a replay would store duplicate
          media), so network errors during uploads are not retried.
        """
        last_exc: APIError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.request(method, path, json=json, params=params, files=files)
            except httpx.HTTPError as exc:
                retriable = files is None and attempt < self.max_retries
                if not retriable:
                    raise NetworkError(f"Could not reach the Datapoint API: {exc}") from exc
                time.sleep(float(2**attempt))
                continue
            if response.status_code < 400:
                return response.json()  # type: ignore[no-any-return]

            exc_api = _error_for(response)
            retriable = isinstance(exc_api, RateLimitError) or (
                isinstance(exc_api, ServerError) and not isinstance(exc_api, DispatchFailedError)
            )
            if not retriable or attempt == self.max_retries:
                raise exc_api
            last_exc = exc_api
            time.sleep(_retry_delay(response, attempt))

        raise last_exc if last_exc else AssertionError("unreachable")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Delay before a retry: server's ``Retry-After`` if sane, else backoff."""
    header = response.headers.get("Retry-After")
    if header is not None:
        try:
            value = float(header)
            if math.isfinite(value):
                return min(max(value, 1.0), 120.0)
        except ValueError:
            pass
    return float(2**attempt)


def _error_for(response: httpx.Response) -> APIError:
    """Map an error response onto the matching typed exception.

    The API wraps errors as ``{"detail": <str|dict|list>, "message": ...}``;
    ``detail`` is kept verbatim on the exception since its shape varies.
    Bodies that are not JSON objects at all are kept verbatim too.
    """
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    detail = body.get("detail") if isinstance(body, dict) else body
    status = response.status_code

    if status == 401:
        return AuthenticationError(status, detail)
    if status == 402:
        return InsufficientCreditsError(detail)
    if status == 404:
        return NotFoundError(status, detail)
    if status == 413:
        return MediaTooLargeError(detail)
    if status == 429:
        retry_after: float | None
        try:
            retry_after = float(response.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        return RateLimitError(detail, retry_after)
    if status in (400, 422):
        if isinstance(detail, dict) and detail.get("code") == "content_blocked":
            return ContentBlockedError(status, detail)
        return InvalidRequestError(status, detail)
    if status >= 500:
        if isinstance(detail, str) and detail.startswith(_DISPATCH_FAILURE_MARKER):
            return DispatchFailedError(status, detail)
        return ServerError(status, detail)
    return APIError(status, detail)


def _user_agent() -> str:
    from . import __version__

    return f"humanevals/{__version__} (+https://github.com/impel-intelligence/humanevals)"

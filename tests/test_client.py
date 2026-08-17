"""Client-level behavior: auth, error mapping, retries, media handling."""

from __future__ import annotations

import httpx
import pytest
from conftest import FakeAPI, quote_response

import humanevals as he
from humanevals.client import DEFAULT_BASE_URL


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("DATAPOINT_API_KEY", raising=False)
    with pytest.raises(he.AuthenticationError, match="DATAPOINT_API_KEY"):
        he.Client()


def test_api_key_from_env(monkeypatch, api: FakeAPI):
    monkeypatch.setenv("DATAPOINT_API_KEY", "dp_live_envkey")
    api.add("GET", "/billing/balance", {"available_credits": 10})
    with he.Client(transport=api.transport()) as client:
        client.balance()
    assert api.sent().headers["X-API-Key"] == "dp_live_envkey"


def test_sends_user_agent(client: he.Client, api: FakeAPI):
    api.add("GET", "/billing/balance", {"available_credits": 10})
    client.balance()
    assert api.sent().headers["User-Agent"].startswith("humanevals/")


def test_base_url_default_and_override(api: FakeAPI, monkeypatch):
    monkeypatch.delenv("DATAPOINT_BASE_URL", raising=False)
    assert he.Client(api_key="k", transport=api.transport()).base_url == DEFAULT_BASE_URL
    custom = he.Client(api_key="k", base_url="https://example.test/v1/", transport=api.transport())
    assert custom.base_url == "https://example.test/v1"


# -- error mapping -----------------------------------------------------------


def _client_for(api: FakeAPI, **kwargs) -> he.Client:
    return he.Client(api_key="dp_live_x", transport=api.transport(), **kwargs)


def test_401_maps_to_authentication_error(api: FakeAPI):
    api.add("GET", "/jobs/job_x", {"detail": "Invalid or inactive API key"}, status=401)
    with pytest.raises(he.AuthenticationError, match="Invalid or inactive"):
        _client_for(api).get_job("job_x")


def test_402_maps_to_insufficient_credits(api: FakeAPI):
    detail = {"message": "Insufficient balance", "needed_credits": 500, "available_credits": 120}
    api.add("POST", "/jobs", {"detail": detail, "message": detail}, status=402)
    with pytest.raises(he.InsufficientCreditsError) as info:
        _client_for(api).create_job({"name": "x"})
    assert info.value.needed_credits == 500
    assert info.value.available_credits == 120


def test_404_maps_to_not_found(api: FakeAPI):
    api.add("GET", "/jobs/job_x", {"detail": "Job not found"}, status=404)
    with pytest.raises(he.NotFoundError):
        _client_for(api).get_job("job_x")


def test_422_content_blocked(api: FakeAPI):
    detail = {"code": "content_blocked", "field": "instruction", "reason": "policy"}
    api.add("POST", "/jobs", {"detail": detail}, status=422)
    with pytest.raises(he.ContentBlockedError) as info:
        _client_for(api).create_job({"name": "x"})
    assert info.value.reason == "policy"
    assert info.value.field == "instruction"


def test_422_pydantic_array_detail(api: FakeAPI):
    detail = [{"loc": ["body", "datapoints"], "msg": "field required", "type": "missing"}]
    api.add("POST", "/jobs", {"detail": detail}, status=422)
    with pytest.raises(he.InvalidRequestError) as info:
        _client_for(api).create_job({"name": "x"})
    assert info.value.detail == detail


def test_413_media_too_large(api: FakeAPI, tmp_path):
    detail = {"code": "media_too_large", "max_bytes": 20971520}
    api.add("POST", "/media", {"detail": detail}, status=413)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"x")
    with pytest.raises(he.MediaTooLargeError) as info:
        _client_for(api).upload_media([big])
    assert info.value.max_bytes == 20971520


def test_non_json_error_body(api: FakeAPI):
    api._routes[("GET", "/jobs/job_x")] = [httpx.Response(500, text="gateway exploded")]
    with pytest.raises(he.ServerError):
        _client_for(api, max_retries=0).get_job("job_x")


# -- retries -----------------------------------------------------------------


def test_429_retries_then_succeeds(api: FakeAPI, _no_real_sleep):
    headers = {"Retry-After": "7"}
    api.add("GET", "/jobs/job_x", {"detail": "Rate limit exceeded"}, status=429, headers=headers)
    api.add("GET", "/jobs/job_x", {"job_id": "job_x", "status": "active"})
    assert _client_for(api).get_job("job_x")["status"] == "active"
    assert _no_real_sleep == [7.0]


def test_429_exhausts_retries(api: FakeAPI):
    api.add("GET", "/jobs/job_x", {"detail": "Rate limit exceeded"}, status=429)
    with pytest.raises(he.RateLimitError):
        _client_for(api, max_retries=2).get_job("job_x")
    assert len(api.requests) == 3  # initial + 2 retries


def test_5xx_retries_with_backoff(api: FakeAPI, _no_real_sleep):
    api.add("GET", "/jobs/job_x", {"detail": "Internal Server Error"}, status=500)
    api.add("GET", "/jobs/job_x", {"detail": "Internal Server Error"}, status=500)
    api.add("GET", "/jobs/job_x", {"job_id": "job_x", "status": "active"})
    assert _client_for(api).get_job("job_x")["status"] == "active"
    assert _no_real_sleep == [1.0, 2.0]  # exponential backoff


def test_dispatch_failure_is_not_retried(api: FakeAPI):
    api.add(
        "POST",
        "/jobs",
        {"detail": "Failed to queue tasks. Please retry with a new name."},
        status=503,
    )
    with pytest.raises(he.DispatchFailedError):
        _client_for(api).create_job({"name": "x"})
    assert len(api.requests) == 1  # exactly one attempt


def test_400_is_not_retried(api: FakeAPI):
    api.add("POST", "/jobs", {"detail": "Must provide 'instruction' or 'dimensions'."}, status=400)
    with pytest.raises(he.InvalidRequestError):
        _client_for(api).create_job({"name": "x"})
    assert len(api.requests) == 1


# -- media -------------------------------------------------------------------


def test_upload_media_rejects_unknown_extension(client: he.Client, tmp_path):
    weird = tmp_path / "file.xyz"
    weird.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported media extension"):
        client.upload_media([weird])


def test_resolve_media_uploads_local_file_once(client: he.Client, api: FakeAPI, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video-bytes")
    api.add(
        "POST",
        "/media",
        {
            "media": [
                {
                    "filename": "clip.mp4",
                    "media_id": "b6a1cd3f-1234-4abc-9def-0123456789ab",
                    "media_ref": "dp://b6a1cd3f1234/clip.mp4",
                    "type": "video",
                    "size_bytes": 16,
                    "url": "/media/v2/b6a1cd3f-1234?cid=1&exp=2&sig=3",
                }
            ]
        },
    )
    first = client.resolve_media(he.Media(video))
    second = client.resolve_media(he.Media(video))
    assert first == second == {"url": "dp://b6a1cd3f1234/clip.mp4", "type": "video"}
    assert api.paths().count("POST /media") == 1  # cached after first upload


def test_resolve_media_passes_remote_through(client: he.Client):
    resolved = client.resolve_media(he.Media("https://example.com/a.png"))
    assert resolved == {"url": "https://example.com/a.png", "type": "image"}
    resolved = client.resolve_media(he.Media("dp://abc123def456/a.mp4"))
    assert resolved == {"url": "dp://abc123def456/a.mp4", "type": "video"}


def test_resolve_media_missing_file(client: he.Client):
    with pytest.raises(FileNotFoundError):
        client.resolve_media(he.Media("/nonexistent/file.png"))


def test_media_type_inference_and_override():
    assert he.Media("x.PNG").resolved_type() == "image"
    assert he.Media("x.mov").resolved_type() == "video"
    assert he.Media("https://cdn.example/y.flac?sig=1").resolved_type() == "audio"
    assert he.Media("no-extension", type="image").resolved_type() == "image"
    with pytest.raises(ValueError, match="Cannot infer media type"):
        he.Media("no-extension").resolved_type()


def test_media_url_absolutizes_relative_signed_paths(client: he.Client):
    relative = "/media/v2/abc?cid=1&exp=2&sig=3"
    assert client.media_url(relative) == f"{client.base_url}{relative}"
    absolute = "https://cdn.example/x.png"
    assert client.media_url(absolute) == absolute


# -- billing -----------------------------------------------------------------


def test_pricing_quote_omits_numeric_range_filters(client: he.Client, api: FakeAPI):
    api.add("POST", "/billing/pricing/quote", quote_response(8))
    quote = client.pricing_quote({"country": ["US"], "median_household_income": {"gte": 50000}})
    assert quote["credits_per_response"] == 8
    # The range filter must not reach the quote endpoint (it would 422).
    assert api.body() == {"annotator_filter": {"country": ["US"]}, "has_screening_steps": False}


def test_pricing_quote_with_no_filter(client: he.Client, api: FakeAPI):
    api.add("POST", "/billing/pricing/quote", quote_response())
    client.pricing_quote(None)
    assert api.body() == {"annotator_filter": None, "has_screening_steps": False}

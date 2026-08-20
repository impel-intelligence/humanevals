# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Allow `HumanRating` media items to include one non-selectable reference
  through `RatingItem.reference`.

### Fixed

- Re-upload local media when its contents change at the same path, keeping
  content-derived job names aligned with the media sent to annotators.

## [0.1.1] - 2026-08-17

### Fixed

- The dashboard link in the README and in the missing-API-key error now
  points to the signup URL.

## [0.1.0] - 2026-08-17

### Added

- Initial release: human-backed scorers (`HumanComparison`, `HumanRating`,
  `HumanMultipleChoice`, `HumanRanking`) returning autoevals-compatible
  `Score` objects.
- Batch-first evaluation: one Datapoint job per eval run, with polling,
  reattachment by job id, and per-row score hydration.
- `EvalJob.cancel()` and `EvalJob.complete()` lifecycle controls.
- Transparent media upload for local files (streamed, cached per client).
- Content-derived idempotent job names, so re-running an identical eval is
  never charged twice; the canonicalization is pinned by a golden test.
- Cost controls: `estimate_credits()`, `max_credits` budget caps, and a
  hard error when a budget cap cannot be priced (numeric-range filters).
- Typed exceptions, including credit-aware `InsufficientCreditsError` and
  `NetworkError` for connectivity failures (retried with backoff first).
- Guards against silent misuse: bare-string item sequences are rejected,
  degenerate rating scales are rejected at construction, and an explicit
  job name that collides with a different existing job raises.

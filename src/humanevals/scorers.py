"""Human-backed scorers: the public heart of humanevals.

Each scorer poses one kind of question to real human annotators via the
Datapoint API and turns their aggregated answers into autoevals-compatible
:class:`~humanevals.Score` objects:

- :class:`HumanComparison`: "which of these two is better?"
- :class:`HumanRating`: "rate this on a scale"
- :class:`HumanMultipleChoice`: "pick the right answer"
- :class:`HumanRanking`: "order these from best to worst"

Design notes that matter to callers:

- **Batch-first.** One ``eval_batch()``/``submit()`` call creates exactly one
  Datapoint job for all items; humans answer in parallel and each item gets
  ``responses_per_item`` independent judgments.
- **Idempotent by content.** Job names are derived from a hash of the full
  request (instruction, config, and item contents). Re-submitting an
  identical eval replays the existing job server-side, so you are never
  charged twice for the same question. Pass ``fresh=True`` to force a new
  collection round.
- **Text pairwise comparisons ride on the ranking task.** The API's
  ``comparison`` task is media-only, so text pairs are submitted as a
  2-candidate ranking. With two items this is mathematically equivalent:
  P(humans prefer A) = 2 - mean_rank(A).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
import warnings
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from .client import Client
from .exceptions import BudgetExceededError
from .job import EvalJob
from .types import ChoiceItem, JobProgress, Media, Pair, RankingItem, RatingItem, Score

__all__ = [
    "HumanComparison",
    "HumanMultipleChoice",
    "HumanRanking",
    "HumanRating",
    "HumanScorer",
]

# Shared lazily-created client for scorers constructed without one.
_default_client: Client | None = None
_MediaHashes = dict[tuple[str, str], str]


def _get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


class HumanScorer:
    """Common machinery for all human scorers. Use a concrete subclass.

    Args:
        instruction: The question shown to annotators. May contain a
            ``{context}`` placeholder, substituted per item.
        responses_per_item: Independent human judgments collected per item.
            Defaults to 5 (the API's own default is 10): an odd count
            avoids ties in pairwise tasks while keeping cost moderate;
            raise it for tighter confidence.
        sandbox: When ``True``, the job runs on Datapoint's free test pool:
            zero credits, but responses come from test annotators. Use it
            for wiring and demos, never for real measurements.
        annotator_filter: Optional audience targeting, e.g.
            ``{"country": ["US", "CA"]}``. Targeting can add per-response
            surcharges; ``estimate_credits()`` accounts for them.
        client: A :class:`~humanevals.Client`. Defaults to a shared client
            configured from the ``DATAPOINT_API_KEY`` environment variable.

    """

    #: Human-readable scorer name used on emitted Scores; set per subclass.
    name: str = "HumanScorer"

    def __init__(
        self,
        instruction: str,
        *,
        responses_per_item: int = 5,
        sandbox: bool = False,
        annotator_filter: dict[str, Any] | None = None,
        client: Client | None = None,
    ) -> None:
        if not instruction or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")
        self.instruction = instruction
        self.responses_per_item = responses_per_item
        self.sandbox = sandbox
        self.annotator_filter = annotator_filter
        self._client = client

    @property
    def client(self) -> Client:
        """The client in use (created lazily from the environment if needed)."""
        if self._client is None:
            self._client = _get_default_client()
        return self._client

    # -- public API ----------------------------------------------------------

    def submit(
        self,
        items: Sequence[Any],
        *,
        name: str | None = None,
        fresh: bool = False,
        max_credits: int | None = None,
    ) -> EvalJob:
        """Submit all items as one job and return immediately with an :class:`EvalJob`.

        Args:
            items: The items to evaluate (type depends on the scorer).
            name: Explicit job name. Defaults to a content hash of the full
                request, which makes identical submissions idempotent:
                the API returns the existing job instead of charging again.
            fresh: Force a brand-new job (fresh human responses) even if an
                identical eval was already run, by uniquifying the name.
            max_credits: Refuse to submit (raising
                :class:`~humanevals.exceptions.BudgetExceededError`) if the
                pre-flight cost estimate exceeds this many credits. Not
                compatible with numeric-range annotator filters, whose
                surcharge the quote endpoint cannot price (a ValueError
                explains this rather than enforcing an underestimate).

        Returns:
            An :class:`EvalJob`; call ``.scores()`` to wait and collect.

        """
        if isinstance(items, (str, bytes, Media)):
            raise TypeError(
                f"items must be a sequence of items, not a single {type(items).__name__}. "
                "Wrap it in a list, or use eval() / scorer(...) for one item."
            )
        prepared = [self._coerce_item(i) for i in items]
        if not prepared:
            raise ValueError("items must be non-empty.")
        self._validate_batch(prepared)
        self._check_context_placeholder(prepared)

        if max_credits is not None and not self.sandbox:
            has_range_filter = any(
                isinstance(v, dict) for v in (self.annotator_filter or {}).values()
            )
            if has_range_filter:
                raise ValueError(
                    "max_credits cannot be enforced together with a numeric-range "
                    "annotator filter (e.g. median_household_income): the quote "
                    "endpoint cannot price its surcharge, so the estimate would "
                    "silently understate the real cost. Drop max_credits or the "
                    "range filter."
                )
            estimate = self.estimate_credits(prepared)
            if estimate > max_credits:
                raise BudgetExceededError(estimate, max_credits)

        media_hashes: _MediaHashes = {}
        job_name = name or self._content_name(prepared, media_hashes=media_hashes)
        if fresh:
            job_name = f"{job_name}-{uuid.uuid4().hex[:8]}"

        body = self._job_body(prepared, media_hashes=media_hashes)
        body["name"] = job_name
        created = self.client.create_job(body)
        # pricing is null exactly when the API replayed an existing job for
        # this name; catch replays that clearly aren't this batch.
        is_replay = created.get("pricing") is None
        existing = created.get("total_datapoints")
        if is_replay and existing is not None and existing != len(prepared):
            raise ValueError(
                f"Job name {job_name!r} already belongs to a different job "
                f"({created.get('job_id')}: {existing} datapoints, this batch has "
                f"{len(prepared)}). Choose another name= or pass fresh=True."
            )
        return EvalJob(
            self.client,
            created["job_id"],
            name=job_name,
            scorer=self,
            items=prepared,
        )

    def eval_batch(
        self,
        items: Sequence[Any],
        *,
        name: str | None = None,
        fresh: bool = False,
        max_credits: int | None = None,
        timeout: float | None = None,
    ) -> list[Score]:
        """Submit all items as one job, wait for humans, and return the Scores.

        Equivalent to ``submit(...).scores(timeout=...)``. This blocks for
        as long as human annotation takes (minutes to hours); use
        ``submit()`` if you'd rather come back later.
        """
        return self.submit(items, name=name, fresh=fresh, max_credits=max_credits).scores(
            timeout=timeout
        )

    def eval(self, item: Any, **submit_kwargs: Any) -> Score:
        """Evaluate a single item, blocking until humans respond.

        Convenience for notebooks and spot checks. For datasets always
        prefer ``eval_batch()``, which creates one job instead of many.
        """
        return self.eval_batch([item], **submit_kwargs)[0]

    def estimate_credits(self, items: Sequence[Any]) -> int:
        """Estimate the credit cost of evaluating ``items`` (0 for sandbox).

        Uses the API's free quote endpoint, so the per-response rate
        matches what submission would actually charge, including audience
        surcharges. Exception: numeric-range filters cannot be quoted and
        are excluded, so the estimate understates cost for jobs using them
        (see :meth:`Client.pricing_quote`).
        """
        if self.sandbox:
            return 0
        quote = self.client.pricing_quote(self.annotator_filter)
        rate = int(quote["credits_per_response"])
        return len(items) * self.responses_per_item * rate

    # -- subclass hooks ------------------------------------------------------

    def _coerce_item(self, item: Any) -> Any:
        """Convert a loosely-typed input into this scorer's item type."""
        raise NotImplementedError

    def _validate_batch(self, items: list[Any]) -> None:
        """Reject invalid or mixed-mode batches before any network call."""

    def _datapoint(
        self, item: Any, *, for_naming: bool, media_hashes: _MediaHashes
    ) -> dict[str, Any]:
        """Build one API datapoint.

        With ``for_naming=True``, local files are represented by a stable
        content hash instead of being uploaded, so job names (and thus
        idempotency) survive re-runs even though re-uploads mint new
        ``dp://`` refs.
        """
        raise NotImplementedError

    def _task_type(self, items: list[Any]) -> str:
        """Return the Datapoint task type used to transport this batch."""
        raise NotImplementedError

    def _response_options(self, items: list[Any]) -> dict[str, Any] | None:
        """Task-type-specific ``response_options``, or ``None``."""
        return None

    def _score_row(self, row: dict[str, Any], item: Any) -> Score:
        """Convert one aggregated result row into a :class:`Score`."""
        raise NotImplementedError

    # -- shared internals ----------------------------------------------------

    def _job_body(
        self, items: list[Any], *, media_hashes: _MediaHashes | None = None
    ) -> dict[str, Any]:
        media_hashes = media_hashes if media_hashes is not None else {}
        body: dict[str, Any] = {
            "instruction": self.instruction,
            "task_type": self._task_type(items),
            "max_responses_per_datapoint": self.responses_per_item,
            "datapoints": [
                self._datapoint(i, for_naming=False, media_hashes=media_hashes) for i in items
            ],
        }
        options = self._response_options(items)
        if options is not None:
            body["response_options"] = options
        if self.annotator_filter is not None:
            body["annotator_filter"] = self.annotator_filter
        if self.sandbox:
            body["serving_environment"] = "sandbox"
        return body

    def _content_name(self, items: list[Any], *, media_hashes: _MediaHashes | None = None) -> str:
        """Deterministic job name from the full request content."""
        media_hashes = media_hashes if media_hashes is not None else {}
        naming_body: dict[str, Any] = {
            "instruction": self.instruction,
            "task_type": self._task_type(items),
            "max_responses_per_datapoint": self.responses_per_item,
            "response_options": self._response_options(items),
            "annotator_filter": self.annotator_filter,
            "sandbox": self.sandbox,
            "datapoints": [
                self._datapoint(i, for_naming=True, media_hashes=media_hashes) for i in items
            ],
        }
        canonical = json.dumps(naming_body, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"humanevals:1:{canonical}".encode()).hexdigest()
        return f"he-{digest[:20]}"

    def _check_context_placeholder(self, items: list[Any]) -> None:
        """Warn when per-item context exists but the instruction never shows it."""
        has_context = any(getattr(i, "context", None) for i in items)
        if has_context and "{context}" not in self.instruction:
            warnings.warn(
                "Some items carry `context`, but the instruction has no {context} "
                "placeholder, so annotators will not see it. Add '{context}' to "
                "the instruction if the context should be shown.",
                UserWarning,
                stacklevel=_user_stacklevel(),
            )

    def _media_item(
        self, media: Media, *, for_naming: bool, media_hashes: _MediaHashes
    ) -> dict[str, str]:
        """Resolve a :class:`Media` into an API media item (or naming token)."""
        media_type = media.resolved_type()
        if media.is_remote:
            return {"url": str(media.source), "type": media_type}
        path = Path(media.source).expanduser().resolve()
        cache_key = (str(path), media_type)
        digest = media_hashes.get(cache_key)
        if digest is None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            media_hashes[cache_key] = digest
        if for_naming:
            return {"content_sha256": digest, "type": media.resolved_type()}
        return self.client.resolve_media(media, content_sha256=digest)

    def _scores_from_results(
        self,
        rows: list[dict[str, Any]],
        snapshot: JobProgress,
        items: list[Any] | None,
        n_items: int | None,
    ) -> list[Score]:
        """Map API result rows back onto submitted items, in order."""
        by_index: dict[int, dict[str, Any]] = {
            row["datapoint_index"]: row
            for row in rows
            if isinstance(row.get("datapoint_index"), int)
        }
        failures = {
            e.get("datapoint_index"): str(e.get("error", "datapoint failed"))
            for e in snapshot.errors
        }
        # Failed/blocked datapoints are excluded from /results, so the row
        # set alone undercounts; prefer the caller's count, then the job's.
        if n_items is not None:
            count = n_items
        elif snapshot.total_datapoints:
            count = snapshot.total_datapoints
        else:
            count = max(by_index, default=-1) + 1
        scores: list[Score] = []
        for i in range(count):
            row = by_index.get(i)
            if row is None:
                scores.append(
                    Score(
                        name=self.name,
                        score=None,
                        metadata={"job_id": snapshot.job_id, "datapoint_index": i},
                        error=failures.get(i, "no result returned for this item"),
                    )
                )
                continue
            item = items[i] if items is not None and i < len(items) else None
            score = self._score_row(row, item)
            score.metadata.setdefault("job_id", snapshot.job_id)
            score.metadata.setdefault("datapoint_index", i)
            scores.append(score)
        return scores


def _user_stacklevel() -> int:
    """Return the warnings stacklevel of the first frame outside this package.

    Makes warnings point at the caller's code regardless of how many
    internal frames (eval_batch -> submit -> check) sit in between.
    """
    package_dir = os.path.dirname(__file__)
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(package_dir):
        frame = frame.f_back
        level += 1
    return level


def _common_metadata(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Copy the given aggregation keys (and their weighted twins) if present."""
    out: dict[str, Any] = {}
    for key in (*keys, *[f"weighted_{k}" for k in keys], "total_responses"):
        if key in row:
            out[key] = row[key]
    return out


class HumanComparison(HumanScorer):
    """Pairwise preference: humans pick the better of two candidates.

    The score is **P(humans prefer** ``a`` **)**, the fraction of judgments
    favoring the first candidate, so ``score > 0.5`` means ``a`` wins.
    Used autoevals-style (``scorer(output=..., expected=...)``), ``a`` is
    ``output``, making the score directly comparable to autoevals'
    ``Battle``.

    Items are :class:`~humanevals.Pair` objects (or ``(a, b)`` tuples).
    Both sides must be text, or both :class:`~humanevals.Media` of the same
    type. A video pair may add an image ``reference``, which becomes an
    image-to-video comparison automatically.

    Example::

        scorer = HumanComparison(
            "Which video looks more realistic?", responses_per_item=9
        )
        scores = scorer.eval_batch(
            [Pair(Media("runs/a/0.mp4"), Media("runs/b/0.mp4"))]
        )
    """

    name = "HumanComparison"

    def _coerce_item(self, item: Any) -> Pair:
        if isinstance(item, Pair):
            return item
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return Pair(a=item[0], b=item[1])
        raise TypeError(f"HumanComparison items must be Pair or (a, b); got {type(item).__name__}")

    @staticmethod
    def _mode(pair: Pair) -> str:
        """Classify one pair: 'text', 'comparison' (media), or 'i2v'."""
        a_text, b_text = isinstance(pair.a, str), isinstance(pair.b, str)
        if a_text and b_text:
            if pair.reference is not None:
                raise ValueError("Text pairs cannot carry a media reference.")
            return "text"
        if a_text or b_text:
            raise ValueError("A Pair must be two texts or two Media, not one of each.")
        a, b = pair.a, pair.b
        assert isinstance(a, Media) and isinstance(b, Media)
        if a.resolved_type() != b.resolved_type():
            raise ValueError(
                f"Both candidates must share a media type; got "
                f"{a.resolved_type()!r} vs {b.resolved_type()!r}."
            )
        if pair.reference is not None:
            ref_type = pair.reference.resolved_type()
            if a.resolved_type() == "video" and ref_type == "image":
                return "i2v"
            if ref_type != a.resolved_type():
                raise ValueError(
                    "A reference must be an image (for video pairs) or match the "
                    f"candidates' type; got reference {ref_type!r} with "
                    f"{a.resolved_type()!r} candidates."
                )
        return "comparison"

    def _validate_batch(self, items: list[Pair]) -> None:
        modes = {self._mode(p) for p in items}
        if len(modes) > 1:
            raise ValueError(
                f"All pairs in a batch must be the same kind; got {sorted(modes)}. "
                "Split text and media comparisons into separate eval_batch() calls."
            )

    def _task_type(self, items: list[Pair]) -> str:
        mode = self._mode(items[0])
        # Text pairs ride on the ranking task (the comparison task is
        # media-only); with 2 candidates the math maps back exactly.
        return {"text": "ranking", "comparison": "comparison", "i2v": "i2v_comparison"}[mode]

    def _datapoint(
        self, item: Pair, *, for_naming: bool, media_hashes: _MediaHashes
    ) -> dict[str, Any]:
        mode = self._mode(item)
        if mode == "text":
            media: dict[str, Any] = {
                "candidates": [
                    {"id": "a", "text": item.a},
                    {"id": "b", "text": item.b},
                ]
            }
        else:
            assert isinstance(item.a, Media) and isinstance(item.b, Media)
            media = {
                "candidates": [
                    self._media_item(item.a, for_naming=for_naming, media_hashes=media_hashes),
                    self._media_item(item.b, for_naming=for_naming, media_hashes=media_hashes),
                ]
            }
            if item.reference is not None:
                media["reference"] = [
                    self._media_item(
                        item.reference, for_naming=for_naming, media_hashes=media_hashes
                    )
                ]
        datapoint: dict[str, Any] = {"media": media}
        if item.context is not None:
            datapoint["context"] = item.context
        return datapoint

    def _score_row(self, row: dict[str, Any], item: Pair | None) -> Score:
        total = row.get("total_responses") or 0
        if "votes" in row:  # media path: native comparison aggregation
            metadata = _common_metadata(row, "votes", "consensus", "confidence", "agreement_rate")
            if total == 0:
                return Score(name=self.name, score=None, metadata=metadata)
            votes = row.get("votes") or {}
            score = float(votes.get("A", 0)) / float(total)
            return Score(name=self.name, score=score, metadata=metadata)

        # Text path: 2-candidate ranking. mean_rank(a) = 2 - P(prefer a).
        metadata = _common_metadata(row, "average_ranks", "ranking_order")
        average_ranks = row.get("average_ranks") or {}
        if total == 0 or "a" not in average_ranks:
            return Score(name=self.name, score=None, metadata=metadata)
        preference_a = min(max(2.0 - float(average_ranks["a"]), 0.0), 1.0)
        preference_b = 1.0 - preference_a
        if preference_a > preference_b:
            consensus = "a"
        elif preference_b > preference_a:
            consensus = "b"
        else:
            consensus = "tie"
        metadata["consensus"] = consensus
        return Score(name=self.name, score=preference_a, metadata=metadata)

    def __call__(
        self, output: str | Media, expected: str | Media, input: str | None = None
    ) -> Score:
        """Autoevals-style call: score = P(humans prefer ``output`` over ``expected``).

        Creates a one-item job and blocks until humans respond. Fine for
        spot checks; use ``eval_batch()`` for datasets.
        """
        return self.eval(Pair(a=output, b=expected, context=input))


class HumanRating(HumanScorer):
    r"""Absolute quality: humans rate each subject on a fixed scale.

    The score is the mean rating normalized to ``[0, 1]`` over the scale's
    range (a mean of 4.2 on a 1-5 scale scores 0.8). Raw mean, median, and
    the full distribution are in metadata.

    Args:
        instruction: Question shown with each subject. For *text* subjects
            it must contain ``{context}``, which is replaced by the text,
            e.g. ``"How helpful is this response?\\n\\n{context}"``.
        scale: Either a ``(low, high)`` *tuple* for an inclusive integer
            range, or an explicit list of numeric values. Note the
            difference: ``scale=(1, 5)`` is the five options 1..5, while
            ``scale=[1, 5]`` is a two-option scale. Defaults to ``(1, 5)``.
        labels: Optional anchor labels, e.g. ``{1: "Poor", 5: "Excellent"}``.

    Items are :class:`~humanevals.RatingItem` objects, or bare ``str`` /
    :class:`~humanevals.Media` subjects.

    """

    name = "HumanRating"

    def __init__(
        self,
        instruction: str,
        *,
        scale: tuple[int, int] | Sequence[float] = (1, 5),
        labels: dict[int | float | str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(instruction, **kwargs)
        if isinstance(scale, tuple) and len(scale) == 2:
            low, high = scale
            if not (isinstance(low, int) and isinstance(high, int)):
                raise ValueError(
                    "A (low, high) scale tuple must use integers; pass an explicit "
                    "list of values for non-integer scales."
                )
            if high <= low:
                raise ValueError(f"A (low, high) scale tuple needs high > low; got {scale!r}.")
            self.scale: list[float] = [float(v) for v in range(low, high + 1)]
        else:
            self.scale = [float(v) for v in scale]
        if len(self.scale) < 2:
            raise ValueError("scale needs at least 2 values.")
        if len(set(self.scale)) != len(self.scale):
            raise ValueError(f"scale values must be unique; got {list(scale)!r}.")
        self.labels = {str(_int_if_whole(k)): v for k, v in (labels or {}).items()}

    def _coerce_item(self, item: Any) -> RatingItem:
        if isinstance(item, RatingItem):
            return item
        if isinstance(item, (str, Media)):
            return RatingItem(subject=item)
        raise TypeError(
            f"HumanRating items must be RatingItem, str, or Media; got {type(item).__name__}"
        )

    def _validate_batch(self, items: list[RatingItem]) -> None:
        for item in items:
            if isinstance(item.subject, str):
                if "{context}" not in self.instruction:
                    raise ValueError(
                        "Text subjects are shown through the instruction's {context} "
                        "placeholder, but the instruction does not contain one. "
                        'Example: HumanRating("Rate this response:\\n\\n{context}")'
                    )
                if item.context is not None:
                    raise ValueError(
                        "A text subject already occupies the context slot; "
                        "RatingItem(subject=<text>, context=...) is ambiguous."
                    )

    def _check_context_placeholder(self, items: list[RatingItem]) -> None:
        # Text subjects are delivered via {context}; _validate_batch already
        # hard-errors when the placeholder is missing, so only warn about
        # unshown *extra* context on media subjects.
        has_extra = any(item.context for item in items if isinstance(item.subject, Media))
        if has_extra and "{context}" not in self.instruction:
            super()._check_context_placeholder(items)

    def _task_type(self, items: list[RatingItem]) -> str:
        return "rating"

    def _response_options(self, items: list[RatingItem]) -> dict[str, Any]:
        options: dict[str, Any] = {"scale": [_int_if_whole(v) for v in self.scale]}
        if self.labels:
            options["labels"] = self.labels
        return options

    def _datapoint(
        self, item: RatingItem, *, for_naming: bool, media_hashes: _MediaHashes
    ) -> dict[str, Any]:
        if isinstance(item.subject, str):
            # Text-only rating: the API requires an explicit empty media dict.
            return {"media": {}, "context": item.subject}
        datapoint: dict[str, Any] = {
            "media": {
                "subject": [
                    self._media_item(item.subject, for_naming=for_naming, media_hashes=media_hashes)
                ]
            }
        }
        if item.context is not None:
            datapoint["context"] = item.context
        return datapoint

    def _score_row(self, row: dict[str, Any], item: RatingItem | None) -> Score:
        metadata = _common_metadata(row, "mean", "median", "distribution")
        metadata["scale"] = [_int_if_whole(v) for v in self.scale]
        mean = row.get("mean")
        if mean is None:
            return Score(name=self.name, score=None, metadata=metadata)
        low, high = min(self.scale), max(self.scale)
        normalized = (float(mean) - low) / (high - low)
        return Score(name=self.name, score=min(max(normalized, 0.0), 1.0), metadata=metadata)

    def __call__(self, output: str | Media, input: str | None = None) -> Score:
        """Autoevals-style call: rate ``output``, returning the normalized mean.

        ``input`` is extra context and works only with :class:`Media`
        outputs. A text output already fills the API's single per-item
        context slot, so framing for text ratings belongs in the
        instruction instead.
        """
        if input is not None and isinstance(output, str):
            raise ValueError(
                "input= cannot be combined with a text output: the text being "
                "rated uses the single per-item context slot. Put shared framing "
                "in the instruction, e.g. "
                "HumanRating('Given the question, rate this answer: {context}')."
            )
        return self.eval(RatingItem(subject=output, context=input))


class HumanMultipleChoice(HumanScorer):
    """Classification: humans answer a multiple-choice question per item.

    With ``expected`` set on an item, the score is the fraction of
    annotators who chose that option: human accuracy against your label.
    Without it, the score is ``None`` and the vote distribution (a human
    labeling of your data) is in metadata.

    Args:
        instruction: Guidance shown alongside every question, e.g.
            "Answer based only on the screenshot." (each item's own
            ``question`` is what annotators answer).
        shuffle: Randomize option display order per annotator (default
            ``True``; recommended, it cancels position bias).

    Items are :class:`~humanevals.ChoiceItem` objects. Option ids are
    assigned positionally (``option_1``...); metadata reports votes both by
    id and by option text.

    """

    name = "HumanMultipleChoice"

    def __init__(self, instruction: str, *, shuffle: bool = True, **kwargs: Any) -> None:
        super().__init__(instruction, **kwargs)
        self.shuffle = shuffle

    def _coerce_item(self, item: Any) -> ChoiceItem:
        if isinstance(item, ChoiceItem):
            return item
        raise TypeError(f"HumanMultipleChoice items must be ChoiceItem; got {type(item).__name__}")

    def _validate_batch(self, items: list[ChoiceItem]) -> None:
        for item in items:
            if not item.question or not item.question.strip():
                raise ValueError("ChoiceItem.question must be non-empty.")
            if len(item.options) < 2:
                raise ValueError("ChoiceItem needs at least 2 options.")
            if len(set(item.options)) != len(item.options):
                raise ValueError(f"Duplicate options in {list(item.options)!r}.")
            if item.expected is not None and item.expected not in item.options:
                raise ValueError(
                    f"expected {item.expected!r} is not among the options {list(item.options)!r}."
                )
            if item.subject is not None and str(item.subject.source).startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    "A choice subject must be a local file or dp:// ref; the API "
                    "does not accept plain https URLs here. Upload it first or "
                    "pass a local path."
                )

    def _task_type(self, items: list[ChoiceItem]) -> str:
        return "multiple_choice"

    def _response_options(self, items: list[ChoiceItem]) -> dict[str, Any]:
        return {"mode": "single", "shuffle": self.shuffle}

    @staticmethod
    def _option_id(index: int) -> str:
        return f"option_{index + 1}"

    def _datapoint(
        self, item: ChoiceItem, *, for_naming: bool, media_hashes: _MediaHashes
    ) -> dict[str, Any]:
        media: dict[str, Any] = {
            "options": [
                {"id": self._option_id(i), "text": text} for i, text in enumerate(item.options)
            ]
        }
        if item.subject is not None:
            media["subject"] = [
                self._media_item(item.subject, for_naming=for_naming, media_hashes=media_hashes)
            ]
        return {"media": media, "context": item.question}

    def _score_row(self, row: dict[str, Any], item: ChoiceItem | None) -> Score:
        metadata = _common_metadata(row, "votes", "consensus", "confidence")
        total = row.get("total_responses") or 0
        votes: dict[str, int] = row.get("votes") or {}
        if item is not None:
            id_to_text = {self._option_id(i): t for i, t in enumerate(item.options)}
            metadata["votes_by_text"] = {id_to_text.get(k, k): v for k, v in votes.items()}
            consensus = row.get("consensus")
            metadata["consensus_text"] = (
                id_to_text.get(consensus, consensus) if isinstance(consensus, str) else consensus
            )
        if total == 0:
            return Score(name=self.name, score=None, metadata=metadata)
        if item is None or item.expected is None:
            return Score(
                name=self.name,
                score=None,
                metadata=metadata,
                error=None if item is not None else "reattached without items",
            )
        expected_id = self._option_id(list(item.options).index(item.expected))
        return Score(
            name=self.name,
            score=float(votes.get(expected_id, 0)) / float(total),
            metadata=metadata,
        )


class HumanRanking(HumanScorer):
    """Ordering: humans rank several candidates from best to worst.

    With ``expected_order`` set on an item, the score is rank agreement:
    Kendall's tau between the human consensus order and yours, scaled to
    ``[0, 1]`` (1.0 = identical order, 0.5 = uncorrelated, 0.0 = exactly
    reversed). Without it, the score is ``None`` and the consensus order is
    in metadata.

    Items are :class:`~humanevals.RankingItem` objects; candidates must be
    uniformly text or uniformly same-type :class:`~humanevals.Media`.
    """

    name = "HumanRanking"

    def _coerce_item(self, item: Any) -> RankingItem:
        if isinstance(item, RankingItem):
            return item
        if isinstance(item, (tuple, list)):
            return RankingItem(candidates=list(item))
        raise TypeError(
            f"HumanRanking items must be RankingItem or a candidate list; got {type(item).__name__}"
        )

    def _validate_batch(self, items: list[RankingItem]) -> None:
        for item in items:
            candidates = list(item.candidates)
            if len(candidates) < 2:
                raise ValueError("RankingItem needs at least 2 candidates.")
            text_flags = {isinstance(c, str) for c in candidates}
            if len(text_flags) > 1:
                raise ValueError("Candidates must be all text or all Media, not mixed.")
            if not text_flags.pop():  # all Media
                types = {c.resolved_type() for c in candidates if isinstance(c, Media)}
                if len(types) > 1:
                    raise ValueError(f"Media candidates must share one type; got {sorted(types)}.")
            if item.expected_order is not None and sorted(item.expected_order) != list(
                range(len(candidates))
            ):
                raise ValueError(
                    "expected_order must be a permutation of candidate indices "
                    f"0..{len(candidates) - 1}; got {list(item.expected_order)!r}."
                )

    def _task_type(self, items: list[RankingItem]) -> str:
        return "ranking"

    def _candidate_ids(self, item: RankingItem) -> list[str]:
        """Return result ids by position: ours for text, server-minted for media."""
        candidates = list(item.candidates)
        if isinstance(candidates[0], str):
            return [f"item_{i + 1}" for i in range(len(candidates))]
        media_type = candidates[0].resolved_type() if isinstance(candidates[0], Media) else ""
        return [f"{media_type}_{i + 1}" for i in range(len(candidates))]

    def _datapoint(
        self, item: RankingItem, *, for_naming: bool, media_hashes: _MediaHashes
    ) -> dict[str, Any]:
        candidates = list(item.candidates)
        ids = self._candidate_ids(item)
        payload: list[dict[str, str]]
        if isinstance(candidates[0], str):
            payload = [{"id": ids[i], "text": str(c)} for i, c in enumerate(candidates)]
        else:
            payload = [
                self._media_item(c, for_naming=for_naming, media_hashes=media_hashes)
                for c in candidates
                if isinstance(c, Media)
            ]
        datapoint: dict[str, Any] = {"media": {"candidates": payload}}
        if item.context is not None:
            datapoint["context"] = item.context
        return datapoint

    def _score_row(self, row: dict[str, Any], item: RankingItem | None) -> Score:
        metadata = _common_metadata(row, "average_ranks", "ranking_order")
        total = row.get("total_responses") or 0
        order: list[str] = row.get("ranking_order") or []
        if item is not None and order:
            ids = self._candidate_ids(item)
            if isinstance(next(iter(item.candidates)), str):
                id_to_text = dict(zip(ids, [str(c) for c in item.candidates], strict=True))
                metadata["ranking_order_texts"] = [id_to_text.get(i, i) for i in order]
        if total == 0 or not order:
            return Score(name=self.name, score=None, metadata=metadata)
        if item is None or item.expected_order is None:
            return Score(
                name=self.name,
                score=None,
                metadata=metadata,
                error=None if item is not None else "reattached without items",
            )
        ids = self._candidate_ids(item)
        expected_ids = [ids[i] for i in item.expected_order]
        tau = _kendall_tau(expected_ids, order)
        metadata["kendall_tau"] = tau
        return Score(name=self.name, score=(tau + 1.0) / 2.0, metadata=metadata)


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall's tau between two orderings of the same ids (1 = identical)."""
    common = [x for x in order_a if x in set(order_b)]
    n = len(common)
    if n < 2:
        return 0.0
    position_b = {x: i for i, x in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if position_b[common[i]] < position_b[common[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def _int_if_whole(value: Any) -> Any:
    """Render 4.0 as 4 so scales/labels look natural in payloads."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

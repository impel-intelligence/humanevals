"""Public data types: the autoevals-compatible ``Score``, media references, and eval items.

The central type is :class:`Score`, which mirrors the shape used by
`autoevals <https://github.com/braintrustdata/autoevals>`_: ``name``, a
``score`` in ``[0, 1]`` (or ``None`` when not computable), free-form
``metadata``, and an optional ``error`` string. Humanevals scores drop
into any pipeline that already consumes autoevals scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ChoiceItem",
    "JobProgress",
    "Media",
    "Pair",
    "RankingItem",
    "RatingItem",
    "Score",
]

#: Media types accepted by the API, keyed by lowercase file extension.
#: Mirrors the server's extension table; extension is the source of truth.
EXTENSION_TYPES: dict[str, str] = {
    ".mp4": "video",
    ".webm": "video",
    ".mov": "video",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".avif": "image",
    ".gif": "image",
    ".svg": "image",
    ".heic": "image",
    ".heif": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".m4a": "audio",
    ".flac": "audio",
}


@dataclass
class Score:
    """Result of one human evaluation of one item.

    Compatible with the autoevals ``Score`` shape.

    Attributes:
        name: Name of the scorer that produced this score.
        score: Normalized value in ``[0, 1]``, or ``None`` when no score
            could be computed (zero responses, failed datapoint, or a
            scorer configured without an ``expected`` target). When
            ``None``, ``metadata`` and ``error`` explain why.
        metadata: Scorer-specific detail: raw vote counts, consensus,
            agreement, the Datapoint ``job_id`` / ``datapoint_index`` for
            traceability, and any trust-weighted aggregates the API returned.
        error: Human-readable error for rows that could not be scored.

    """

    name: str
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict form, convenient for JSON serialization."""
        return {
            "name": self.name,
            "score": self.score,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass(frozen=True)
class Media:
    """A media input: a local file path, a public ``https://`` URL, or a ``dp://`` ref.

    Local paths are uploaded automatically at submission time and replaced
    with durable ``dp://`` refs. Plain strings passed to scorers are always
    treated as *text*, never as file paths; wrap anything that should be
    fetched or uploaded in ``Media(...)``.

    Attributes:
        source: File path, ``https://`` URL, or ``dp://`` ref.
        type: ``"image"``, ``"audio"``, or ``"video"``. Inferred from the
            file extension when omitted; required when the source has no
            recognizable extension.

    """

    source: str | Path
    type: str | None = None

    @property
    def is_remote(self) -> bool:
        """Whether the source is already addressable by the API (https/dp ref)."""
        s = str(self.source)
        return s.startswith(("https://", "http://", "dp://"))

    def resolved_type(self) -> str:
        """Return the media type, inferring it from the extension if needed.

        Raises:
            ValueError: If the type is not given and cannot be inferred.

        """
        if self.type is not None:
            return self.type
        suffix = Path(str(self.source).split("?")[0]).suffix.lower()
        inferred = EXTENSION_TYPES.get(suffix)
        if inferred is None:
            raise ValueError(
                f"Cannot infer media type of {self.source!r} from its extension; "
                "pass Media(..., type='image'|'audio'|'video') explicitly."
            )
        return inferred


@dataclass(frozen=True)
class Pair:
    """One pairwise-comparison item for :class:`~humanevals.HumanComparison`.

    ``a`` and ``b`` must both be text (``str``) or both be :class:`Media`
    of the same type, except that a video pair may carry an *image*
    ``reference`` (the image-to-video case, routed automatically).

    Attributes:
        a: First candidate. Reported scores are P(humans prefer ``a``).
        b: Second candidate.
        reference: Optional reference shown alongside the candidates
            (e.g. the source image both videos were generated from).
        context: Optional per-item context. Shown to annotators only if the
            scorer's ``instruction`` contains a ``{context}`` placeholder.

    """

    a: str | Media
    b: str | Media
    reference: Media | None = None
    context: str | None = None


@dataclass(frozen=True)
class RatingItem:
    """One item for :class:`~humanevals.HumanRating`.

    Attributes:
        subject: The thing to rate: text (``str``) or a single
            :class:`Media`. Text subjects are shown via the instruction's
            ``{context}`` placeholder, so a text subject requires the
            scorer's ``instruction`` to contain ``{context}`` and forbids
            also setting ``context``.
        context: Optional extra context for a *media* subject. Shown only if
            the instruction contains ``{context}``.
        reference: Optional non-selectable media shown alongside the subject
            (for example, the original image for an edited-image rating, or
            an image whose text description is being rated).

    """

    subject: str | Media
    context: str | None = None
    reference: Media | None = None


@dataclass(frozen=True)
class ChoiceItem:
    """One item for :class:`~humanevals.HumanMultipleChoice`.

    Attributes:
        question: The question shown to annotators (required by the API).
        options: The answer options, in display order (shuffled per
            annotator by default). Two or more required.
        subject: Optional single media item rendered above the options.
            Must be a local file or ``dp://`` ref; the API does not accept
            plain ``https://`` URLs for a choice subject.
        expected: The correct/expected option, if any, matched against the
            option text. When set, the item's score is the fraction of
            annotators who chose it; when ``None``, the score is ``None``
            and the vote distribution is reported in metadata.

    """

    question: str
    options: tuple[str, ...] | list[str]
    subject: Media | None = None
    expected: str | None = None


@dataclass(frozen=True)
class RankingItem:
    """One item for :class:`~humanevals.HumanRanking`.

    Attributes:
        candidates: Items to rank, all text or all same-type :class:`Media`.
            Two or more required (5-7 is a practical ceiling for annotators).
        expected_order: Optional expected ranking, best first, as indices
            into ``candidates``. When set, the score is rank agreement
            (Kendall's tau scaled to ``[0, 1]``) between the human consensus
            order and this order; when ``None``, the score is ``None`` and
            the consensus order is reported in metadata.
        context: Optional per-item context. Shown to annotators only if the
            scorer's ``instruction`` contains a ``{context}`` placeholder.

    """

    candidates: tuple[str | Media, ...] | list[str | Media]
    expected_order: tuple[int, ...] | list[int] | None = None
    context: str | None = None


@dataclass(frozen=True)
class JobProgress:
    """A point-in-time snapshot of a running (or finished) job.

    Attributes:
        job_id: Datapoint job id.
        status: ``processing | active | completed | failed | blocked | cancelled``.
        is_paused: Whether collection is paused (orthogonal to ``status``).
        total_datapoints: Datapoints in the job.
        completed_datapoints: Datapoints that reached the response target.
        failed_datapoints: Datapoints that failed ingestion.
        blocked_datapoints: Datapoints blocked by content moderation.
        total_responses: Billable responses collected so far. Live and not
            guaranteed monotonic; treat as an indicator, not a counter.
        cost_credits: Settled or currently-estimated cost in credits.
        refundable_credits: Credits returned if the job were cancelled now.
        errors: Per-datapoint failures, ``{"datapoint_index": int, "error": str}``.
        raw: The full API response, for anything not surfaced above.

    """

    job_id: str
    status: str
    is_paused: bool
    total_datapoints: int
    completed_datapoints: int
    failed_datapoints: int
    blocked_datapoints: int
    total_responses: int
    cost_credits: int | None
    refundable_credits: int | None
    errors: list[dict[str, Any]]
    raw: dict[str, Any] = field(repr=False)

    #: Job statuses after which nothing further will change.
    TERMINAL_STATUSES = frozenset({"completed", "failed", "blocked", "cancelled"})

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a final state (any of the four)."""
        return self.status in self.TERMINAL_STATUSES

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> JobProgress:
        """Build a snapshot from a ``GET /jobs/{id}`` response body."""
        return cls(
            job_id=payload["job_id"],
            status=payload["status"],
            is_paused=bool(payload.get("is_paused", False)),
            total_datapoints=payload.get("total_datapoints", 0),
            completed_datapoints=payload.get("completed_datapoints", 0),
            failed_datapoints=payload.get("failed_datapoints", 0),
            blocked_datapoints=payload.get("blocked_datapoints", 0),
            total_responses=payload.get("total_responses", 0),
            cost_credits=payload.get("cost_credits"),
            refundable_credits=payload.get("refundable_credits"),
            errors=payload.get("errors") or [],
            raw=payload,
        )

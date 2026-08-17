"""HumanEvals: human-backed scorers for your evals.

Real human judgments with the ergonomics of an evals library: scores come
back as autoevals-compatible ``Score`` objects, collected from the
`Datapoint <https://trydatapoint.com>`_ annotator pool.

Quickstart (free sandbox run)::

    import humanevals as he

    scorer = he.HumanComparison(
        "Which response answers the user's question better?",
        sandbox=True,  # free test pool; drop for real measurements
    )
    scores = scorer.eval_batch(
        [he.Pair(a=answer_from_model_a, b=answer_from_model_b)]
    )
    print(scores[0].score)  # P(humans prefer model A) in [0, 1]

Authentication: set ``DATAPOINT_API_KEY`` (or pass ``Client(api_key=...)``).
"""

from importlib.metadata import PackageNotFoundError, version

from .client import Client
from .exceptions import (
    APIError,
    AuthenticationError,
    BudgetExceededError,
    ContentBlockedError,
    DispatchFailedError,
    HumanEvalsError,
    InsufficientCreditsError,
    InvalidRequestError,
    JobFailedError,
    MediaTooLargeError,
    NetworkError,
    NotFoundError,
    PollTimeoutError,
    RateLimitError,
    ServerError,
)
from .job import EvalJob
from .scorers import (
    HumanComparison,
    HumanMultipleChoice,
    HumanRanking,
    HumanRating,
    HumanScorer,
)
from .types import (
    ChoiceItem,
    JobProgress,
    Media,
    Pair,
    RankingItem,
    RatingItem,
    Score,
)

try:
    __version__ = version("humanevals")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "APIError",
    "AuthenticationError",
    "BudgetExceededError",
    "ChoiceItem",
    "Client",
    "ContentBlockedError",
    "DispatchFailedError",
    "EvalJob",
    "HumanComparison",
    "HumanEvalsError",
    "HumanMultipleChoice",
    "HumanRanking",
    "HumanRating",
    "HumanScorer",
    "InsufficientCreditsError",
    "InvalidRequestError",
    "JobFailedError",
    "JobProgress",
    "Media",
    "MediaTooLargeError",
    "NetworkError",
    "NotFoundError",
    "Pair",
    "PollTimeoutError",
    "RankingItem",
    "RateLimitError",
    "RatingItem",
    "Score",
    "ServerError",
    "__version__",
]

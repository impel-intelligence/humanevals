# Contributing to HumanEvals

Thanks for your interest in contributing!

## Development setup

HumanEvals uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/impel-intelligence/humanevals
cd humanevals
uv sync --group dev
```

## Running checks

All of these must pass before a PR can merge (CI runs the same commands):

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # types (strict mode)
uv run pytest                # tests (no network access required)
```

## Guidelines

- **Tests never hit the network.** The suite uses `httpx.MockTransport`;
  every API interaction is exercised against recorded response shapes.
- **Public API changes need docs.** Update the README and docstrings in the
  same PR.
- **Keep dependencies minimal.** The runtime dependency set is `httpx` only.
  Adding a dependency needs a strong justification.
- **Changelog.** Add a line to `CHANGELOG.md` under the unreleased version
  for anything user-visible.

## Reporting issues

Please include the library version, Python version, and, for API errors,
the exception type, HTTP status code, and the `job_id` if one is involved.
Never include your API key in an issue.

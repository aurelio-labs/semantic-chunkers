# AGENTS.md

What an agent needs to work in this repository. Facts about the code and how to test it. The direction lives in `VISION.md`.

## Overview

semantic-chunkers splits text into semantically coherent chunks. A **splitter** turns a document into small units (sentences, by regex). A **chunker** groups those units into chunks using embedding similarity. Four chunkers exist:

| Chunker | Idea | File |
|---|---|---|
| `StatisticalChunker` | Rolling-window similarity with an automatically chosen threshold; also enforces token limits | `semantic_chunkers/chunkers/statistical.py` |
| `ConsecutiveChunker` | Compare each unit with the next; cut when similarity drops below a threshold | `semantic_chunkers/chunkers/consecutive.py` |
| `CumulativeChunker` | Compare the running chunk embedding with the next unit; cut when it diverges | `semantic_chunkers/chunkers/cumulative.py` |
| `RegexChunker` | No embeddings; cut on a regex and a token budget | `semantic_chunkers/chunkers/regex.py` |

`Chunk` (`semantic_chunkers/schema.py`) holds `splits`, `is_triggered`, `triggered_score`, `token_count`, `metadata`. `content` currently joins the splits with a space; the vision replaces that with exact text and offsets.

Encoders currently come from `semantic_router.encoders` (`DenseEncoder` subclasses with `__call__` and `acall`). The vision moves them in-house behind a protocol.

## Commands

The project uses `uv`. Prefix Python commands with `uv run`.

```bash
uv sync --extra dev          # install everything
make lint                    # ruff check, ruff format, mypy
make format                  # ruff --fix
make test                    # pytest with coverage
uv run pytest tests/unit/test_chunkers.py::test_statistical_chunker -vv
```

CI runs `make lint` on Python 3.13 and `make test` on 3.10 and 3.13. Conventional-commit PR titles are enforced by a separate workflow.

## Layout

```
semantic_chunkers/
  __init__.py          public exports and __version__
  schema.py            Chunk
  chunkers/            base.py (BaseChunker), statistical.py, consecutive.py, cumulative.py, regex.py
  splitters/           base.py (BaseSplitter), regex.py (RegexSplitter)
  utils/               logger.py, text.py (token counting via tiktoken)
tests/unit/            unit tests with mocked encoders
docs/                  notebooks and markdown guides (get-started/ is published to docs.aurelio.ai)
```

## Testing rules

- Unit tests (`tests/unit/`) may mock the encoder. The existing tests construct an `OpenAIEncoder` with a fake key and then replace `chunker.encoder` with a `Mock` whose return value is a small numpy array.
- Functional tests (`tests/functional/`, to be created) run a real small local encoder with cached embeddings so they are deterministic and need no keys.
- Anything that calls a paid API is marked `@pytest.mark.live` and is not part of `make test`.
- Every chunker has sync and async tests, and they assert the same outputs.
- Test names describe behaviour: `test_statistical_chunker_respects_max_split_tokens`, not `test_chunker_2`.

## Conventions

- Python 3.10 minimum; keep `typing` compatible with it (no `match`, no `Self` without `typing_extensions`).
- ruff with line length 88; mypy with `ignore_missing_imports`. `make lint` must pass.
- Pydantic models for public objects. The code base is mid-migration from `pydantic.v1` to v2; new code uses v2 (`model_config`, `field_validator`).
- Docstrings on public classes and methods, with a short example where the call shape is not obvious.
- Conventional commits. Scopes in use: `chunkers`, `splitters`, `encoders`, `schema`, `bench`, `docs`, `ci`.

## Gotchas

- `semantic_chunkers/__init__.py` hard-codes `__version__`, and it disagrees with `pyproject.toml`. Read the version from package metadata when you touch it.
- `StatisticalChunker` mixes the algorithm, batching, statistics, and matplotlib plotting in one 680-line file. Plotting imports are lazy; keep them that way or move them out.
- `BaseChunker` accepts `encoder=None` and silently substitutes a bare `DenseEncoder(name="default")`, which cannot encode. Do not rely on that default.
- Token counting uses tiktoken's `cl100k_base` regardless of the encoder in use.
- The async path (`acall`) exists on every chunker but the base class's `__call__` raises `NotImplementedError`; issue 32 reports hitting that from a script.
- `.github/workflows/visionary-*.yml` are callers into the visionary framework. Do not edit them as part of a library change.

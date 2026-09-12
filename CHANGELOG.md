# Changelog

All notable changes to semantic-chunkers. Breaking changes are listed under **Breaking** with a before-and-after snippet, as VISION.md requires.

## Unreleased

### Breaking
- `Chunk`, `BaseChunker`, and `BaseSplitter` are native pydantic v2 models instead of `pydantic.v1` shim models. Nesting them inside a `pydantic.v1` model no longer validates, and subclasses that declared a `class Config` should switch to `model_config`.

  ```python
  # before
  from pydantic.v1 import BaseModel
  from semantic_chunkers.schema import Chunk

  class Result(BaseModel):
      chunk: Chunk

  # after
  from pydantic import BaseModel
  from semantic_chunkers.schema import Chunk

  class Result(BaseModel):
      chunk: Chunk
  ```

- `Chunk` field validation is stricter, because pydantic v2 does not coerce as loosely as the v1 shim did. A `token_count` float with a fractional part used to be truncated to an `int`; it now raises `ValidationError`. Convert before constructing the chunk. Whole floats such as `5.0` are still accepted, and boolean-like strings for `is_triggered` still coerce, so neither of those needs a change.

  ```python
  # before: token_count == 5
  Chunk(splits=["a"], token_count=5.7)

  # after: ValidationError, so convert first
  Chunk(splits=["a"], token_count=int(5.7))
  ```

### Changed
- `semantic_chunkers.__version__` is read from the installed package metadata instead of a hard-coded string that had drifted from `pyproject.toml`.

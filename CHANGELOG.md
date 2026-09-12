# Changelog

All notable changes to semantic-chunkers. Breaking changes are listed under **Breaking** with a before-and-after snippet, as VISION.md requires.

## Unreleased

### Changed
- `Chunk`, `BaseChunker`, and `BaseSplitter` are now native pydantic v2 models instead of using the `pydantic.v1` compatibility shim. Subclasses that declared a `class Config` should use `model_config = ConfigDict(...)`.
- `semantic_chunkers.__version__` is read from the installed package metadata instead of a hard-coded string that had drifted from `pyproject.toml`.

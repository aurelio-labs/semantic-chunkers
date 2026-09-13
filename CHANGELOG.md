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

- `RegexChunker` returns one list of chunks per document instead of always returning a single list. It also no longer lets a chunk span two documents. Code that passed one document and read `chunks[0]` is unaffected; code that passed several and read `chunks[0]` was reading every document's chunks flattened into one list, with the boundary between documents buried inside a chunk.

  ```python
  chunker = RegexChunker(max_chunk_tokens=300)
  docs = ["Alpha one. Alpha two.", "Beta one. Beta two."]

  # before: one list for two documents, and one chunk holding both
  chunks = chunker(docs)
  len(chunks)                 # 1
  chunks[0][0].splits         # ['Alpha one.', 'Alpha two.', 'Beta one.', 'Beta two.']

  # after: one list per document, matching every other chunker
  chunks = chunker(docs)
  len(chunks)                 # 2
  chunks[0][0].splits         # ['Alpha one.', 'Alpha two.']
  chunks[1][0].splits         # ['Beta one.', 'Beta two.']
  ```

- `RegexSplitter` applies each delimiter to the output of the previous one instead of re-splitting the whole document every pass, so two or more delimiters no longer duplicate the document. A compiled `regex.Pattern` passed as a delimiter is now used for the split; previously it was ignored and the built-in sentence pattern was used in its place.

  ```python
  splitter = RegexSplitter()
  pattern = regex.compile(splitter.regex_pattern, flags=regex.VERBOSE)

  # before: every piece repeated once per delimiter
  splitter("Alpha one. Alpha two.\nBeta one. Beta two.", delimiters=["\n", pattern])
  # ['Alpha one.', 'Alpha two.', 'Beta one.', 'Beta two.',
  #  'Alpha one.', 'Alpha two.', 'Beta one.', 'Beta two.']

  # after
  # ['Alpha one.', 'Alpha two.', 'Beta one.', 'Beta two.']

  # before: a custom pattern was ignored
  splitter("a1b2c3", delimiters=[regex.compile(r"\d")])   # ['a1b2c3']
  # after
  # ['a', 'b', 'c']
  ```

- Plotting moved out of `StatisticalChunker` and into `semantic_chunkers.stats`, behind the `stats` extra. `chunker.plot_chunks = True` and both `plot_*` methods still work, but with matplotlib missing they now raise `ImportError` naming the extra instead of logging a warning and silently drawing nothing. The old warning pointed at `semantic-router[processing]`, which is neither the right package nor the right extra. `ChunkStatistics` moved with them; import it from `semantic_chunkers.stats` rather than `semantic_chunkers.chunkers.statistical`.

  ```python
  from semantic_chunkers import StatisticalChunker

  chunker = StatisticalChunker(encoder=encoder, plot_chunks=True)

  # before, without matplotlib: a warning in the log, no plot, chunks returned
  chunks = chunker(docs)

  # after, without matplotlib
  # ImportError: Plotting requires matplotlib, which is not installed by
  # default. Install it with `pip install semantic-chunkers[stats]`.
  ```

### Changed
- `semantic_chunkers.__version__` is read from the installed package metadata instead of a hard-coded string that had drifted from `pyproject.toml`.
- The mutable default argument `delimiters=[]` on `RegexChunker.__init__` and `RegexSplitter.__call__` is now `None`. The first call to `RegexSplitter` with no delimiters used to append the compiled sentence pattern into the shared signature default, where it stayed for the life of the process.

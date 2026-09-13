# Vision

This is the owner's steering document for semantic-chunkers. Every visionary role reads it before acting. The planner treats it as the authority on what to propose; the reviewer flags changes that move away from it. Editing it on `main` re-runs the planner.

## What this project is

semantic-chunkers is a small Python library that splits content into semantically coherent chunks for retrieval and LLM pipelines. It offers a few genuinely different algorithms behind one interface, works with any embedding model, and is honest about what each method costs. Text is the main case; video and audio are first-class, not an afterthought. It is not a document loader, not a RAG framework, and not a wrapper around one vendor's API.

## What good looks like

- A `Chunk` carries the exact original content and its offsets into the source. Joining a document's chunks reproduces the document exactly.
- Every chunker has a number. Boundary quality and cost on a shared benchmark suite, published in the docs, so a user can pick a chunker from data instead of a hunch.
- Chunking a thousand ordinary documents with a small local encoder takes seconds, not hours. Issue 18 measured 13,124 seconds against 8.8 for a greedy competitor; that gap closes.
- **Importing is fast and installing is small.** `import semantic_chunkers` is measured in milliseconds, and `pip install semantic-chunkers` pulls a short list of light dependencies. Anything heavy is an extra, and heavy means it, such as the 2 GB that sentence-transformers brings.
- **The API is small enough to hold in your head.** Chunkers, splitters, encoders, `Chunk`. A user should get a good result from one import and one call, with every parameter optional and sensibly defaulted. A new concept has to earn its place by removing more confusion than it adds.
- **The code reads top to bottom.** Straight-line code over indirection, with single-line comments at the steps that are not obvious, so a reader follows the algorithm without jumping between files. A helper exists to remove duplication, not to name a step. Prefer one clear module over a package of small ones.
- **Multimodal works, not just in a notebook.** The same chunkers, the same `Chunk`, and the same benchmark treatment for video and audio as for text.
- Sync and async paths behave identically and are tested together against a real, small, local encoder. Mocks are for unit tests of pure logic only.
- Every public method has a docstring with an example. Every algorithm has a method page that explains how it decides where to cut.

## Priorities when they conflict

1. Quality of chunk boundaries, as measured by the benchmark.
2. Latency and weight: import time, install size, encoder calls, wall-clock per thousand tokens, memory. A change that makes the library heavier needs to buy something in priority 1.
3. Simplicity of the public API. Prefer fewer names, fewer required arguments, and fewer ways to do one thing.
4. API stability. Release 0.2 may break compatibility where the first three priorities require it, and each break is listed in the changelog with a before-and-after snippet. After 0.2, a breaking change needs a human decision on the proposal before it is implemented.

## Current direction (September 2026)

The target is 0.2.0. In rough order:

1. **Own the encoders.** semantic-chunkers stops importing from semantic-router. It defines a minimal encoder protocol (`__call__(docs) -> array`, `acall(docs)`) plus an adapter for any callable, so anything satisfying the protocol keeps working and existing semantic-router encoder objects still plug in by duck typing. The OpenAI encoder is first-party and works from a bare `pip install semantic-chunkers`: it calls the embeddings endpoint over `httpx` rather than depending on the `openai` SDK, with a configurable base URL so Azure and OpenAI-compatible endpoints work. sentence-transformers stays behind `[local]` because of its size. Encoders are imported lazily so a local-only user never pays for the HTTP stack, and vice versa. Reason: the library is pinned to a large dependency for one base class, users hit install conflicts (issues 19, 29, 34), and encoder changes upstream can break this repo.
2. **Exact chunks.** `Chunk.content` is the original content, not splits joined with a space, with `start` and `end` offsets into the source sequence: character positions for text, element indices for a frame or sample sequence. Splits stay available. Both are optional, so a chunker that has no sliceable source leaves them unset rather than failing to construct. The benchmark depends on this, and so does anyone who wants to highlight a chunk in the source.
3. **A benchmark harness** in `benchmarks/` with `make bench` writing results in the visionary contract. Synthetic boundary suite first (concatenated passages with known seams), then section-boundary datasets, then retrieval recall over answer spans. Baselines for every chunker before any algorithm changes are proposed.
4. **Performance.** Batch encoding across documents, cached embeddings keyed on model and text, and a cheaper threshold search in the statistical chunker. Import time and install size are measured the same way, with a regression check in the benchmark. Measured against the harness, not asserted.
5. **Multimodal for real.** Every chunker accepts a sequence of already-decoded elements, not just `str`, so video frames and audio segments work everywhere rather than in `ConsecutiveChunker` alone. A benchmark suite with known boundaries to score it, and a documented path from element index to timestamp for a caller who has the frame rate. Today the README promises text, video and audio while only one chunker handles frames and nothing handles audio; that gap closes or the claim goes.
6. **Housekeeping that unblocks the rest.** Pydantic v2 native. Python 3.10 through 3.14. Looser pins. `__version__` from package metadata. Plotting out of the core module and into the `[stats]` extra. The async path that raises `NotImplementedError` from a script (issue 32) fixed with a functional test.
7. **Docs that explain the methods.** One page per algorithm with its benchmark numbers, closing issue 35.

Later, only with benchmark support: structure-aware splitters for markdown and code.

## Non-goals

- Decoding or parsing sources. PDF, HTML, OCR, video containers and audio codecs are someone else's job. Input is text, or a sequence of elements the caller has already decoded; the library never depends on a parser, `cv2`, or `ffmpeg`.
- Vector store or framework integrations. Chunks are plain objects; integrating them is the caller's business.
- LLM-driven chunking as a default path. Too slow and too expensive for the core. It is a legitimate experiment to compare against.
- Plotting in the core module. It drags in matplotlib and hides the algorithm.
- Configuration surface for its own sake. A new argument needs a user who is stuck without it.

## Rules for agents

- Every change to an algorithm comes with a benchmark delta in the PR. Once the harness exists, no delta means no merge.
- Breaking changes are allowed for 0.2 and each one is listed under "Breaking" in `CHANGELOG.md` with a before-and-after snippet.
- A new core dependency must be light, widely installed, and needed on the common path; say in the PR body what it costs in install size and import time. Anything heavy is an extra. Import it lazily if only some users need it.
- Write code a reader can follow top to bottom. Single-line comments where the reason is not obvious from the line. No abstraction that is used once.
- Functional tests use a real small local encoder with cached embeddings so they are deterministic and free. Tests that call a paid API are marked `live` and excluded from the default suite.
- Sync and async are one feature. A change to one without the other is incomplete.
- A change that only works for `str` input is incomplete unless the PR says why the other modalities are out of scope.
- Do not edit the notebooks under `docs/` to reflect API changes; update the markdown guides under `docs/` and the docs site pulls from those. Notebooks get a separate pass once 0.2 is out.
- Conventional commit titles are enforced on pull requests.

# Vision

This is the owner's steering document for semantic-chunkers. Every visionary role reads it before acting. The planner treats it as the authority on what to propose; the reviewer flags changes that move away from it. Editing it on `main` re-runs the planner.

## What this project is

semantic-chunkers is a small Python library that splits text into semantically coherent chunks for retrieval and LLM pipelines. It offers a few genuinely different algorithms behind one interface, works with any embedding model, and is honest about what each method costs. It is not a document loader, not a RAG framework, and not a wrapper around one vendor's API.

## What good looks like

- A `Chunk` carries the exact original text and its character offsets. Joining a document's chunks reproduces the document exactly.
- Every chunker has a number. Boundary quality and cost on a shared benchmark suite, published in the docs, so a user can pick a chunker from data instead of a hunch.
- Chunking a thousand ordinary documents with a small local encoder takes seconds, not hours. Issue 18 measured 13,124 seconds against 8.8 for a greedy competitor; that gap closes.
- Sync and async paths behave identically and are tested together against a real, small, local encoder. Mocks are for unit tests of pure logic only.
- The public surface is small and named plainly: chunkers, splitters, encoders, `Chunk`. Every public method has a docstring with an example. Every algorithm has a method page that explains how it decides where to cut.
- Installing is boring. `pip install semantic-chunkers` pulls a handful of dependencies. Encoders are extras.

## Priorities when they conflict

1. Quality of chunk boundaries, as measured by the benchmark.
2. Speed and cost: encoder calls, wall-clock per thousand tokens, memory.
3. API stability. Release 0.2 may break compatibility where the first two priorities require it, and each break is listed in the changelog with a before-and-after snippet. After 0.2, a breaking change needs a human decision on the proposal before it is implemented.

## Current direction (September 2026)

The target is 0.2.0. In rough order:

1. **Own the encoders.** semantic-chunkers stops importing from semantic-router. It defines a minimal encoder protocol (`__call__(docs) -> array`, `acall(docs)`) and ships first-party implementations for OpenAI and sentence-transformers behind extras (`[openai]`, `[local]`), plus an adapter for any callable. Anything that satisfies the protocol keeps working, so existing semantic-router encoder objects still plug in by duck typing. Reason: the library is pinned to a large dependency for one base class, users hit install conflicts (issues 19, 29, 34), and encoder changes upstream can break this repo.
2. **Exact chunks.** `Chunk.content` is the original text, not splits joined with a space. `start` and `end` offsets. Splits stay available. The benchmark depends on this, and so does anyone who wants to highlight a chunk in the source.
3. **A benchmark harness** in `benchmarks/` with `make bench` writing results in the visionary contract. Synthetic boundary suite first (concatenated passages with known seams), then section-boundary datasets, then retrieval recall over answer spans. Baselines for every chunker before any algorithm changes are proposed.
4. **Performance.** Batch encoding across documents, cached embeddings keyed on model and text, and a cheaper threshold search in the statistical chunker. Measured against the harness, not asserted.
5. **Housekeeping that unblocks the rest.** Pydantic v2 native. Python 3.10 through 3.14. Looser pins. `__version__` from package metadata. Plotting out of the core module and into the `[stats]` extra. The async path that raises `NotImplementedError` from a script (issue 32) fixed with a functional test.
6. **Docs that explain the methods.** One page per algorithm with its benchmark numbers, closing issue 35.

Later, only with benchmark support: structure-aware splitters for markdown and code, and the multimodal path the video notebook hints at.

## Non-goals

- Parsing documents. PDF, HTML, and OCR are someone else's job. Input is text.
- Vector store or framework integrations. Chunks are plain objects; integrating them is the caller's business.
- LLM-driven chunking as a default path. Too slow and too expensive for the core. It is a legitimate experiment to compare against.
- Plotting in the core module. It drags in matplotlib and hides the algorithm.

## Rules for agents

- Every change to an algorithm comes with a benchmark delta in the PR. Once the harness exists, no delta means no merge.
- Breaking changes are allowed for 0.2 and each one is listed under "Breaking" in `CHANGELOG.md` with a before-and-after snippet.
- No new core dependencies. Extras only, with the reason in the PR body.
- Functional tests use a real small local encoder with cached embeddings so they are deterministic and free. Tests that call a paid API are marked `live` and excluded from the default suite.
- Sync and async are one feature. A change to one without the other is incomplete.
- Do not edit the notebooks under `docs/` to reflect API changes; update the markdown guides under `docs/` and the docs site pulls from those. Notebooks get a separate pass once 0.2 is out.
- Conventional commit titles are enforced on pull requests.

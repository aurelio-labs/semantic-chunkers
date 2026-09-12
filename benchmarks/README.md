# Benchmarks

`make bench` runs every variant in `experiments/default.json` on every suite and writes `benchmarks/results.json` in the [visionary results contract](https://github.com/jamescalam/visionary/blob/main/plugin/skills/experiment/results-schema.md). The bench workflow runs it on each pull request and on the merge base and comments the delta.

## Suites

**synthetic-boundaries.** Concatenate the introductions of three to six unrelated Wikipedia articles and record the seams. The chunker sees the concatenated text; the gold boundaries are the sentence indices where a new article begins. Metrics: boundary precision, recall, and F1 with a one-sentence tolerance; Pk and WindowDiff (lower is better); `wall_s`, the variant's chunker work plus the embedding of its unique texts, measured in the variant's own cache namespace so no variant benefits from another's embeddings and order does not matter; note the cache dedupes sentences repeated across documents within a variant too, so this is unique-text embedding cost, not the cost of every request (cold on CI, where deltas are computed; warm on a repeat local run of the same variant); `encoder_requests` and `encoder_texts_requested`, what a user would pay, cache or not; `encoder_model_calls` and `encoder_model_texts`, cache misses only, and note the cache also dedupes sentences repeated across documents, so `encoder_seconds` is unique-text cost rather than user cost; chunks per document and, for chunkers that record it, mean chunk tokens.

Planned: section boundaries from Wikipedia-derived segmentation sets, and retrieval recall over answer spans.

## Encoders

`all-MiniLM-L6-v2` through sentence-transformers on CPU, wrapped as a semantic-router `DenseEncoder` with a SQLite embedding cache in `benchmarks/.cache/`. The cache is namespaced by variant, so a repeat run of an unchanged variant is free while a new variant pays its own way. Delete `benchmarks/.cache/` for a fully cold run. A variant that raises is recorded with an `error` field, the rest of the table is still written, and the command exits non-zero.

## Running one experiment

Copy `experiments/default.json`, change the variants or suite parameters, and run `uv run python -m benchmarks.run experiments/<name>.json --out /tmp/results.json`.

## Data

`data/wiki_intros.json` holds 46 article introductions fetched by `build_corpus.py` from the English Wikipedia REST API. Text is © Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each entry records its source URL.

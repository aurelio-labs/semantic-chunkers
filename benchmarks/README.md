# Benchmarks

`make bench` runs every variant in `experiments/default.json` on every suite and writes `benchmarks/results.json` in the [visionary results contract](https://github.com/jamescalam/visionary/blob/main/plugin/skills/experiment/results-schema.md). The bench workflow runs it on each pull request and on the merge base and comments the delta.

## Suites

**synthetic-boundaries.** Concatenate the introductions of three to six unrelated Wikipedia articles and record the seams. The chunker sees the concatenated text; the gold boundaries are the sentence indices where a new article begins. Metrics: boundary precision, recall, and F1 with a one-sentence tolerance; Pk and WindowDiff (lower is better); wall time with the model already loaded; encoder requests and texts requested (what a user would pay, cache or not); model calls and texts actually embedded (cache misses); chunks per document and mean chunk tokens.

Planned: section boundaries from Wikipedia-derived segmentation sets, and retrieval recall over answer spans.

## Encoders

`all-MiniLM-L6-v2` through sentence-transformers on CPU, wrapped as a semantic-router `DenseEncoder` with a SQLite embedding cache in `benchmarks/.cache/`. After the first run a parameter sweep costs no encoder calls.

## Running one experiment

Copy `experiments/default.json`, change the variants or suite parameters, and run `uv run python -m benchmarks.run experiments/<name>.json --out /tmp/results.json`.

## Data

`data/wiki_intros.json` holds 46 article introductions fetched by `build_corpus.py` from the English Wikipedia REST API. Text is © Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each entry records its source URL.

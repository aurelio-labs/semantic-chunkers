# Benchmarks

`make bench` runs every variant in `experiments/default.json` on every suite, writes `benchmarks/results.json` in the [visionary results contract](https://github.com/jamescalam/visionary/blob/main/plugin/skills/experiment/results-schema.md), and renders `benchmarks/report.html`. The bench workflow runs it on each pull request and on the merge base and comments the delta.

## Reading the results

Open `benchmarks/report.html`. It is one self-contained file — inline SVG, no JavaScript, no external assets, light and dark — with the headline F1, quality and cost charts, a boundary-F1 trend across past runs, and a table holding every number the charts show. Every value in a chart is also in that table, so nothing is reachable only by hovering.

`make bench_report` re-renders the report from an existing `results.json` without re-running the suite or recording a run.

**History.** Each `make bench` appends its run to `benchmarks/history.jsonl`, one JSON object per run keyed by commit; re-running the same commit replaces its row. The trend chart plots the last twelve. The file is git-ignored, so history is per-checkout: it accumulates locally and starts empty on a fresh CI runner. Point `--history` at a shared path to collect runs somewhere durable.

**Percentiles.** The per-document metrics carry a distribution as well as a mean, because a variant with a good average can still segment one document in twenty very badly. `pk`, `windowdiff` and per-document latency report `_p50` and `_p95`, where p95 is the bad tail. Boundary F1 is higher-is-better, so its bad tail is the low end: it reports `_p05`, `_p50` and `_p95`. `wall_s` stays the whole-suite total; `doc_s_p50` and `doc_s_p95` are per document.

## Suites

**synthetic-boundaries.** Concatenate the introductions of three to six unrelated Wikipedia articles and record the seams. The chunker sees the concatenated text; the gold boundaries are the sentence indices where a new article begins. Metrics: boundary precision, recall, and F1 with a one-sentence tolerance; Pk and WindowDiff (lower is better); `wall_s`, the variant's chunker work plus the embedding of its unique texts, measured in the variant's own cache namespace so no variant benefits from another's embeddings and order does not matter; note the cache dedupes sentences repeated across documents within a variant too, so this is unique-text embedding cost, not the cost of every request (cold on CI, where deltas are computed; warm on a repeat local run of the same variant); `encoder_requests` and `encoder_texts_requested`, what a user would pay, cache or not; `encoder_model_calls` and `encoder_model_texts`, cache misses only, and note the cache also dedupes sentences repeated across documents, so `encoder_seconds` is unique-text cost rather than user cost; chunks per document and, for chunkers that record it, mean chunk tokens.

Planned: section boundaries from Wikipedia-derived segmentation sets, and retrieval recall over answer spans.

## Encoders

`all-MiniLM-L6-v2` through sentence-transformers on CPU, wrapped as a semantic-router `DenseEncoder` with a SQLite embedding cache in `benchmarks/.cache/`. The cache is namespaced by variant, so a repeat run of an unchanged variant is free while a new variant pays its own way. Delete `benchmarks/.cache/` for a fully cold run. A variant that raises is recorded with an `error` field and the rest of the table is still written. The command exits non-zero only when every variant failed: the bench workflow runs it under `set -e`, so failing on a partial run would kill the step before the delta comment is rendered and the `failed:` row carrying the error would never reach the pull request.

## Running one experiment

Copy `experiments/default.json`, change the variants or suite parameters, and run `uv run python -m benchmarks.run experiments/<name>.json --out /tmp/results.json`.

## Data

`data/wiki_intros.json` holds 46 article introductions fetched by `build_corpus.py` from the English Wikipedia REST API. Text is © Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each entry records its source URL.

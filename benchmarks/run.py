"""Run the benchmark and write results in the visionary results contract.

    uv run python -m benchmarks.run [experiments/<config>.json] [--out benchmarks/results.json]

A config lists variants. Each variant names a chunker, its parameters, and
an encoder. Every variant runs on every suite in the config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks import metrics, synthetic
from benchmarks.encoders import CachedSentenceTransformerEncoder

if TYPE_CHECKING:  # the runtime import stays inside make_chunker, see below
    from semantic_chunkers.chunkers.base import BaseChunker
    from semantic_chunkers.schema import Chunk

ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "experiments" / "default.json"
DIRECTIONS = {
    "boundary_f1": "up",
    "boundary_f1_p05": "up",  # the weak tail: 1 doc in 20 scores at or below this
    "boundary_f1_p50": "up",
    "boundary_f1_p95": "up",
    "boundary_precision": "up",
    "boundary_recall": "up",
    "pk": "down",
    "pk_p50": "down",
    "pk_p95": "down",  # the bad tail for an error rate
    "windowdiff": "down",
    "windowdiff_p50": "down",
    "windowdiff_p95": "down",
    # chunker work plus embedding, in a cache namespace per variant and
    # document, so no variant and no document is served free by an earlier
    # one. Texts repeated within a single document still dedupe.
    "wall_s": "down",
    # per document, on equal footing: every document embeds its own text.
    "doc_s_p50": "down",
    "doc_s_p95": "down",  # the slow tail
    "encoder_requests": "down",
    "encoder_texts_requested": "down",
    "encoder_model_calls": None,
    "encoder_model_texts": None,
    "encoder_seconds": None,
    "chunks_per_doc": None,
    "mean_chunk_tokens": None,
}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


_MODELS: dict[str, Any] = {}


def make_encoder(
    spec: dict[str, Any], namespace: str
) -> CachedSentenceTransformerEncoder:
    """One encoder per variant, sharing one loaded model per model name."""
    name = spec.get("name", "all-MiniLM-L6-v2")
    encoder = CachedSentenceTransformerEncoder(name=name, namespace=namespace)
    if name in _MODELS:
        encoder._model = _MODELS[name]
    else:
        encoder.warm_up()
        _MODELS[name] = encoder._model
    return encoder


def uses_encoder(variant: dict[str, Any]) -> bool:
    return variant["chunker"] != "regex"


def make_chunker(
    spec: dict[str, Any], encoder: CachedSentenceTransformerEncoder | None
) -> BaseChunker:
    from semantic_chunkers import (
        ConsecutiveChunker,
        CumulativeChunker,
        RegexChunker,
        StatisticalChunker,
    )

    kind = spec["chunker"]
    params = dict(spec.get("params", {}))
    if kind == "regex":
        return RegexChunker(**params)
    if encoder is None:
        # BaseChunker would otherwise substitute a bare DenseEncoder that
        # cannot encode, and the variant would fail deep in the chunker.
        raise ValueError(f"chunker {kind} needs an encoder")
    if kind == "statistical":
        return StatisticalChunker(encoder=encoder, **params)
    if kind == "consecutive":
        return ConsecutiveChunker(encoder=encoder, **params)
    if kind == "cumulative":
        return CumulativeChunker(encoder=encoder, **params)
    raise ValueError(f"unknown chunker {kind}")


def predicted_boundaries(chunks: list[Chunk], sentences: list[str]) -> list[int]:
    """Map chunk starts back to sentence indices.

    Chunks are made of the same sentence units the synthetic suite produced,
    so the first split of each chunk is located by walking the sentence list
    in order.
    """
    boundaries: list[int] = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        if not chunk.splits:
            continue
        first = chunk.splits[0].strip()
        # advance to the sentence matching this chunk's first split
        while cursor < len(sentences) and sentences[cursor].strip() != first:
            cursor += 1
        if cursor >= len(sentences):
            raise ValueError(
                f"chunk {index} starts with a split that matches no remaining sentence: "
                f"{first[:60]!r}. Boundaries after this point would be lost, so the run "
                "stops rather than publish a score for the wrong segmentation."
            )
        if cursor > 0:
            boundaries.append(cursor)
        cursor += len(chunk.splits)
    return boundaries


def run_synthetic(variant: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    docs = synthetic.build(
        n_docs=suite.get("n_docs", 30),
        min_sources=suite.get("min_sources", 3),
        max_sources=suite.get("max_sources", 6),
        seed=suite.get("seed", 0),
    )
    suite_name = suite.get("name", "suite")
    encoder = (
        make_encoder(
            variant.get("encoder", {}), namespace=f"{suite_name}/{variant['name']}"
        )
        if uses_encoder(variant)
        else None
    )
    chunker = make_chunker(variant, encoder)
    if encoder is not None:
        encoder.reset_counters()
    pks, wds, ps, rs, f1s, doc_s = [], [], [], [], [], []
    n_chunks = 0
    chunk_tokens: list[int | None] = []
    t0 = time.perf_counter()
    for index, doc in enumerate(docs):
        if encoder is not None:
            # Each document gets its own cache partition, so it pays for its
            # own embeddings whatever the documents before it embedded. The
            # synthetic suite reuses source articles, so without this the
            # first few documents carry almost all the embedding cost and
            # doc_s_p50/doc_s_p95 rank documents by position, not difficulty.
            # The suite name is part of the partition too: two suites can
            # share text, and whichever ran first would otherwise be billed
            # for both.
            encoder.use_namespace(f"{suite_name}/{variant['name']}/{index}")
        d0 = time.perf_counter()
        result = chunker([doc.text])
        doc_s.append(time.perf_counter() - d0)
        chunks = result[0]
        hyp = predicted_boundaries(chunks, doc.sentences)
        n = len(doc.sentences)
        pks.append(metrics.pk(doc.boundaries, hyp, n))
        wds.append(metrics.windowdiff(doc.boundaries, hyp, n))
        p, r, f = metrics.boundary_prf(
            doc.boundaries, hyp, n, tolerance=suite.get("tolerance", 1)
        )
        ps.append(p)
        rs.append(r)
        f1s.append(f)
        n_chunks += len(chunks)
        chunk_tokens.extend(c.token_count for c in chunks)
    wall = time.perf_counter() - t0
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    pct = lambda xs, q, nd=4: round(metrics.percentile(xs, q), nd)  # noqa: E731
    out = {
        # Means are the headline; the percentiles describe the spread across
        # documents, because a variant with a good average can still segment
        # one document in twenty very badly.
        "boundary_f1": mean(f1s),
        "boundary_f1_p05": pct(f1s, 5),
        "boundary_f1_p50": pct(f1s, 50),
        "boundary_f1_p95": pct(f1s, 95),
        "boundary_precision": mean(ps),
        "boundary_recall": mean(rs),
        "pk": mean(pks),
        "pk_p50": pct(pks, 50),
        "pk_p95": pct(pks, 95),
        "windowdiff": mean(wds),
        "windowdiff_p50": pct(wds, 50),
        "windowdiff_p95": pct(wds, 95),
        "wall_s": round(wall, 3),
        "doc_s_p50": pct(doc_s, 50, 4),
        "doc_s_p95": pct(doc_s, 95, 4),
        "chunks_per_doc": round(n_chunks / len(docs), 3),
    }
    measured = [t for t in chunk_tokens if t is not None]
    if measured:
        # Consecutive and cumulative chunkers do not set token_count; leave
        # the metric out rather than publish a zero that means "not measured".
        out["mean_chunk_tokens"] = mean(measured)
    if encoder is not None:
        out.update(encoder.counters())
    return out


SUITES = {"synthetic-boundaries": run_synthetic}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "results.json"))
    args = ap.parse_args(argv)
    config = json.loads(Path(args.config).read_text())

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for suite in config["suites"]:
        runner = SUITES[suite["name"]]
        for variant in config["variants"]:
            print(f"{suite['name']} :: {variant['name']}", file=sys.stderr)
            entry: dict[str, Any] = {
                "name": suite["name"],
                "variant": variant["name"],
                "config": {
                    **{k: v for k, v in variant.items() if k != "name"},
                    "suite": {k: v for k, v in suite.items() if k != "name"},
                },
                "metrics": {},
            }
            try:
                entry["metrics"] = runner(variant, suite)
                print(json.dumps(entry["metrics"]), file=sys.stderr)
            except Exception as exc:  # one broken variant must not lose the table
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(variant["name"])
                print(f"ERROR {variant['name']}: {entry['error']}", file=sys.stderr)
            results.append(entry)

    payload = {
        "schema": 1,
        "commit": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "directions": {k: v for k, v in DIRECTIONS.items() if v},
        "suites": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    if failures:
        print(
            f"{len(failures)} variant(s) failed: {', '.join(failures)}", file=sys.stderr
        )
    # A partial failure exits 0 on purpose. The bench action runs this command
    # under `set -e`, so a non-zero exit kills the step before the delta is
    # rendered and the `failed:` row carrying the error never reaches the PR.
    # Only a run that produced no usable row at all is a command failure.
    return 1 if len(failures) == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())

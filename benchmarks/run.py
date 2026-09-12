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
from typing import Any

from benchmarks import metrics, synthetic
from benchmarks.encoders import CachedSentenceTransformerEncoder

ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "experiments" / "default.json"
DIRECTIONS = {
    "boundary_f1": "up",
    "boundary_precision": "up",
    "boundary_recall": "up",
    "pk": "down",
    "windowdiff": "down",
    "wall_s": "down",
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


def make_encoder(spec: dict[str, Any]) -> CachedSentenceTransformerEncoder:
    return CachedSentenceTransformerEncoder(name=spec.get("name", "all-MiniLM-L6-v2"))


def make_chunker(spec: dict[str, Any], encoder):
    from semantic_chunkers import (
        ConsecutiveChunker,
        CumulativeChunker,
        RegexChunker,
        StatisticalChunker,
    )

    kind = spec["chunker"]
    params = dict(spec.get("params", {}))
    if kind == "statistical":
        return StatisticalChunker(encoder=encoder, **params)
    if kind == "consecutive":
        return ConsecutiveChunker(encoder=encoder, **params)
    if kind == "cumulative":
        return CumulativeChunker(encoder=encoder, **params)
    if kind == "regex":
        return RegexChunker(**params)
    raise ValueError(f"unknown chunker {kind}")


def predicted_boundaries(chunks, sentences: list[str]) -> list[int]:
    """Map chunk starts back to sentence indices.

    Chunks are made of the same sentence units the synthetic suite produced,
    so the first split of each chunk is located by walking the sentence list
    in order.
    """
    boundaries: list[int] = []
    cursor = 0
    for chunk in chunks:
        if not chunk.splits:
            continue
        first = chunk.splits[0].strip()
        # advance to the sentence matching this chunk's first split
        while cursor < len(sentences) and sentences[cursor].strip() != first:
            cursor += 1
        if cursor >= len(sentences):
            break
        if cursor > 0:
            boundaries.append(cursor)
        cursor += len(chunk.splits)
    return boundaries


def run_synthetic(variant: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    encoder = make_encoder(variant.get("encoder", {}))
    chunker = make_chunker(variant, encoder)
    docs = synthetic.build(
        n_docs=suite.get("n_docs", 30),
        min_sources=suite.get("min_sources", 3),
        max_sources=suite.get("max_sources", 6),
        seed=suite.get("seed", 0),
    )
    if hasattr(encoder, "warm_up"):
        encoder.warm_up()
    encoder.reset_counters()
    pks, wds, ps, rs, f1s = [], [], [], [], []
    n_chunks = 0
    chunk_tokens: list[int] = []
    t0 = time.perf_counter()
    for doc in docs:
        result = chunker([doc.text])
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
        chunk_tokens.extend(c.token_count or 0 for c in chunks)
    wall = time.perf_counter() - t0
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    out = {
        "boundary_f1": mean(f1s),
        "boundary_precision": mean(ps),
        "boundary_recall": mean(rs),
        "pk": mean(pks),
        "windowdiff": mean(wds),
        "wall_s": round(wall, 3),
        "chunks_per_doc": round(n_chunks / len(docs), 3),
        "mean_chunk_tokens": mean([t for t in chunk_tokens if t]),
    }
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
    for suite in config["suites"]:
        runner = SUITES[suite["name"]]
        for variant in config["variants"]:
            print(f"{suite['name']} :: {variant['name']}", file=sys.stderr)
            m = runner(variant, suite)
            results.append(
                {
                    "name": suite["name"],
                    "variant": variant["name"],
                    "config": {
                        **{k: v for k, v in variant.items() if k != "name"},
                        "suite": {k: v for k, v in suite.items() if k != "name"},
                    },
                    "metrics": m,
                }
            )
            print(json.dumps(m), file=sys.stderr)

    payload = {
        "schema": 1,
        "commit": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "directions": {k: v for k, v in DIRECTIONS.items() if v},
        "suites": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

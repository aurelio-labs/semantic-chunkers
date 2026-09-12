"""Synthetic boundary suite.

Concatenate the introductions of unrelated Wikipedia articles and record
the seams. The chunker sees the concatenated text; the gold boundaries are
the sentence indices where a new article begins. Cheap, deterministic, and
free of licence questions beyond attribution.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "wiki_intros.json"


@dataclass
class SyntheticDoc:
    text: str
    sentences: list[str]
    boundaries: list[int]  # sentence indices where a new source document starts
    sources: list[str]


def load_corpus() -> list[dict]:
    return json.loads(DATA.read_text())


def build(
    n_docs: int = 30,
    min_sources: int = 3,
    max_sources: int = 6,
    seed: int = 0,
    split=None,
) -> list[SyntheticDoc]:
    """Build synthetic documents.

    ``split`` is a callable ``str -> list[str]`` used to sentence-split each
    source before concatenation, so gold boundaries are expressed in the same
    sentence units the chunker under test will see. Defaults to
    ``RegexSplitter``.
    """
    if split is None:
        from semantic_chunkers.splitters import RegexSplitter

        split = RegexSplitter()
    corpus = load_corpus()
    rng = random.Random(seed)
    docs: list[SyntheticDoc] = []
    for _ in range(n_docs):
        k = rng.randint(min_sources, max_sources)
        picks = rng.sample(corpus, k)
        sentences: list[str] = []
        boundaries: list[int] = []
        sources: list[str] = []
        for entry in picks:
            sents = [s for s in split(entry["text"]) if s.strip()]
            if not sents:
                continue
            if sentences:
                boundaries.append(len(sentences))
            sentences.extend(sents)
            sources.append(entry["title"])
        docs.append(
            SyntheticDoc(
                text=" ".join(sentences),
                sentences=sentences,
                boundaries=boundaries,
                sources=sources,
            )
        )
    return docs

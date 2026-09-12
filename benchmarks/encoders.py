"""Encoders for the benchmark, with an on-disk embedding cache and a call
counter so cost can be reported alongside quality.

The cache is keyed on (model name, namespace, text). The runner namespaces
by variant, so within one run every variant pays its own embedding cost and
the order of variants cannot change a number; across runs, an unchanged
variant is served from the cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from semantic_router.encoders.base import DenseEncoder

CACHE_DIR = Path(__file__).parent / ".cache"


class CachedSentenceTransformerEncoder(DenseEncoder):
    """A ``DenseEncoder`` backed by sentence-transformers with a SQLite cache.

    Counts every call that reaches the model (``model_calls``) and every text
    embedded by the model (``model_texts``); cache hits are free.
    """

    name: str = "all-MiniLM-L6-v2"
    score_threshold: float = 0.5
    type: str = "sentence-transformers"
    namespace: str = ""

    _model: Any = None
    _conn: Optional[sqlite3.Connection] = None
    model_calls: int = 0
    model_texts: int = 0
    model_seconds: float = 0.0
    requests: int = 0
    requested_texts: int = 0

    def __init__(
        self,
        name: str = "all-MiniLM-L6-v2",
        cache_dir: Path = CACHE_DIR,
        namespace: str = "",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.namespace = namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(cache_dir / "embeddings.sqlite")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)"
        )

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device="cpu")
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.name}\x00{self.namespace}\x00{text}".encode()
        ).hexdigest()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        assert self._conn is not None
        self.requests += 1
        self.requested_texts += len(docs)
        keys = [self._key(d) for d in docs]
        out: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for i, key in enumerate(keys):
            row = self._conn.execute(
                "SELECT vec FROM emb WHERE key=?", (key,)
            ).fetchone()
            if row:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                missing.append(i)
        if missing:
            model = self._load()
            t0 = time.perf_counter()
            vecs = model.encode(
                [docs[i] for i in missing],
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False,
            )
            self.model_seconds += time.perf_counter() - t0
            self.model_calls += 1
            self.model_texts += len(missing)
            for i, vec in zip(missing, vecs):
                vec32 = np.asarray(vec, dtype=np.float32)
                out[i] = vec32
                self._conn.execute(
                    "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                    (keys[i], vec32.tobytes()),
                )
            self._conn.commit()
        return [out[i].tolist() for i in range(len(docs))]

    async def acall(self, docs: List[str]) -> List[List[float]]:
        return self(docs)

    def warm_up(self) -> None:
        """Load the model so its start-up cost is not charged to the first variant."""
        self._load()

    def reset_counters(self) -> None:
        self.model_calls = 0
        self.model_texts = 0
        self.model_seconds = 0.0
        self.requests = 0
        self.requested_texts = 0

    def counters(self) -> dict[str, float]:
        """Cost counters.

        ``encoder_requests`` and ``encoder_texts_requested`` count every call the
        chunker made, cache hit or miss, and are the cost a user would pay.
        ``encoder_model_calls`` and ``encoder_model_texts`` count cache misses
        only, so they show how much the cache saved on this run.
        """
        return {
            "encoder_requests": self.requests,
            "encoder_texts_requested": self.requested_texts,
            "encoder_model_calls": self.model_calls,
            "encoder_model_texts": self.model_texts,
            "encoder_seconds": round(self.model_seconds, 3),
        }


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

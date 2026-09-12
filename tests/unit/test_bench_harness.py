import json

import pytest

from benchmarks import run, synthetic
from benchmarks.run import predicted_boundaries
from semantic_chunkers.schema import Chunk


def test_synthetic_build_is_deterministic_and_records_seams():
    a = synthetic.build(n_docs=5, seed=3)
    b = synthetic.build(n_docs=5, seed=3)
    assert [d.text for d in a] == [d.text for d in b]
    for doc in a:
        assert doc.text == " ".join(doc.sentences)
        assert len(doc.boundaries) == len(doc.sources) - 1
        assert all(0 < i < len(doc.sentences) for i in doc.boundaries)
        assert doc.boundaries == sorted(set(doc.boundaries))


def test_synthetic_build_respects_source_range():
    docs = synthetic.build(n_docs=10, min_sources=2, max_sources=2, seed=1)
    assert all(len(d.sources) == 2 for d in docs)
    assert all(len(d.boundaries) == 1 for d in docs)


def test_predicted_boundaries_maps_chunk_starts_to_sentence_indices():
    sentences = ["a.", "b.", "c.", "d.", "e."]
    chunks = [
        Chunk(splits=["a.", "b."]),
        Chunk(splits=["c."]),
        Chunk(splits=["d.", "e."]),
    ]
    assert predicted_boundaries(chunks, sentences) == [2, 3]


def test_predicted_boundaries_ignores_whitespace_and_empty_chunks():
    sentences = ["a.", " b. ", "c."]
    chunks = [Chunk(splits=[]), Chunk(splits=["a."]), Chunk(splits=["b.", "c."])]
    assert predicted_boundaries(chunks, sentences) == [1]


def test_predicted_boundaries_raises_when_a_chunk_cannot_be_placed():
    sentences = ["a.", "b.", "c."]
    chunks = [Chunk(splits=["a."]), Chunk(splits=["zzz."])]
    with pytest.raises(ValueError, match="matches no remaining sentence"):
        predicted_boundaries(chunks, sentences)


def test_main_records_a_failed_variant_and_still_writes_the_table(tmp_path):
    """A variant that raises must not cost the whole table.

    Both variants use the regex chunker, which needs no encoder, so this runs
    with no model. The first is given a parameter the chunker rejects.
    """
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "suites": [{"name": "synthetic-boundaries", "n_docs": 2, "seed": 0}],
                "variants": [
                    {
                        "name": "broken",
                        "chunker": "regex",
                        "params": {"no_such_param": 1},
                    },
                    {
                        "name": "ok",
                        "chunker": "regex",
                        "params": {"max_chunk_tokens": 300},
                    },
                ],
            }
        )
    )
    out = tmp_path / "results.json"
    assert run.main([str(config), "--out", str(out)]) == 1

    payload = json.loads(out.read_text())
    broken, ok = payload["suites"]
    assert broken["variant"] == "broken"
    assert broken["error"]
    assert broken["metrics"] == {}
    assert ok["variant"] == "ok"
    assert "error" not in ok
    assert ok["metrics"]["boundary_f1"] >= 0

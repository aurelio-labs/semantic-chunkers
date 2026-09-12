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


def _config(tmp_path, variants):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "suites": [{"name": "synthetic-boundaries", "n_docs": 2, "seed": 0}],
                "variants": variants,
            }
        )
    )
    return config


_BROKEN = {"name": "broken", "chunker": "regex", "params": {"no_such_param": 1}}
_OK = {"name": "ok", "chunker": "regex", "params": {"max_chunk_tokens": 300}}


def test_main_records_a_failed_variant_and_still_writes_the_table(tmp_path):
    """A variant that raises must not cost the whole table.

    Both variants use the regex chunker, which needs no encoder, so this runs
    with no model. The first is given a parameter the chunker rejects.

    The exit code is 0: the bench action runs the command under ``set -e``, so
    failing here would kill the step before the delta comment is rendered and
    the ``failed:`` row would never reach the pull request.
    """
    out = tmp_path / "results.json"
    assert run.main([str(_config(tmp_path, [_BROKEN, _OK])), "--out", str(out)]) == 0

    payload = json.loads(out.read_text())
    broken, ok = payload["suites"]
    assert broken["variant"] == "broken"
    assert broken["error"]
    assert broken["metrics"] == {}
    assert ok["variant"] == "ok"
    assert "error" not in ok
    assert ok["metrics"]["boundary_f1"] >= 0


def test_main_exits_nonzero_when_every_variant_failed(tmp_path):
    """Nothing to report is a command failure, and CI should see it."""
    out = tmp_path / "results.json"
    assert run.main([str(_config(tmp_path, [_BROKEN])), "--out", str(out)]) == 1
    payload = json.loads(out.read_text())
    assert [s["variant"] for s in payload["suites"]] == ["broken"]
    assert payload["suites"][0]["error"]


def test_make_chunker_refuses_to_build_an_embedding_chunker_without_an_encoder():
    """BaseChunker would otherwise substitute a DenseEncoder that cannot encode."""
    with pytest.raises(ValueError, match="needs an encoder"):
        run.make_chunker({"chunker": "statistical"}, None)


def test_metrics_carry_a_distribution_not_just_a_mean(tmp_path):
    """Regex needs no encoder, so the whole runner is exercised with no model."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "suites": [{"name": "synthetic-boundaries", "n_docs": 6, "seed": 0}],
                "variants": [
                    {
                        "name": "ok",
                        "chunker": "regex",
                        "params": {"max_chunk_tokens": 300},
                    }
                ],
            }
        )
    )
    out = tmp_path / "results.json"
    assert run.main([str(config), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    m = payload["suites"][0]["metrics"]

    assert m["boundary_f1_p05"] <= m["boundary_f1_p50"] <= m["boundary_f1_p95"]
    assert m["pk_p50"] <= m["pk_p95"]
    assert m["windowdiff_p50"] <= m["windowdiff_p95"]
    assert 0 < m["doc_s_p50"] <= m["doc_s_p95"] <= m["wall_s"]
    # every reported percentile declares which way is better
    for key in m:
        if key.endswith(("_p05", "_p50", "_p95")):
            assert payload["directions"][key] in ("up", "down")

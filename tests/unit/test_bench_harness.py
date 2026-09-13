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


def test_every_document_is_embedded_in_its_own_cache_namespace(
    monkeypatch, fake_encoder
):
    """Per-document timing is only honest if each document pays for itself.

    The synthetic suite draws documents from one pool of articles, so without a
    partition per document the later ones are served from the cache and
    ``doc_s_p50``/``doc_s_p95`` rank documents by position in the suite rather
    than by difficulty. A stub model keeps this off the network.
    """
    encoder = fake_encoder("consecutive/stub")
    monkeypatch.setattr(run, "make_encoder", lambda spec, namespace: encoder)
    variant = {
        "name": "consecutive/stub",
        "chunker": "consecutive",
        "params": {"score_threshold": 0.45},
    }
    metrics = run.run_synthetic(variant, {"name": "synthetic-boundaries", "n_docs": 3})

    model = encoder._model
    # one partition per document, entered in order and never returned to, each
    # qualified by the suite so nothing is billed to another suite's run
    namespaces = list(dict.fromkeys(model.namespaces))
    assert [n.rsplit("/", 1)[1] for n in namespaces] == ["0", "1", "2"]
    assert all(n.startswith("synthetic-boundaries@") for n in namespaces)
    assert all(n.endswith(f"/consecutive/stub/{i}") for i, n in enumerate(namespaces))
    # the consequence: text shared between documents is embedded once per
    # document, so more texts reach the model than the run has distinct texts
    assert metrics["encoder_model_texts"] == len(model.texts) > len(set(model.texts))
    assert metrics["doc_s_p50"] <= metrics["doc_s_p95"]


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


def test_a_second_suite_does_not_reuse_the_first_suite_cache(monkeypatch, fake_encoder):
    """``main`` loops suites x variants, and two suites can share text.

    Without the suite in the partition, suite B's document *i* would inherit
    whatever suite A's document *i* embedded, and B's per-document timing and
    encoder counts would be charged to A.
    """
    encoder = fake_encoder("consecutive/stub")
    monkeypatch.setattr(run, "make_encoder", lambda spec, namespace: encoder)
    variant = {
        "name": "consecutive/stub",
        "chunker": "consecutive",
        "params": {"score_threshold": 0.45},
    }
    run.run_synthetic(variant, {"name": "suite-a", "n_docs": 2, "seed": 0})
    run.run_synthetic(variant, {"name": "suite-b", "n_docs": 2, "seed": 0})

    namespaces = list(dict.fromkeys(encoder._model.namespaces))
    assert len(namespaces) == 4, "two suites, two documents each, no partition shared"
    assert sum(n.startswith("suite-a@") for n in namespaces) == 2
    assert sum(n.startswith("suite-b@") for n in namespaces) == 2


def test_the_same_suite_twice_with_different_parameters_does_not_share_a_cache(
    monkeypatch, fake_encoder
):
    """A seed sweep lists one suite repeatedly; those runs are as distinct as two suites."""
    encoder = fake_encoder("consecutive/stub")
    monkeypatch.setattr(run, "make_encoder", lambda spec, namespace: encoder)
    variant = {
        "name": "consecutive/stub",
        "chunker": "consecutive",
        "params": {"score_threshold": 0.45},
    }
    name = "synthetic-boundaries"
    run.run_synthetic(variant, {"name": name, "n_docs": 2, "seed": 0})
    run.run_synthetic(variant, {"name": name, "n_docs": 2, "seed": 1})

    namespaces = set(encoder._model.namespaces)
    assert len(namespaces) == 4, "a differing seed must not reuse the earlier partition"


def test_a_scoring_only_knob_does_not_discard_the_embedding_cache(
    monkeypatch, fake_encoder
):
    """``tolerance`` reaches the scorer only, so changing it must not re-embed."""
    encoder = fake_encoder("consecutive/stub")
    monkeypatch.setattr(run, "make_encoder", lambda spec, namespace: encoder)
    variant = {
        "name": "consecutive/stub",
        "chunker": "consecutive",
        "params": {"score_threshold": 0.45},
    }
    base = {"name": "synthetic-boundaries", "n_docs": 2, "seed": 0}
    run.run_synthetic(variant, {**base, "tolerance": 1})
    first = set(encoder._model.namespaces)
    run.run_synthetic(variant, {**base, "tolerance": 2})

    assert set(encoder._model.namespaces) == first, (
        "tolerance shapes no document, so it must not open new cache partitions"
    )

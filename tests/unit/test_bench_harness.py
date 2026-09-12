import pytest

from benchmarks import synthetic
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

"""The embedding cache: what it saves, what it charges, and how it partitions.

The runner's per-document timing rests on ``use_namespace`` really moving the
cache partition, so that is asserted here rather than assumed.
"""


def test_the_cache_serves_a_repeated_text_without_the_model(fake_encoder):
    encoder = fake_encoder("variant")

    first = encoder(["a sentence.", "another."])
    second = encoder(["a sentence.", "another."])

    assert second == first
    assert encoder._model.texts == ["a sentence.", "another."]
    # both calls are billed to the chunker; only the miss reached the model
    counters = encoder.counters()
    assert counters["encoder_requests"] == 2
    assert counters["encoder_texts_requested"] == 4
    assert counters["encoder_model_calls"] == 1
    assert counters["encoder_model_texts"] == 2


def test_use_namespace_makes_the_same_text_a_miss_again(fake_encoder):
    """A new namespace is a fresh partition: nothing before it is reachable."""
    encoder = fake_encoder("variant/0")
    encoder(["shared sentence."])

    encoder.use_namespace("variant/1")
    encoder(["shared sentence."])

    assert encoder._model.calls == [
        ("variant/0", ["shared sentence."]),
        ("variant/1", ["shared sentence."]),
    ]
    assert encoder.model_texts == 2


def test_a_namespace_still_caches_within_itself(fake_encoder):
    """Partitioning must not cost the cache: a repeat inside one is still free."""
    encoder = fake_encoder("variant/0")
    encoder(["shared sentence."])
    encoder.use_namespace("variant/1")
    encoder(["shared sentence."])
    encoder(["shared sentence."])

    assert encoder.model_texts == 2
    assert encoder.requested_texts == 3


def test_counters_reset_without_clearing_the_cache(fake_encoder):
    encoder = fake_encoder("variant")
    encoder(["a sentence."])
    encoder.reset_counters()
    encoder(["a sentence."])

    assert encoder.counters()["encoder_requests"] == 1
    assert encoder.counters()["encoder_model_texts"] == 0

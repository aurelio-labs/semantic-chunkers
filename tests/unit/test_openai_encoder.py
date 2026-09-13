"""The OpenAI encoder, driven through ``httpx.MockTransport``.

No key, no network, no ``openai`` SDK. Every test builds the encoder around a
handler that returns whatever the case needs, so the request shape, the
batching, the retries and the error mapping are all asserted against what
actually went over the wire.
"""

import asyncio
import json

import httpx
import pytest

from semantic_chunkers import OpenAIEncoder
from semantic_chunkers.encoders import openai as openai_encoder
from semantic_chunkers.encoders.openai import OpenAIEncoderError


def embeddings_response(request: httpx.Request) -> httpx.Response:
    """One vector per input, so a caller can count what it got back."""
    inputs = json.loads(request.content)["input"]
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": i, "embedding": [float(i), 0.5]} for i in range(len(inputs))
            ]
        },
    )


def encoder_with(handler, **kwargs) -> OpenAIEncoder:
    return OpenAIEncoder(
        api_key="sk-test", transport=httpx.MockTransport(handler), **kwargs
    )


def recording(handler):
    """Wrap a handler so the test can read the requests it received."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return record, requests


def test_the_request_carries_the_key_the_model_and_the_documents():
    handler, requests = recording(embeddings_response)
    encoder = encoder_with(handler)

    encoder(["one", "two"])

    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "input": ["one", "two"],
        "model": "text-embedding-3-small",
    }


def test_dimensions_are_sent_only_when_asked_for():
    handler, requests = recording(embeddings_response)

    encoder_with(handler)(["one"])
    encoder_with(handler, dimensions=256)(["one"])

    assert "dimensions" not in json.loads(requests[0].content)
    assert json.loads(requests[1].content)["dimensions"] == 256


def test_the_organisation_header_is_sent_when_given():
    handler, requests = recording(embeddings_response)

    encoder_with(handler, org_id="org-42")(["one"])

    assert requests[0].headers["OpenAI-Organization"] == "org-42"


def test_the_key_and_base_url_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://azure.example.com/v1/")
    handler, requests = recording(embeddings_response)

    OpenAIEncoder(transport=httpx.MockTransport(handler))(["one"])

    assert str(requests[0].url) == "https://azure.example.com/v1/embeddings"
    assert requests[0].headers["Authorization"] == "Bearer sk-from-env"


def test_a_missing_key_is_caught_at_construction(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIEncoderError, match="OPENAI_API_KEY"):
        OpenAIEncoder()


def test_documents_over_the_batch_size_are_split_across_requests():
    handler, requests = recording(embeddings_response)
    encoder = encoder_with(handler, batch_size=2)

    vectors = encoder(["a", "b", "c", "d", "e"])

    assert [json.loads(r.content)["input"] for r in requests] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert len(vectors) == 5


def test_vectors_come_back_in_the_order_the_documents_went_in():
    """The endpoint indexes its embeddings and need not return them in order."""

    def shuffled(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "embedding": [2.0]},
                    {"index": 0, "embedding": [0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            },
        )

    assert encoder_with(shuffled)(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]


def test_a_rate_limited_request_is_retried_until_it_succeeds():
    attempts = []

    def rate_limited_once(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return embeddings_response(request)

    assert encoder_with(rate_limited_once)(["a"]) == [[0.0, 0.5]]
    assert len(attempts) == 2


def test_a_server_fault_is_retried_on_the_async_path():
    attempts = []

    def broken_once(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, json={})
        return embeddings_response(request)

    encoder = encoder_with(broken_once)

    assert asyncio.run(encoder.acall(["a"])) == [[0.0, 0.5]]
    assert len(attempts) == 2


def test_retries_are_bounded_and_the_last_failure_is_raised():
    attempts = []

    def always_rate_limited(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={"error": {"message": "Rate limit reached"}},
        )

    with pytest.raises(OpenAIEncoderError, match="Rate limit reached"):
        encoder_with(always_rate_limited, max_retries=2)(["a"])

    assert len(attempts) == 3  # the first try plus two retries


def test_an_authentication_failure_raises_the_api_message_without_retrying():
    attempts = []

    def unauthorised(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            401, json={"error": {"message": "Incorrect API key provided"}}
        )

    with pytest.raises(OpenAIEncoderError) as error:
        encoder_with(unauthorised)(["a"])

    assert "401" in str(error.value)
    assert "Incorrect API key provided" in str(error.value)
    assert len(attempts) == 1


def test_an_error_without_a_json_body_still_names_the_status():
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    with pytest.raises(OpenAIEncoderError, match="400"):
        encoder_with(broken)(["a"])


def test_a_success_with_no_embeddings_is_an_error_not_a_crash():
    """An OpenAI-compatible endpoint that answers with something else."""

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list"})

    with pytest.raises(OpenAIEncoderError, match="no embeddings"):
        encoder_with(empty)(["a"])


def test_the_async_path_returns_what_the_sync_path_returns():
    encoder = encoder_with(embeddings_response, batch_size=2)

    assert asyncio.run(encoder.acall(["a", "b", "c"])) == encoder(["a", "b", "c"])


def test_retry_after_decides_the_wait_when_the_server_sends_one():
    """Asserted on the backoff itself: observing it through a call means sleeping."""
    response = httpx.Response(429, headers={"Retry-After": "2"})

    assert openai_encoder._backoff(response, attempt=0) == 2.0


def test_the_wait_backs_off_exponentially_without_a_retry_after():
    response = httpx.Response(429)

    waits = [openai_encoder._backoff(response, attempt) for attempt in range(3)]

    assert waits == [0.5, 1.0, 2.0]


def test_an_http_date_in_retry_after_falls_back_to_the_backoff():
    response = httpx.Response(
        429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
    )

    assert openai_encoder._backoff(response, attempt=0) == 0.5


def test_the_wait_is_capped_however_long_the_server_asks_for():
    response = httpx.Response(429, headers={"Retry-After": "600"})

    assert openai_encoder._backoff(response, attempt=0) == (
        openai_encoder.MAX_BACKOFF_SECONDS
    )


@pytest.mark.live
def test_live_the_real_endpoint_embeds_a_document():
    """Run by hand with `uv run pytest -m live`; needs OPENAI_API_KEY."""
    encoder = OpenAIEncoder()

    vectors = encoder(["semantic chunkers splits text into coherent chunks."])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1536

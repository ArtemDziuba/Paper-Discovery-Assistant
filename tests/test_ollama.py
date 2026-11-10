"""Local model discovery, generation options and failures use mocked HTTP only."""

import json

import httpx
import pytest

from pda.ollama import OllamaClient
from pda.summary_config import SummaryError, SummarySettings, validate_summary


def response(content="scientific findings", **updates):
    return {
        "message": {"content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 50,
        **updates,
    }


def test_installed_model_digest_and_configurable_inference_options():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3.5:4b", "digest": "weights-v1"}]}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-server"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"qwen.context_length": 262144}})
        return httpx.Response(200, json=response())

    client = OllamaClient(SummarySettings(), httpx.MockTransport(handler))
    assert client.identity().digest == "weights-v1"
    assert client.chat([{"role": "user", "content": "test"}]) == "scientific findings"
    payload = json.loads(requests[-1].content)
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"]["temperature"] == 0.2 and payload["options"]["num_ctx"] == 8192
    assert payload["keep_alive"] == "10m"
    assert all("pull" not in str(r.url) for r in requests)


def test_missing_model_does_not_download_it():
    client = OllamaClient(
        SummarySettings(), httpx.MockTransport(lambda _: httpx.Response(200, json={"models": []}))
    )
    with pytest.raises(SummaryError, match="ollama pull qwen3.5:4b"):
        client.identity()


@pytest.mark.parametrize("problem", ["malformed_list", "missing_digest", "remote", "too_large"])
def test_model_identity_must_be_local_versioned_and_support_configured_context(problem):
    def handler(request):
        if request.url.path == "/api/tags":
            if problem == "malformed_list":
                return httpx.Response(200, json={"models": [None]})
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3.5:4b",
                            "digest": "" if problem == "missing_digest" else "local-weights",
                        }
                    ]
                },
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test-server"})
        return httpx.Response(
            200,
            json={
                "remote_host": "remote.example" if problem == "remote" else "",
                "model_info": {"qwen.context_length": 4096 if problem == "too_large" else 262144},
            },
        )

    client = OllamaClient(SummarySettings(), httpx.MockTransport(handler))
    with pytest.raises(SummaryError):
        client.identity()


@pytest.mark.parametrize(
    "text", ["", "# Summary\n", "<think>reasoning</think> ", "https://example.com ", "[1] "]
)
def test_invalid_summary_format_is_rejected(text):
    with pytest.raises(SummaryError):
        validate_summary(text + ("supported finding " * 55 if text else ""))


@pytest.mark.parametrize(
    "problem", ["unavailable", "timeout", "empty", "incomplete", "bad_context", "invalid_json"]
)
def test_inference_failures_are_understandable(problem):
    def handler(request):
        if problem == "unavailable":
            raise httpx.ConnectError("offline")
        if problem == "timeout":
            raise httpx.ReadTimeout("slow")
        if problem == "invalid_json":
            return httpx.Response(200, text="not json")
        body = response()
        if problem == "empty":
            body["message"]["content"] = ""
        if problem == "incomplete":
            body["done_reason"] = "length"
        if problem == "bad_context":
            body["prompt_eval_count"] = 8192
        return httpx.Response(200, json=body)

    client = OllamaClient(SummarySettings(), httpx.MockTransport(handler))
    with pytest.raises(SummaryError):
        client.chat([{"role": "user", "content": "test"}])


def test_unsupported_thinking_flag_is_omitted_on_retry():
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        return (
            httpx.Response(400, text="think is not supported")
            if "think" in payload
            else httpx.Response(200, json=response())
        )

    client = OllamaClient(SummarySettings(), httpx.MockTransport(handler))
    assert client.chat([{"role": "user", "content": "test"}])
    assert len(payloads) == 2 and "think" not in payloads[1]


def test_oversized_prompt_is_rejected_before_contacting_ollama():
    client = OllamaClient(
        SummarySettings(),
        httpx.MockTransport(lambda _: pytest.fail("No oversized request should be sent")),
    )
    with pytest.raises(SummaryError, match="no paper text was truncated"):
        client.chat([{"role": "user", "content": "a" * 10000}])


@pytest.mark.parametrize(
    "endpoint",
    ["https://example.com", "http://192.168.1.1:11434", "http://user:secret@localhost:11434"],
)
def test_inference_endpoint_is_local_only(endpoint):
    with pytest.raises(ValueError, match="local HTTP"):
        SummarySettings(endpoint=endpoint)


@pytest.mark.parametrize("count", [49, 50, 51, 99, 100, 150, 151, 153, 179, 180, 181])
def test_summary_word_boundaries(count):
    text = " ".join(["evidence"] * count)
    if 50 <= count <= 180:
        assert validate_summary(text) == text
    else:
        with pytest.raises(SummaryError, match="words"):
            validate_summary(text)

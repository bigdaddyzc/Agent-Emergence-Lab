"""Tests for OllamaClient.generate_topic random-seed topic generation."""

import pytest
import requests

from src.ollama_client import OllamaClient, OllamaError, OllamaRequestError, OllamaResponse


def _make_client():
    # __init__ only sets attributes / creates a Session — no network is touched.
    return OllamaClient(base_url="http://localhost:11434")


def _fake_response(text: str) -> OllamaResponse:
    return OllamaResponse(
        text=text, tokens_generated=10, tokens_prompt=20,
        total_duration_ms=100, tokens_per_second=10.0, model="test-model",
    )


class TestGenerateTopic:
    def test_injects_random_domain_and_angle_seed(self):
        client = _make_client()
        captured = {}

        def fake_generate(model, prompt, **kwargs):
            captured["prompt"] = prompt
            return _fake_response("一个具体而新颖的话题")

        client.generate = fake_generate
        topic = client.generate_topic(model="test-model")

        assert topic == "一个具体而新颖的话题"
        # The prompt must carry one domain seed and one angle seed.
        assert any(d in captured["prompt"] for d in client._TOPIC_DOMAINS)
        assert any(a in captured["prompt"] for a in client._TOPIC_ANGLES)

    def test_keeps_only_first_line(self):
        client = _make_client()
        client.generate = lambda model, prompt, **kw: _fake_response(
            "真正的话题\n这是模型多嘴的解释")
        assert client.generate_topic(model="test-model") == "真正的话题"

    def test_strips_quotes(self):
        client = _make_client()
        client.generate = lambda model, prompt, **kw: _fake_response('"带引号的话题"')
        assert client.generate_topic(model="test-model") == "带引号的话题"

    def test_fallback_pool_on_failure(self):
        client = _make_client()

        def boom(model, prompt, **kwargs):
            raise RuntimeError("ollama down")

        client.generate = boom
        topic = client.generate_topic(model="test-model")
        # Fallback must come from the small pool, never empty.
        assert topic in client._TOPIC_FALLBACKS

    def test_fallback_on_empty_response(self):
        client = _make_client()
        client.generate = lambda model, prompt, **kw: _fake_response("   ")
        topic = client.generate_topic(model="test-model")
        assert topic in client._TOPIC_FALLBACKS

    def test_not_from_removed_preset_list(self):
        """Topic generation must not silently echo a fixed single value."""
        client = _make_client()
        outputs = []

        def varied(model, prompt, **kwargs):
            # Echo the domain seed so different calls yield different topics.
            for d in client._TOPIC_DOMAINS:
                if d in prompt:
                    return _fake_response(f"关于{d}的一个话题")
            return _fake_response("兜底")

        client.generate = varied
        for _ in range(20):
            outputs.append(client.generate_topic(model="test-model"))
        # With 35 domains x random selection, 20 draws should not collapse to one.
        assert len(set(outputs)) > 1


# ── HTTP robustness (no network: requests.Session is stubbed) ────────────


class _Resp:
    def __init__(self, status=200, payload=None, text="", bad_json=False):
        self.status_code = status
        self._payload = payload
        self.text = text
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value")
        return self._payload


class _Session:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0

    def request(self, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.responses.pop(0)

    def head(self, url, timeout):
        if self.exc:
            raise self.exc
        return _Resp(200)


def _client_with(session):
    client = _make_client()
    client.session = session
    return client


def test_chat_invalid_json_raises_ollama_error():
    client = _client_with(_Session([_Resp(bad_json=True, text="<html>")]))
    with pytest.raises(OllamaError, match="Invalid JSON"):
        client.chat("m", [{"role": "user", "content": "hi"}])


def test_chat_error_field_in_200_body_raises():
    client = _client_with(_Session([_Resp(payload={"error": "model failed to load"})]))
    with pytest.raises(OllamaRequestError, match="model failed to load"):
        client.chat("m", [])


def test_chat_missing_message_returns_empty_text():
    client = _client_with(_Session([_Resp(payload={"message": None, "eval_count": 3})]))
    resp = client.chat("m", [])
    assert resp.text == ""
    assert resp.tokens_generated == 3


def test_list_models_skips_malformed_entries():
    client = _client_with(_Session([_Resp(payload={"models": [{"name": "a"}, {"size": 1}, "x"]})]))
    assert client.list_models() == ["a"]


def test_is_running_false_on_timeout():
    client = _client_with(_Session(exc=requests.Timeout("slow")))
    assert client.is_running() is False


def test_generic_request_exception_is_wrapped_after_retries(monkeypatch):
    monkeypatch.setattr("src.ollama_client.time.sleep", lambda s: None)
    session = _Session(exc=requests.exceptions.ChunkedEncodingError("dropped"))
    client = _client_with(session)
    with pytest.raises(OllamaError, match="dropped"):
        client.chat("m", [])
    assert session.calls == 3

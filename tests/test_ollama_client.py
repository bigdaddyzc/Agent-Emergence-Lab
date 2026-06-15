"""Tests for OllamaClient.generate_topic random-seed topic generation."""

from src.ollama_client import OllamaClient, OllamaResponse


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

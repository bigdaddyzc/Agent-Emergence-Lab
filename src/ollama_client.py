"""Ollama HTTP API wrapper."""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama errors."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama server is unreachable."""


class OllamaRequestError(OllamaError):
    """Raised on HTTP 4xx errors."""


class OllamaServerError(OllamaError):
    """Raised on HTTP 5xx errors."""


class OllamaTimeoutError(OllamaError):
    """Raised on request timeout."""


@dataclass
class OllamaResponse:
    text: str
    tokens_generated: int
    tokens_prompt: int
    total_duration_ms: int
    tokens_per_second: float
    model: str

    @classmethod
    def from_api(cls, data: dict, model: str) -> "OllamaResponse":
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)  # nanoseconds
        prompt_eval_count = data.get("prompt_eval_count", 0)
        total_duration = data.get("total_duration", 0)  # nanoseconds

        duration_s = eval_duration / 1e9 if eval_duration else 0
        tok_per_sec = eval_count / duration_s if duration_s > 0 else 0

        return cls(
            text=data.get("response", ""),
            tokens_generated=eval_count,
            tokens_prompt=prompt_eval_count,
            total_duration_ms=int(total_duration / 1_000_000),
            tokens_per_second=round(tok_per_sec, 2),
            model=model,
        )


class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 default_timeout: int = 300,
                 keep_alive: str = "5m"):
        self.base_url = base_url.rstrip("/")
        self.default_timeout = default_timeout
        self.keep_alive = keep_alive
        self.session = requests.Session()

    # ---- Health / Info ----

    def is_running(self) -> bool:
        try:
            resp = self.session.head(self.base_url, timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        resp = self._request("GET", "/api/tags")
        models = resp.json().get("models", [])
        return [m["name"] for m in models]

    def pull_model(self, model: str) -> None:
        logger.info("Pulling model %s ...", model)
        self._request("POST", "/api/pull", json_data={"model": model})

    # ---- Generation ----

    def generate(self, model: str, prompt: str, *,
                 system: str = "",
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 context_window: int = 4096,
                 repeat_penalty: float = 1.1,
                 top_p: float = 0.9,
                 top_k: int = 40) -> OllamaResponse:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": context_window,
                "repeat_penalty": repeat_penalty,
                "top_p": top_p,
                "top_k": top_k,
            },
        }
        resp = self._request("POST", "/api/generate", json_data=payload)
        data = resp.json()
        return OllamaResponse.from_api(data, model)

    def chat(self, model: str, messages: list[dict], *,
             temperature: float = 0.7,
             max_tokens: int = 512,
             context_window: int = 4096,
             repeat_penalty: float = 1.1,
             top_p: float = 0.9,
             top_k: int = 40) -> OllamaResponse:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": context_window,
                "repeat_penalty": repeat_penalty,
                "top_p": top_p,
                "top_k": top_k,
            },
        }
        resp = self._request("POST", "/api/chat", json_data=payload)
        data = resp.json()
        # Extract the assistant's reply
        msg = data.get("message", {})
        reply_text = msg.get("content", "")
        # Wrap into the same shape as generate response
        wrapped = {
            "response": reply_text,
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "total_duration": data.get("total_duration", 0),
        }
        return OllamaResponse.from_api(wrapped, model)

    def embed(self, model: str, input_text: str) -> list[float]:
        resp = self._request(
            "POST", "/api/embed",
            json_data={"model": model, "input": input_text},
        )
        data = resp.json()
        embeddings = data.get("embeddings", [])
        return embeddings[0] if embeddings else []

    # ---- Topic Generation ----

    def generate_topic(self, model: str = "") -> str:
        """Generate a random creative topic using Ollama."""
        if not model:
            models = self.list_models()
            # Pick the first available model that's not too large
            model = models[0] if models else "qwen2.5:3b"

        choices = [
            # Abstract / philosophical
            "Generate a single abstract or philosophical conversation topic for a deep discussion between two curious AI agents. The topic should explore concepts like meaning, truth, identity, time, or reality — anything goes. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # 'What if' speculative
            "Generate a single creative 'what if' speculative conversation topic for a deep discussion between two curious AI agents. Think alternate history, future scenarios, hypothetical science — anything imaginative. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Everyday / concrete
            "Generate a single creative, concrete conversation topic grounded in everyday life or human experience for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Science / nature
            "Generate a single conversation topic about science, nature, or the universe for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Art / culture
            "Generate a single conversation topic about art, culture, or creativity for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Society / psychology
            "Generate a single conversation topic about society, psychology, or human behavior for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Consciousness / AI / technology
            "Generate a single conversation topic about consciousness, AI, or technology for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Ethics / morality
            "Generate a single conversation topic about ethics, morality, or values for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Metaphysics / existence
            "Generate a single conversation topic about metaphysics, existence, or the nature of reality for a deep discussion between two curious AI agents. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
            # Completely random / surreal
            "Generate a single completely random, surreal, or unexpected conversation topic for a deep discussion between two curious AI agents. It can be funny, strange, or deeply insightful — be creative. Output ONLY the topic, no explanation, no quotes, no prefix. Generate a unique topic now:",
        ]
        prompt = random.choice(choices)
        try:
            resp = self.generate(
                model=model,
                prompt=prompt,
                temperature=0.95,
                max_tokens=64,
            )
            topic = resp.text.strip().strip('"').strip("'").strip()
            if topic:
                return topic
        except Exception:
            logger.warning("Topic generation failed, using fallback")
        return "如果人类突然获得心灵感应能力，社会会变成什么样"

    # ---- Internal ----

    def _request(self, method: str, path: str, *,
                 json_data: Optional[dict] = None,
                 max_retries: int = 3,
                 retry_delay: int = 5) -> requests.Response:
        url = f"{self.base_url}{path}"
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    timeout=self.default_timeout,
                )
                if resp.status_code == 404:
                    raise OllamaRequestError(
                        f"Model not found or resource missing (404): {resp.text}"
                    )
                if 400 <= resp.status_code < 500:
                    raise OllamaRequestError(
                        f"HTTP {resp.status_code}: {resp.text}"
                    )
                if resp.status_code >= 500:
                    raise OllamaServerError(
                        f"HTTP {resp.status_code}: {resp.text}"
                    )
                return resp

            except requests.ConnectionError as e:
                last_exc = OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is 'ollama serve' running? ({e})"
                )
            except requests.Timeout as e:
                last_exc = OllamaTimeoutError(
                    f"Request timed out after {self.default_timeout}s: {e}"
                )
            except (OllamaRequestError, OllamaServerError):
                raise

            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed, retrying in %ds ...",
                    attempt, max_retries, retry_delay,
                )
                time.sleep(retry_delay)

        raise last_exc or OllamaError("Request failed after retries")

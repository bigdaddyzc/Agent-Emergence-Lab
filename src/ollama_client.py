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

    # Random seeds for topic generation. Each call pairs one domain with one
    # angle, so the LLM gets a fresh, unpredictable starting point every time
    # instead of cycling through a handful of fixed prompt templates.
    _TOPIC_DOMAINS = [
        "科学", "自然", "宇宙", "技术", "人工智能", "历史", "未来",
        "艺术", "音乐", "文学", "哲学", "心理", "情绪", "记忆",
        "梦境", "语言", "数学", "生物", "进化", "城市", "建筑",
        "游戏", "食物", "睡眠", "时间", "金钱", "童年", "孤独",
        "习惯", "意识", "身份", "死亡", "运气", "日常生活", "人际关系",
    ]
    _TOPIC_ANGLES = [
        "如果……会发生什么",
        "为什么……",
        "一个被严重低估的……",
        "两个看似无关的事物之间隐藏的联系",
        "一个反直觉的现象",
        "一个我们习以为常、却从没认真想过的问题",
        "假如把它推到极端",
        "如果它突然消失",
        "它在一百年后会变成什么样",
        "它和另一个完全不同的领域的奇妙相似之处",
    ]

    _TOPIC_FALLBACKS = [
        "如果人类突然获得心灵感应能力，社会会变成什么样",
        "为什么我们会对小时候的某些小事记忆犹新",
        "如果时间可以倒流但只有一分钟，你会怎么用",
        "最被低估的日常发明是什么",
        "如果记忆可以像U盘一样拷贝和传输，会改变什么",
        "动物如果会说话，第一句可能说什么",
        "凌晨三点脑子里冒出来的那些问题都去哪了",
        "如果重力突然减半，生活会变成什么样",
        "梦境如果能被录下来回放，会发生什么",
        "为什么有些旋律听一次就忘不掉",
    ]

    def generate_topic(self, model: str = "") -> str:
        """Generate a random creative topic using Ollama.

        Each call combines a random domain with a random angle and lets the
        model invent a topic around that seed, maximizing variety. Falls back
        to a random pick from a small pool if the model call fails.
        """
        if not model:
            models = self.list_models()
            # Pick the first available model that's not too large
            model = models[0] if models else "qwen2.5:3b"

        domain = random.choice(self._TOPIC_DOMAINS)
        angle = random.choice(self._TOPIC_ANGLES)
        prompt = (
            "你是一个擅长提出有趣话题的人。请围绕下面的线索，生成一个适合两个好奇的AI"
            "深入讨论的对话话题。\n"
            f"领域线索：{domain}\n"
            f"切入角度：{angle}\n\n"
            "要求：\n"
            "- 只输出话题本身，一句话，不要解释、不要引号、不要任何前缀\n"
            "- 话题要具体、新颖、能引发联想，不要空泛的大词\n"
            "- 用中文\n\n"
            "现在生成这个话题："
        )
        try:
            resp = self.generate(
                model=model,
                prompt=prompt,
                temperature=0.95,
                max_tokens=64,
            )
            topic = resp.text.strip().strip('"').strip("'").strip()
            # Keep only the first line in case the model adds commentary
            topic = topic.split("\n")[0].strip()
            if topic:
                return topic
        except Exception:
            logger.warning("Topic generation failed, using fallback")
        return random.choice(self._TOPIC_FALLBACKS)

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

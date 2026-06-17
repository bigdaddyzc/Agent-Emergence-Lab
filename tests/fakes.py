"""Test doubles used across the suite."""

from src.ollama_client import OllamaResponse


class FakeOllamaClient:
    def __init__(self):
        self.chat_calls = 0
        self.generate_calls = 0

    def chat(self, model, messages, **kwargs):
        self.chat_calls += 1
        text = (
            f"【新概念：测试概念{self.chat_calls}】"
            f"步骤1：回应你的观点。步骤2：提出但是也许有反例。"
        )
        return OllamaResponse(
            text=text,
            tokens_generated=64,
            tokens_prompt=32,
            total_duration_ms=100,
            tokens_per_second=20.0,
            model=model,
        )

    def generate(self, model, prompt, **kwargs):
        self.generate_calls += 1
        if "格式：类型|新鲜度|内容" in prompt:
            text = "insight|8|测试洞见可以继续追问\nquestion|6|是否存在反例"
        elif "概括" in prompt:
            text = "测试摘要"
        else:
            text = "测试话题"
        return OllamaResponse(
            text=text,
            tokens_generated=16,
            tokens_prompt=16,
            total_duration_ms=50,
            tokens_per_second=20.0,
            model=model,
        )

    def is_running(self):
        return True

    def list_models(self):
        return ["qwen2.5:0.5b", "llama3.2:1b", "test-model"]

    def generate_topic(self, model=""):
        return "测试新话题"

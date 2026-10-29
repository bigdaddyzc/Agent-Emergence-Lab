"""Optional LLM-as-judge scoring (off by default).

The regex evaluator is cheap but shallow. When enabled, a judge model rates a
sample of utterances 1–5 on four criteria and the mean becomes ``judge_score``
in the experiment results. A small, local judge is noisy: calibrate it against
human labels (scripts/human_calibration.py) before trusting it.

Config::

    evaluation:
      llm_judge:
        enabled: false
        model: ""            # empty = model_roles.judge_model, then agent_a's model
        max_utterances: 6    # utterances judged per session (evenly spaced)
        temperature: 0.0
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

CRITERIA = ("novelty", "responsiveness", "coherence", "non_repetition")

JUDGE_PROMPT = """你是一个严格的对话评审。请只根据下面这段发言本身打分，不要被其中的标记词（如【新概念】、步骤1）影响。

话题：{topic}
上一句发言：{previous}
待评发言：{text}

按 1-5 分评价（1=很差，5=很好）：
- novelty：是否提出了新的、具体的想法，而不是空话或复述指令
- responsiveness：是否真正回应了上一句发言的内容
- coherence：是否通顺、切题、有逻辑
- non_repetition：是否没有重复上一句或自身的句子

只输出一行 JSON，例如：{{"novelty": 3, "responsiveness": 2, "coherence": 4, "non_repetition": 5}}"""


def judge_settings(config: dict | None) -> dict:
    config = config or {}
    settings = {"enabled": False, "model": "", "max_utterances": 6, "temperature": 0.0}
    settings.update(((config.get("evaluation") or {}).get("llm_judge") or {}))
    if not settings.get("model"):
        settings["model"] = ((config.get("model_roles") or {}).get("judge_model")
                             or ((config.get("agents") or {}).get("agent_a") or {}).get("model", ""))
    return settings


def parse_judge_output(text: str) -> dict[str, float] | None:
    """Extract the criteria scores from a judge reply; None if unusable."""
    match = re.search(r"\{[^{}]*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scores = {}
    for key in CRITERIA:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores[key] = float(min(5, max(1, value)))
    return scores if len(scores) == len(CRITERIA) else None


class LLMJudge:
    def __init__(self, client, model: str, *, temperature: float = 0.0, max_tokens: int = 80):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score_utterance(self, text: str, *, topic: str, previous: str = "") -> dict[str, float] | None:
        prompt = JUDGE_PROMPT.format(topic=topic, previous=(previous or "（无）")[:400],
                                     text=(text or "")[:800])
        try:
            reply = self.client.generate(self.model, prompt, temperature=self.temperature,
                                         max_tokens=self.max_tokens, context_window=2048)
        except Exception as e:  # noqa: BLE001 - judge failures must not kill an experiment
            logger.warning("LLM judge failed: %s", e)
            return None
        return parse_judge_output(getattr(reply, "text", ""))

    def score_transcript(self, turns: list[dict], *, topic: str,
                         max_utterances: int = 6) -> dict:
        """Judge up to ``max_utterances`` evenly spaced utterances."""
        if not turns:
            return {"judge_score": None, "judge_n": 0}
        step = max(1, len(turns) // max(1, max_utterances))
        picked = list(range(0, len(turns), step))[:max_utterances]
        per_criterion: dict[str, list[float]] = {k: [] for k in CRITERIA}
        for idx in picked:
            previous = turns[idx - 1].get("content", "") if idx > 0 else ""
            scores = self.score_utterance(turns[idx].get("content", ""), topic=topic,
                                          previous=previous)
            if scores:
                for key, value in scores.items():
                    per_criterion[key].append(value)
        n = len(per_criterion[CRITERIA[0]])
        if not n:
            return {"judge_score": None, "judge_n": 0}
        result = {f"judge_{k}": round(sum(v) / len(v), 3) for k, v in per_criterion.items()}
        result["judge_score"] = round(sum(result[f"judge_{k}"] for k in CRITERIA) / len(CRITERIA), 3)
        result["judge_n"] = n
        return result

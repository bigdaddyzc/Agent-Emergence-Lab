"""Evaluator v2: template-echo stripping, repetition penalty, LLM judge, calibration."""

import csv
import json

import pytest
import yaml

from src.agent import Agent, AgentConfig, instruction_corpus_from_config
from src.evaluator import (
    EmergenceSignalEvaluator,
    build_reference_ngrams,
    dedupe_for_scoring,
    drop_echo_lines,
    internal_repetition,
    overall_score,
    repetition_score,
    strip_template_echo,
)
from src.experiment import score_transcript
from src.llm_judge import LLMJudge, judge_settings, parse_judge_output
from src.ollama_client import OllamaResponse


def _config():
    with open("configs/low_resource_4g.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _corpus():
    return instruction_corpus_from_config(_config())


# ── Template echo ────────────────────────────────────────────────────────

def test_strip_template_echo_removes_copied_instruction():
    refs = _corpus()
    echo = '先提到对方刚说的一个具体观点（用"你"来指对方）。如果对方的观点有漏洞或可质疑之处，一定要指出来'
    clean, ratio = strip_template_echo(echo, refs)
    assert ratio > 0.9
    assert "具体观点" not in clean


def test_strip_template_echo_keeps_original_text():
    clean, ratio = strip_template_echo("记忆更像一条不断改道的河流，而不是仓库。", _corpus())
    assert ratio == 0.0
    assert clean == "记忆更像一条不断改道的河流，而不是仓库。"


def test_topic_alone_is_not_counted_as_echo():
    # The topic is rendered into the system prompt but left out of the corpus.
    _, ratio = strip_template_echo("一起探索话题记忆的本质很有意思", _corpus())
    assert ratio < 0.5


def test_template_echo_no_longer_scores():
    """Copying the prompt's challenge/reasoning instructions earns no signals."""
    config = _config()
    evaluator = EmergenceSignalEvaluator(config["emergence"])
    echo = ('如果对方的观点有漏洞或可质疑之处，一定要指出来——用"但是……"或"有没有可能……"\n'
            "用因果链展开：'因为X→所以Y（机制是Z）→这意味着W'")
    raw = sum(evaluator.evaluate_dict(echo).values())
    clean, ratio = evaluator.evaluate_clean(echo, build_reference_ngrams(_corpus()))
    assert raw >= 3          # the old evaluator was fooled
    assert sum(clean.as_dict().values()) == 0
    assert ratio > 0.8


def test_genuine_signals_still_count_after_stripping():
    evaluator = EmergenceSignalEvaluator(_config()["emergence"])
    text = "【新概念：记忆河流】记忆就像河道。但是有没有可能，遗忘才是重写的前提？"
    signals, ratio = evaluator.evaluate_clean(text, build_reference_ngrams(_corpus()))
    assert signals.novel_concepts == 1
    assert signals.cross_domain_analogies == 1
    assert signals.critical_challenges >= 2


def test_duplicate_reasoning_steps_collapse():
    text = ("【步骤1】记忆依赖编码和提取两个过程。\n"
            "【步骤2】记忆依赖编码和提取两个过程。\n"
            "【步骤3】")
    deduped = dedupe_for_scoring(text)
    assert deduped.count("步骤") == 1
    evaluator = EmergenceSignalEvaluator({"reasoning_patterns": ["步骤1", "步骤2", "步骤3"]})
    assert evaluator.evaluate_dict(text)["reasoning_steps"] == 3
    assert evaluator.evaluate_clean(text)[0].reasoning_steps == 1


def test_drop_echo_lines_keeps_own_words():
    refs = build_reference_ngrams(_corpus())
    text = "然后说你的新角度或新想法\n我觉得记忆会在每次回想时被悄悄改写，所以越常想起的事越不可靠。"
    out = drop_echo_lines(text, refs)
    assert "新角度或新想法" not in out
    assert "悄悄改写" in out


# ── Repetition as a penalty ──────────────────────────────────────────────

def test_internal_repetition_detects_loops():
    loop = "编码、提取、编码。" * 4
    assert internal_repetition(loop) > 0.5
    assert internal_repetition("记忆先被编码。之后在睡眠中巩固。最后在提取时被改写。") == 0.0


def test_repetitive_text_scores_lower_overall():
    varied = [
        {"agent_name": "Nova", "content": "【新概念：记忆河流】记忆就像河流，但是河道会改。"},
        {"agent_name": "Riven", "content": "你说河道会改，那遗忘就像泥沙淤积？【质疑：遗忘是否主动】"},
        {"agent_name": "Nova", "content": "有没有可能遗忘是大脑在省电？就像手机清理缓存。"},
    ]
    repeated = [
        {"agent_name": "Nova", "content": "【新概念：记忆河流】记忆就像河流，但是河道会改。"},
        {"agent_name": "Nova", "content": "【新概念：记忆河流】记忆就像河流，但是河道会改。"},
        {"agent_name": "Nova", "content": "【新概念：记忆河流】记忆就像河流，但是河道会改。"},
    ]
    s_varied = score_transcript(varied, "记忆的本质")
    s_repeated = score_transcript(repeated, "记忆的本质")
    assert s_repeated["repetition_combined_mean"] > s_varied["repetition_combined_mean"]
    assert s_repeated["overall_score"] < s_varied["overall_score"]


def test_overall_score_formula():
    assert overall_score(2.0, 0.0) == 2.0
    assert overall_score(2.0, 0.5) == 1.0
    assert overall_score(2.0, 1.0) == 0.0
    assert overall_score(2.0, 0.5, repetition_weight=0.0) == 2.0


def test_repetition_score_uses_previous_and_self():
    assert repetition_score("记忆是河流。记忆是河流。记忆是河流。") > 0.5
    assert repetition_score("记忆像河流在流动", ["记忆像河流在流动"]) == 1.0
    assert repetition_score("完全不同的句子", []) == 0.0


def test_score_transcript_prefers_raw_content_and_reports_echo():
    turns = [{"agent_name": "Nova", "content": "我觉得记忆会被改写。",
              "raw_content": "先提到对方刚说的一个具体观点\n我觉得记忆会被改写。"}]
    scores = score_transcript(turns, "记忆", references=_corpus())
    assert scores["template_echo_ratio"] > 0.3
    assert "raw_signals_per_utterance" in scores and "overall_score" in scores


# ── LLM judge ────────────────────────────────────────────────────────────

class _JudgeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, model, prompt, **kwargs):
        self.calls.append((model, prompt))
        return OllamaResponse(self.reply, 10, 10, 1, 1.0, model)


def test_judge_is_off_by_default_and_model_configurable():
    config = _config()
    settings = judge_settings(config)
    assert settings["enabled"] is False
    assert settings["model"] == config["agents"]["agent_a"]["model"]
    config["evaluation"]["llm_judge"]["model"] = "qwen2.5:3b"
    assert judge_settings(config)["model"] == "qwen2.5:3b"


def test_parse_judge_output():
    assert parse_judge_output('好的 {"novelty": 4, "responsiveness": 3, "coherence": 5, '
                              '"non_repetition": 9}') == {
        "novelty": 4.0, "responsiveness": 3.0, "coherence": 5.0, "non_repetition": 5.0}
    assert parse_judge_output("no json") is None
    assert parse_judge_output('{"novelty": 4}') is None


def test_llm_judge_scores_transcript():
    client = _JudgeClient('{"novelty": 4, "responsiveness": 2, "coherence": 3, "non_repetition": 3}')
    judge = LLMJudge(client, "judge-model")
    turns = [{"content": f"发言{i}"} for i in range(4)]
    result = judge.score_transcript(turns, topic="t", max_utterances=2)
    assert result["judge_n"] == 2
    assert result["judge_score"] == 3.0
    assert all(model == "judge-model" for model, _ in client.calls)


def test_llm_judge_tolerates_bad_output():
    judge = LLMJudge(_JudgeClient("???"), "m")
    assert judge.score_transcript([{"content": "x"}], topic="t") == {"judge_score": None, "judge_n": 0}


def test_experiment_with_judge_adds_judge_score(tmp_path, capsys):
    from src.experiment import run_experiment
    from tests.fakes import FakeOllamaClient

    config = _config()
    config["logging"] = {"base_dir": str(tmp_path), "conversation_dir": str(tmp_path / "c"),
                         "memory_dir": str(tmp_path / "m"), "log_level": "INFO"}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    class JudgingFake(FakeOllamaClient):
        def generate(self, model, prompt, **kwargs):
            if "严格的对话评审" in prompt:
                return OllamaResponse('{"novelty": 2, "responsiveness": 2, "coherence": 2, '
                                      '"non_repetition": 2}', 5, 5, 1, 1.0, model)
            return super().generate(model, prompt, **kwargs)

    results = run_experiment(path, "t", 1, ["single_agent"], 1, client=JudgingFake(),
                             judge_model="judge")
    assert results[0]["scores"]["judge_score"] == 2.0


# ── Human calibration ────────────────────────────────────────────────────

def test_human_calibration_export_and_report(tmp_path):
    from scripts import human_calibration as hc

    jsonl = tmp_path / "s.jsonl"
    rows = [{"type": "session_start", "topic": "t", "config_snapshot": _config()}]
    rows += [{"type": "turn", "agent_name": "Nova", "content": f"【新概念：概念{i}】内容{i}" * (i + 1)}
             for i in range(6)]
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"mode": "dual_basic", "session_id": "s",
                                     "jsonl_path": str(jsonl)}]), encoding="utf-8")
    out = hc.export(manifest, tmp_path / "labels.csv", n=5, seed=1)
    with open(out, encoding="utf-8-sig", newline="") as f:
        sample = list(csv.DictReader(f))
    assert len(sample) == 5
    assert "mode" not in sample[0]                 # blind: mode only in the key file
    assert json.loads(out.with_suffix(".key.json").read_text())["s000"]["mode"] == "dual_basic"

    for i, row in enumerate(sample):
        row["human_score"] = str(1 + i)
        row["human_score_2"] = str(1 + i)
        row["auto_score"] = str(i * 0.5)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hc.FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    result = hc.report(out)
    assert result["labelled"] == 5
    assert result["spearman_human_vs_auto_score"] == pytest.approx(1.0)
    assert result["inter_rater_within_1"] == 1.0


def test_spearman_handles_ties_and_degenerate():
    from scripts.human_calibration import spearman

    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None

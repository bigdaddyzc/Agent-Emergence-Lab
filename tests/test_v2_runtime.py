"""v2 runtime: prompt leakage, anti-repetition, multi-topic statistics, Markdown logs."""

import json

import pytest
import yaml

from src.agent import CHANGE_ANGLE_NUDGE, Agent, AgentConfig
from src.anti_repetition import RepetitionGuard
from src.experiment import (
    build_mode_config,
    format_comparison,
    load_topics,
    run_experiment,
    run_single_agent,
)
from src.logger import Logger, SessionLog, TurnLog
from src.main import build_agents, run_conversation
from src.memory import MemorySystem
from src.ollama_client import OllamaResponse
from src.stats import compare_samples, permutation_test, welch_t
from tests.fakes import FakeOllamaClient

INSTRUCTION_MARKERS = ("请先接对方刚才的话", "推理要求", "当前探索层次", "只说", "不要替",
                       "single-agent baseline", "以上为摘要", "换个话题")


def _config(tmp_path=None):
    with open("configs/low_resource_4g.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if tmp_path is not None:
        config["logging"] = {"base_dir": str(tmp_path), "conversation_dir": str(tmp_path / "c"),
                             "memory_dir": str(tmp_path / "m"), "log_level": "INFO"}
    return config


class RecordingClient(FakeOllamaClient):
    """Records every chat call; optionally returns a fixed (repetitive) reply."""

    def __init__(self, reply=None):
        super().__init__()
        self.calls = []
        self.reply = reply

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self.reply is None:
            return super().chat(model, messages, **kwargs)
        self.chat_calls += 1
        return OllamaResponse(self.reply, 64, 32, 100, 20.0, model)


def _non_system(messages):
    return "\n".join(m["content"] for m in messages if m["role"] != "system")


# ── Prompt leakage ───────────────────────────────────────────────────────

def test_build_messages_keeps_instructions_in_system_only():
    config = _config()
    agent = Agent(AgentConfig(**config["agents"]["agent_a"]), "Riven", None)
    agent.compile_system_prompt("记忆的本质")
    history = [{"agent": "Nova", "content": "记忆像河流。" * 30},
               {"agent": "Riven", "content": "那遗忘呢？"}]
    messages = agent.build_messages(conversation_history=history, depth_level=3,
                                    depth_label="原理机制", transition_to_next_depth=True,
                                    meta_reflection_due=True, nudge=CHANGE_ANGLE_NUDGE)
    assert messages[0]["role"] == "system"
    assert sum(m["role"] == "system" for m in messages) == 1
    assert "推理要求" in messages[0]["content"] and CHANGE_ANGLE_NUDGE in messages[0]["content"]
    rest = _non_system(messages)
    for marker in INSTRUCTION_MARKERS:
        assert marker not in rest
    assert [m["role"] for m in messages[1:]] == ["assistant", "user"]
    assert messages[-1]["content"] == "那遗忘呢？"


def test_build_messages_adds_neutral_cue_for_monologue():
    agent = Agent(AgentConfig("Nova", "m", "", "{topic}", 0.5, 64, 512), "Riven", None)
    agent.compile_system_prompt("t")
    first = agent.build_messages(conversation_history=[])
    assert [m["role"] for m in first] == ["system", "user"]
    mono = agent.build_messages(conversation_history=[{"agent": "Nova", "content": "a"},
                                                      {"agent": "Nova", "content": "b"}],
                                truncate_own_turns=False)
    assert [m["role"] for m in mono] == ["system", "assistant", "user"]
    assert mono[1]["content"] == "a\n\nb"


def test_single_agent_prompt_has_no_mode_description(tmp_path):
    config = build_mode_config(_config(tmp_path), "single_agent")
    client = RecordingClient()
    agent = Agent(AgentConfig(**config["agents"]["agent_a"]), "Riven", client)
    run_single_agent(config, agent, MemorySystem(config, client), Logger(config), "测试", 1)
    for call in client.calls:
        everything = "\n".join(m["content"] for m in call["messages"])
        assert "baseline" not in everything.lower()
        assert "single" not in everything.lower()
        assert "具体例子" in call["messages"][0]["content"]   # same depth label as dual modes


def test_shared_history_contains_only_utterances(tmp_path):
    config = _config(tmp_path)
    config["memory"]["long_term_enabled"] = False
    client = RecordingClient()
    a, b = build_agents(config, client)
    memory = MemorySystem(config, client)
    run_conversation(config, a, b, memory, Logger(config), "测试话题", 2)
    for entry in memory.short_term:
        for marker in INSTRUCTION_MARKERS:
            assert marker not in entry["content"]
    for call in client.calls:
        for marker in INSTRUCTION_MARKERS:
            assert marker not in _non_system(call["messages"])


def test_echoed_instruction_lines_do_not_enter_shared_history(tmp_path):
    config = _config(tmp_path)
    config["memory"]["long_term_enabled"] = False
    echo = ('先提到对方刚说的一个具体观点（用"你"来指对方）\n'
            "我觉得记忆每次被回想都会被改写，所以回忆越多越不可靠。")
    client = RecordingClient(reply=echo)
    a, b = build_agents(config, client)
    memory = MemorySystem(config, client)
    logger = Logger(config)
    run_conversation(config, a, b, memory, logger, "测试话题", 1)
    assert all("具体观点" not in e["content"] for e in memory.short_term)
    rows = [json.loads(line) for line in open(logger._jsonl_path, encoding="utf-8")]
    turn = next(r for r in rows if r["type"] == "turn")
    assert turn["raw_content"] == echo            # raw output kept for honest scoring


def test_topic_switch_is_not_a_fake_utterance(tmp_path):
    config = _config(tmp_path)
    config["memory"]["long_term_enabled"] = False
    config["topic"]["max_turns_before_switch"] = 1
    config["topic"]["turns_per_depth"] = 99
    client = RecordingClient()
    a, b = build_agents(config, client)
    memory = MemorySystem(config, client)
    run_conversation(config, a, b, memory, Logger(config), "旧话题", 2)
    assert all("换个话题" not in e["content"] for e in memory.short_term)
    third = client.calls[2]["messages"]           # first utterance after the switch
    assert "话题已经换成" in third[0]["content"]
    assert "话题已经换成" not in _non_system(third)


# ── Anti-repetition ──────────────────────────────────────────────────────

def test_repeat_penalty_is_configurable_and_sent(tmp_path):
    config = _config(tmp_path)
    config["memory"]["long_term_enabled"] = False
    config["anti_repetition"]["repeat_penalty"] = 1.3
    config["agents"]["agent_b"]["repeat_penalty"] = 1.05
    client = RecordingClient()
    a, b = build_agents(config, client)
    run_conversation(config, a, b, MemorySystem(config, client), Logger(config), "t", 1)
    assert [c["repeat_penalty"] for c in client.calls] == [1.3, 1.05]


def test_repeat_penalty_falls_back_to_model_default():
    agent = Agent(AgentConfig("N", "qwen2.5:0.5b", "", "{topic}", 0.5, 64, 512), "R", None)
    assert agent._get_repeat_penalty() == 1.15


def test_ollama_client_passes_repeat_penalty(monkeypatch):
    from src.ollama_client import OllamaClient

    sent = {}
    client = OllamaClient()

    def fake_request(method, path, json_data=None, **kwargs):
        sent.update(json_data)

        class R:
            status_code = 200

            def json(self):
                return {"message": {"content": "ok"}, "eval_count": 1}
        return R()

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "_json", lambda r: r.json())
    client.chat("m", [{"role": "user", "content": "x"}], repeat_penalty=1.25)
    assert sent["options"]["repeat_penalty"] == 1.25


def test_guard_nudges_after_repetitive_utterance():
    guard = RepetitionGuard({"anti_repetition": {"threshold": 0.6, "max_consecutive": 0}})
    first = guard.observe("记忆是编码、存储和提取的过程。")
    assert not first.repetitive and guard.take_nudge() == ""
    second = guard.observe("记忆是编码、存储和提取的过程。")
    assert second.repetitive and second.nudge_next and not second.end_round
    assert guard.take_nudge() == CHANGE_ANGLE_NUDGE
    assert guard.take_nudge() == ""               # consumed once


def test_guard_end_round_and_end_session():
    guard = RepetitionGuard({"anti_repetition": {"action": "end_round", "max_consecutive": 2}})
    guard.observe("同一句话重复同一句话。")
    v1 = guard.observe("同一句话重复同一句话。")
    v2 = guard.observe("同一句话重复同一句话。")
    assert v1.end_round and not v1.end_session
    assert v2.end_session


def test_guard_disabled():
    guard = RepetitionGuard({"anti_repetition": {"enabled": False}})
    guard.observe("一样的话")
    assert not guard.observe("一样的话").repetitive


def test_conversation_injects_nudge_into_next_system_message(tmp_path):
    config = _config(tmp_path)
    config["memory"]["long_term_enabled"] = False
    config["anti_repetition"].update({"action": "nudge", "max_consecutive": 0})
    client = RecordingClient(reply="记忆就是编码提取编码提取。记忆就是编码提取编码提取。" * 3)
    a, b = build_agents(config, client)
    logger = Logger(config)
    run_conversation(config, a, b, MemorySystem(config, client), logger, "t", 2)
    assert CHANGE_ANGLE_NUDGE in client.calls[1]["messages"][0]["content"]
    assert all(CHANGE_ANGLE_NUDGE not in _non_system(c["messages"]) for c in client.calls)
    rows = [json.loads(line) for line in open(logger._jsonl_path, encoding="utf-8")]
    turns = [r for r in rows if r["type"] == "turn"]
    assert turns[0]["repetition_score"] > 0.6 and turns[1]["nudged"] is True


def test_conversation_end_round_skips_partner_and_ends_early(tmp_path):
    config = _config(tmp_path)
    config["memory"]["long_term_enabled"] = False
    config["anti_repetition"].update({"action": "end_round", "max_consecutive": 2})
    client = RecordingClient(reply="一模一样的回答。一模一样的回答。一模一样的回答。")
    a, b = build_agents(config, client)
    logger = Logger(config)
    run_conversation(config, a, b, MemorySystem(config, client), logger, "t", 5)
    # round 1: A repeats itself -> B skipped; round 2: A again -> 2 in a row -> stop
    assert [c["model"] for c in client.calls] == [a.config.model, a.config.model]
    text = open(logger._jsonl_path, encoding="utf-8").read()
    assert "round_ended_early_repetition" in text and "ended_early_repetition" in text


# ── Multi-topic experiments and significance ─────────────────────────────

def test_load_topics_merges_sources(tmp_path):
    f = tmp_path / "topics.txt"
    f.write_text("# comment\n时间\n\n记忆\n", encoding="utf-8")
    assert load_topics("记忆", ["梦"], f) == ["记忆", "梦", "时间"]
    assert load_topics() == [""]


def test_run_experiment_topics_times_runs(tmp_path, capsys):
    config = _config(tmp_path)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    results = run_experiment(path, "", 1, ["single_agent", "dual_basic"], 2,
                             topics=["甲", "乙", "丙"], client=FakeOllamaClient())
    assert len(results) == 3 * 2 * 2
    assert {r["topic"] for r in results} == {"甲", "乙", "丙"}
    assert {r["topic_index"] for r in results} == {0, 1, 2}
    report = format_comparison(results)
    assert "## Significance vs single_agent" in report
    assert "| dual_basic | overall_score |" in report


def test_experiment_cli_accepts_topics_file(tmp_path, monkeypatch, capsys):
    import src.experiment as experiment

    config = _config(tmp_path)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    topics = tmp_path / "topics.txt"
    topics.write_text("甲\n乙\n", encoding="utf-8")
    real_run = experiment.run_experiment
    monkeypatch.setattr(experiment, "run_experiment",
                        lambda *a, **k: real_run(*a, client=FakeOllamaClient(), **k))
    out = tmp_path / "exp.json"
    assert experiment.main(["--config", str(path), "--topics-file", str(topics), "--turns", "1",
                            "--modes", "single_agent", "--output", str(out)]) == 0
    assert {m["topic"] for m in json.loads(out.read_text(encoding="utf-8"))} == {"甲", "乙"}


def test_single_run_report_says_no_significance(tmp_path, capsys):
    results = [{"mode": "single_agent", "topic": "t", "scores": {"utterances": 2, "overall_score": 1}},
               {"mode": "dual_basic", "topic": "t", "scores": {"utterances": 2, "overall_score": 2}}]
    report = format_comparison(results)
    assert "Not enough sessions for a significance test" in report


def test_permutation_test_exact_and_obvious():
    assert permutation_test([1, 2, 3], [1, 2, 3]) == 1.0
    p = permutation_test([10, 11, 12, 13, 14], [0, 1, 2, 3, 4])
    assert p == pytest.approx(2 / 252)          # exact: only the two extreme splits
    assert permutation_test([1.0] * 12, [2.0] * 12, n_resamples=500) < 0.01


def test_compare_samples_with_and_without_scipy(monkeypatch):
    import src.stats as stats

    a, b = [5.0, 6.0, 7.0, 6.5], [1.0, 2.0, 1.5, 2.5]
    with_any = compare_samples(a, b)
    assert with_any["p"] is not None and with_any["p"] < 0.05
    assert with_any["cliffs_delta"] == 1.0
    monkeypatch.setattr(stats, "_scipy_stats", None)
    fallback = stats.compare_samples(a, b)
    assert fallback["test"] == "permutation" and fallback["p"] < 0.05
    assert stats.compare_samples([1.0], [2.0])["p"] is None


def test_welch_t_matches_hand_computation():
    t, df = welch_t([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert t == pytest.approx(-3.674, abs=1e-3)
    assert df == pytest.approx(4.0, abs=1e-6)


# ── Markdown logs ────────────────────────────────────────────────────────

def _turn(logger, turn_number, name, role="", rep=None, nudged=False, raw=None):
    return TurnLog(session_id=logger.session_id, turn_number=turn_number, agent_name=name,
                   agent_role=role, model="m", temperature=0.5, content=f"{name} 的话",
                   token_count_prompt=1, token_count_response=2, duration_ms=3,
                   tokens_per_second=4.0, memory_count=0, timestamp="2026-01-01T00:00:01",
                   repetition_score=rep, nudged=nudged, raw_content=raw)


def test_markdown_rounds_roles_and_repetition(tmp_path):
    logger = Logger(_config(tmp_path))
    logger.log_session_metadata(SessionLog(
        session_id=logger.session_id, topic="t", config_snapshot={},
        agents=[{"name": "Nova", "model": "qwen", "role": ""},
                {"name": "Riven", "model": "llama", "role": "critic"}],
        start_time="s", end_time="", total_turns=2, total_tokens=0, total_duration_ms=0))
    logger.log_turn(_turn(logger, 0, "Nova", rep=0.0))
    logger.log_turn(_turn(logger, 0, "Riven", role="critic", rep=0.81, nudged=True))
    logger.log_turn(_turn(logger, 1, "Nova", rep=0.12, raw="原始 输出"))
    md = open(logger._md_path, encoding="utf-8").read()
    assert md.count("## Round 1") == 1 and md.count("## Round 2") == 1
    assert "## Turn" not in md
    assert "()" not in md
    assert "### #1 Nova - 00:00:01" in md
    assert "### #2 Riven (critic)" in md
    assert "Nova (qwen) x Riven (llama, critic)" in md
    assert "repetition 0.81" in md and "change-angle nudge" in md
    assert "raw model output" in md and "原始 输出" in md

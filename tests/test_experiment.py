import json

import pytest
import yaml

from src.agent import Agent, AgentConfig
from src.experiment import (
    build_mode_config,
    compare_modes,
    format_comparison,
    main as experiment_main,
    pick_baseline,
    run_experiment,
    run_single_agent,
    score_session,
    score_transcript,
)
from src.logger import Logger
from src.memory import MemorySystem
from tests.fakes import FakeOllamaClient


def _config():
    with open("configs/low_resource_4g.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_dual_basic_disables_memory_and_depth():
    config = build_mode_config(_config(), "dual_basic")
    assert config["memory"]["long_term_enabled"] is False
    assert config["memory"]["max_memories_per_prompt"] == 0
    assert config["topic"]["turns_per_depth"] > 1000


def test_full_preserves_memory_setting():
    base = _config()
    config = build_mode_config(base, "full")
    assert config["memory"]["long_term_enabled"] == base["memory"]["long_term_enabled"]


def test_single_agent_mode_config():
    config = build_mode_config(_config(), "single_agent")
    assert config["memory"]["long_term_enabled"] is False
    assert config["topic"]["turns_per_depth"] > 1000


def test_run_single_agent_with_fake_client(tmp_path):
    config = build_mode_config(_config(), "single_agent")
    config["logging"] = {
        "base_dir": str(tmp_path),
        "conversation_dir": str(tmp_path / "conversations"),
        "memory_dir": str(tmp_path / "memory"),
        "log_level": "INFO",
    }
    client = FakeOllamaClient()
    agent = Agent(AgentConfig(**config["agents"]["agent_a"]), "self", client)
    memory = MemorySystem(config, client)
    logger = Logger(config)
    run_single_agent(config, agent, memory, logger, "测试", 2)
    text = open(logger._jsonl_path, encoding="utf-8").read()
    assert '"type": "turn"' in text
    # 2 rounds x 2 utterances: same generation budget as a dual-agent session
    assert client.chat_calls == 4


# ── Baseline / control experiment ────────────────────────────────────────


def _tmp_config_file(tmp_path):
    config = _config()
    config["logging"] = {
        "base_dir": str(tmp_path),
        "conversation_dir": str(tmp_path / "conversations"),
        "memory_dir": str(tmp_path / "memory"),
        "log_level": "INFO",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_self_dialogue_uses_agent_a_model_for_both_agents():
    base = _config()
    config = build_mode_config(base, "self_dialogue")
    assert config["agents"]["agent_b"]["model"] == base["agents"]["agent_a"]["model"]
    # persona of agent_b is kept, only the model changes
    assert config["agents"]["agent_b"]["name"] == base["agents"]["agent_b"]["name"]
    assert config["memory"]["long_term_enabled"] is False
    assert config["topic"]["turns_per_depth"] > 1000
    # the base config must not be mutated
    assert base["agents"]["agent_b"]["model"] != base["agents"]["agent_a"]["model"]


def test_controlled_trio_shares_memory_and_depth_settings():
    base = _config()
    trio = [build_mode_config(base, m) for m in ("single_agent", "self_dialogue", "dual_basic")]
    assert len({json.dumps(c["memory"], sort_keys=True) for c in trio}) == 1
    assert len({json.dumps(c["topic"], sort_keys=True) for c in trio}) == 1


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_mode_config(_config(), "nope")


def test_score_transcript_counts_signals_and_concepts():
    turns = [
        {"agent_name": "Nova", "content": "【新概念：回声】就像山谷。但是有反例。", "token_count_response": 10},
        {"agent_name": "Riven", "content": "回声这个想法很好，【新概念：回声】还能延伸。", "token_count_response": 10},
    ]
    scores = score_transcript(turns, "山谷回声")
    assert scores["utterances"] == 2
    assert scores["novel_concepts"] == 2
    assert scores["unique_concepts"] == 1
    assert scores["concept_uptake"] == 1
    assert scores["cross_domain_analogies"] == 1
    assert scores["critical_challenges"] >= 2
    assert scores["signals_per_utterance"] == pytest.approx(scores["signal_total"] / 2, abs=1e-3)
    assert scores["signals_per_100_tokens"] == pytest.approx(scores["signal_total"] * 5, abs=1e-3)
    assert 0.0 <= scores["repetition_mean"] <= 1.0


def test_score_transcript_empty():
    scores = score_transcript([], "x")
    assert scores["utterances"] == 0
    assert scores["signals_per_utterance"] == 0.0
    assert scores["signals_per_100_tokens"] == 0.0


def test_score_session_reads_jsonl(tmp_path):
    path = tmp_path / "s.jsonl"
    rows = [
        {"type": "session_start", "topic": "测试"},
        {"type": "turn", "agent_name": "Nova", "content": "【新概念：A】", "token_count_response": 4},
        {"type": "metric", "name": "x", "value": 1},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    scores = score_session(path)
    assert scores["utterances"] == 1
    assert scores["novel_concepts"] == 1


def test_compare_modes_mean_and_std():
    results = [
        {"mode": "single_agent", "scores": {"signals_per_utterance": 1.0}},
        {"mode": "single_agent", "scores": {"signals_per_utterance": 3.0}},
        {"mode": "dual_basic", "scores": {"signals_per_utterance": 4.0}},
    ]
    summary = compare_modes(results)
    assert summary["single_agent"]["signals_per_utterance"]["mean"] == 2.0
    assert summary["single_agent"]["signals_per_utterance"]["std"] == pytest.approx(1.414, abs=1e-3)
    assert summary["single_agent"]["signals_per_utterance"]["n"] == 2
    assert summary["dual_basic"]["signals_per_utterance"]["std"] == 0.0


def test_pick_baseline_prefers_single_agent():
    assert pick_baseline(["dual_basic", "self_dialogue", "single_agent"]) == "single_agent"
    assert pick_baseline(["dual_basic", "self_dialogue"]) == "self_dialogue"
    assert pick_baseline(["dual_basic", "full"]) is None
    assert pick_baseline(["dual_basic", "full"], "full") == "full"
    assert pick_baseline(["dual_basic"], "full") is None


def test_run_experiment_end_to_end_with_fake_client(tmp_path, capsys):
    config_path = _tmp_config_file(tmp_path)
    client = FakeOllamaClient()
    results = run_experiment(config_path, "共享话题", 2,
                             ["single_agent", "self_dialogue", "dual_basic"], 2,
                             seed=1, client=client)
    assert len(results) == 6
    # every session gets its own log file even when started in the same second
    assert len({r["jsonl_path"] for r in results}) == 6
    assert {r["topic"] for r in results} == {"共享话题"}
    for r in results:
        # equal utterance budget across control and dual modes: 2 rounds x 2
        assert r["scores"]["utterances"] == 4
        assert r["scores"]["novel_concepts"] == 4
    self_models = next(r["models"] for r in results if r["mode"] == "self_dialogue")
    assert len(set(self_models)) == 1
    report = format_comparison(results)
    assert "| signals_per_utterance |" in report
    assert "Baseline: single_agent" in report
    assert "dual_basic:" in report and "1.00x" in report


def test_run_experiment_shares_generated_topic_across_modes(tmp_path, capsys):
    config_path = _tmp_config_file(tmp_path)
    results = run_experiment(config_path, "", 1, ["single_agent", "dual_basic"], 1,
                             client=FakeOllamaClient())
    assert {r["topic"] for r in results} == {"测试新话题"}


def test_run_experiment_validates_arguments(tmp_path):
    config_path = _tmp_config_file(tmp_path)
    with pytest.raises(ValueError):
        run_experiment(config_path, "t", 0, ["single_agent"], 1, client=FakeOllamaClient())
    with pytest.raises(ValueError):
        run_experiment(config_path, "t", 1, ["single_agent"], 0, client=FakeOllamaClient())
    with pytest.raises(ValueError):
        run_experiment(config_path, "t", 1, ["bogus"], 1, client=FakeOllamaClient())


def test_single_agent_sees_its_full_history(tmp_path):
    config = build_mode_config(_config(), "single_agent")
    config["logging"] = {
        "base_dir": str(tmp_path),
        "conversation_dir": str(tmp_path / "c"),
        "memory_dir": str(tmp_path / "m"),
        "log_level": "INFO",
    }
    client = FakeOllamaClient()
    seen = []
    original = client.chat

    def spy(model, messages, **kwargs):
        seen.append("\n".join(m["content"] for m in messages if m["role"] != "system"))
        return original(model, messages, **kwargs)

    client.chat = spy
    agent = Agent(AgentConfig(**config["agents"]["agent_a"]), "Riven", client)
    run_single_agent(config, agent, MemorySystem(config, client), Logger(config), "测试", 1)
    # second utterance's context contains the first utterance untruncated
    assert "…（你上一轮说的" not in seen[1]
    assert "测试概念1" in seen[1]


def test_single_agent_stops_cleanly_on_ollama_error(tmp_path, capsys):
    from src.ollama_client import OllamaConnectionError

    config = build_mode_config(_config(), "single_agent")
    config["logging"] = {
        "base_dir": str(tmp_path),
        "conversation_dir": str(tmp_path / "c"),
        "memory_dir": str(tmp_path / "m"),
        "log_level": "INFO",
    }
    client = FakeOllamaClient()

    def boom(*args, **kwargs):
        raise OllamaConnectionError("down")

    client.chat = boom
    agent = Agent(AgentConfig(**config["agents"]["agent_a"]), "Riven", client)
    logger = Logger(config)
    run_single_agent(config, agent, MemorySystem(config, client), logger, "测试", 2)
    rows = [json.loads(l) for l in open(logger._jsonl_path, encoding="utf-8")]
    end = [r for r in rows if r["type"] == "session_end"]
    assert end and end[0]["total_turns"] == 0


def test_experiment_cli_writes_manifest_and_report(tmp_path, monkeypatch, capsys):
    import src.experiment as experiment

    config_path = _tmp_config_file(tmp_path)
    real_run = experiment.run_experiment

    def fake_run(*args, **kwargs):
        return real_run(*args, client=FakeOllamaClient(), **kwargs)

    monkeypatch.setattr(experiment, "run_experiment", fake_run)
    out = tmp_path / "out" / "exp.json"
    rc = experiment_main(["--config", str(config_path), "--topic", "t", "--turns", "1",
                          "--modes", "single_agent", "dual_basic", "--output", str(out)])
    assert rc == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert [m["mode"] for m in manifest] == ["single_agent", "dual_basic"]
    assert "scores" in manifest[0]
    assert "# Baseline Comparison" in out.with_suffix(".md").read_text(encoding="utf-8")

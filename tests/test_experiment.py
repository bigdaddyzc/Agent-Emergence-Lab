import yaml

from src.agent import Agent, AgentConfig
from src.experiment import build_mode_config, run_single_agent
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
    assert client.chat_calls == 2

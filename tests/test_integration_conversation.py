import yaml

from src.agent import Agent, AgentConfig
from src.logger import Logger
from src.main import run_conversation
from src.memory import MemorySystem
from tests.fakes import FakeOllamaClient


def test_run_conversation_one_round_with_fake_client(tmp_path):
    with open("configs/low_resource_4g.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["logging"] = {
        "base_dir": str(tmp_path),
        "conversation_dir": str(tmp_path / "conversations"),
        "memory_dir": str(tmp_path / "memory"),
        "log_level": "INFO",
    }
    config["memory"]["long_term_enabled"] = False

    client = FakeOllamaClient()
    agent_a = Agent(AgentConfig(**config["agents"]["agent_a"]), "Riven", client)
    agent_b = Agent(AgentConfig(**config["agents"]["agent_b"]), "Nova", client)
    memory = MemorySystem(config, client)
    logger = Logger(config)

    run_conversation(config, agent_a, agent_b, memory, logger, "测试话题", 1)

    log_text = open(logger._jsonl_path, encoding="utf-8").read()
    assert '"type": "turn"' in log_text
    assert "novel_concepts_nova" in log_text
    assert "novel_concepts_riven" in log_text
    assert client.chat_calls == 2

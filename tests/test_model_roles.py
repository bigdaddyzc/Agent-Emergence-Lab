import yaml

from src.model_roles import get_helper_model, get_judge_model, required_runtime_models


def test_default_helper_reuses_agent_a():
    with open("configs/low_resource_4g.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert get_helper_model(config) == "qwen2.5:0.5b"
    assert get_judge_model(config) == ""
    assert required_runtime_models(config) == ["qwen2.5:0.5b", "llama3.2:1b"]


def test_optional_helper_and_judge_are_required_when_set():
    config = {
        "agents": {
            "agent_a": {"model": "a"},
            "agent_b": {"model": "b"},
        },
        "model_roles": {
            "helper_model": "h",
            "judge_model": "j",
        },
    }
    assert required_runtime_models(config) == ["a", "b", "h", "j"]

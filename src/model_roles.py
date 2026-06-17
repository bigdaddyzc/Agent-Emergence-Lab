"""Model role resolution for optional helper and judge models."""

from __future__ import annotations


def get_agent_model(config: dict, agent_key: str) -> str:
    return config["agents"][agent_key]["model"]


def get_helper_model(config: dict) -> str:
    """Return the model used for memory extraction, compression, and topics."""
    roles = config.get("model_roles", {})
    helper = roles.get("helper_model", "")
    return helper or get_agent_model(config, "agent_a")


def get_judge_model(config: dict) -> str:
    """Return judge model if configured; otherwise an empty string."""
    return config.get("model_roles", {}).get("judge_model", "")


def required_runtime_models(config: dict) -> list[str]:
    """Models that should be installed for the configured runtime."""
    models = [
        get_agent_model(config, "agent_a"),
        get_agent_model(config, "agent_b"),
    ]
    helper = config.get("model_roles", {}).get("helper_model", "")
    judge = config.get("model_roles", {}).get("judge_model", "")
    if helper:
        models.append(helper)
    if judge:
        models.append(judge)
    seen = set()
    result = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result

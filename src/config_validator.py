"""Configuration validation for Agent Emergence Lab."""

from __future__ import annotations


# The conversation loop advances through depth levels 1..5 and indexes
# memory.depth_labels[depth_level - 1], so at least five labels are required.
MAX_DEPTH_LEVEL = 5
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigValidationError(ValueError):
    """Raised when config.yaml is missing required fields or has invalid values."""


def validate_config(config: dict) -> None:
    """Validate the runtime configuration.

    The validator is intentionally lightweight so the project can keep a small
    dependency footprint on low-resource machines.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Config must be a mapping")

    for section in ("agents", "memory", "emergence", "logging", "ollama", "topic"):
        _require(config, section, "config")

    agents = config["agents"]
    for key in ("agent_a", "agent_b"):
        _require(agents, key, "agents")
        _validate_agent(agents[key], f"agents.{key}")

    _validate_memory(config["memory"])
    _validate_logging(config["logging"])
    _validate_ollama(config["ollama"])
    _validate_topic(config["topic"])
    _validate_emergence(config["emergence"])
    _validate_model_roles(config.get("model_roles", {}))
    _validate_anti_repetition(config.get("anti_repetition") or {})
    _validate_evaluation(config.get("evaluation") or {})
    _validate_prompt(config.get("prompt") or {})
    _validate_feishu(config.get("feishu", {}))


def _validate_agent(agent: dict, path: str) -> None:
    for field in (
        "name",
        "model",
        "role",
        "system_prompt_template",
        "temperature",
        "max_tokens",
        "context_window",
    ):
        _require(agent, field, path)

    if not isinstance(agent["name"], str) or not agent["name"].strip():
        raise ConfigValidationError(f"{path}.name must be a non-empty string")
    if not isinstance(agent["model"], str) or not agent["model"].strip():
        raise ConfigValidationError(f"{path}.model must be a non-empty string")
    template = agent["system_prompt_template"]
    if not isinstance(template, str):
        raise ConfigValidationError(f"{path}.system_prompt_template must be a string")
    if "{topic}" not in template:
        raise ConfigValidationError(f"{path}.system_prompt_template must include {{topic}}")
    # Agent.compile_system_prompt() calls str.format(partner_name=..., topic=...);
    # any other placeholder (or a stray "{") would crash the first turn.
    try:
        template.format(partner_name="", topic="")
    except (KeyError, IndexError, ValueError) as e:
        raise ConfigValidationError(
            f"{path}.system_prompt_template may only use {{topic}} and {{partner_name}} "
            f"placeholders (escape literal braces as {{{{ }}}}): {e!r}"
        ) from e
    _range(agent["temperature"], f"{path}.temperature", 0.0, 2.0)
    if agent.get("repeat_penalty") is not None:
        _range(agent["repeat_penalty"], f"{path}.repeat_penalty", 0.5, 2.0)
    unknown = set(agent) - {"name", "model", "role", "system_prompt_template", "temperature",
                            "max_tokens", "context_window", "repeat_penalty"}
    if unknown:
        raise ConfigValidationError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")
    _int_min(agent["max_tokens"], f"{path}.max_tokens", 1)
    _int_min(agent["context_window"], f"{path}.context_window", 512)


def _validate_memory(memory: dict) -> None:
    for field in (
        "short_term_window",
        "extraction_interval",
        "max_memories_per_prompt",
        "consolidation_interval",
        "compression_threshold",
        "turns_kept_verbatim",
    ):
        if field in memory:
            _int_min(memory[field], f"memory.{field}", 0 if field == "turns_kept_verbatim" else 1)
    if "long_term_enabled" in memory and not isinstance(memory["long_term_enabled"], bool):
        raise ConfigValidationError("memory.long_term_enabled must be boolean")
    if "metacognitive_reflection_interval" in memory:
        # Used as a modulo divisor in the conversation loop: 0 would crash.
        _int_min(memory["metacognitive_reflection_interval"],
                 "memory.metacognitive_reflection_interval", 1)
    if "depth_labels" in memory:
        labels = memory["depth_labels"]
        if (not isinstance(labels, list) or len(labels) < MAX_DEPTH_LEVEL
                or not all(isinstance(item, str) and item for item in labels)):
            raise ConfigValidationError(
                f"memory.depth_labels must be a list of at least {MAX_DEPTH_LEVEL} non-empty strings"
            )
    if "novelty_boost_weight" in memory:
        _range(memory["novelty_boost_weight"], "memory.novelty_boost_weight", 0.0, 2.0)


def _validate_logging(logging_config: dict) -> None:
    for field in ("base_dir", "conversation_dir", "memory_dir"):
        _require(logging_config, field, "logging")
        if not isinstance(logging_config[field], str) or not logging_config[field]:
            raise ConfigValidationError(f"logging.{field} must be a non-empty string")
    if "log_level" in logging_config and logging_config["log_level"] not in _LOG_LEVELS:
        raise ConfigValidationError(f"logging.log_level must be one of {', '.join(_LOG_LEVELS)}")


def _validate_ollama(ollama: dict) -> None:
    _require(ollama, "host", "ollama")
    if not isinstance(ollama["host"], str) or not ollama["host"].startswith(("http://", "https://")):
        raise ConfigValidationError("ollama.host must be an HTTP URL")
    if "timeout" in ollama:
        _int_min(ollama["timeout"], "ollama.timeout", 1)


def _validate_topic(topic: dict) -> None:
    for field in ("turns_per_depth", "max_turns_before_switch"):
        if field in topic:
            _int_min(topic[field], f"topic.{field}", 1)
    if "topic_transition_enabled" in topic and not isinstance(topic["topic_transition_enabled"], bool):
        raise ConfigValidationError("topic.topic_transition_enabled must be boolean")


def _validate_emergence(emergence: dict) -> None:
    for key, value in emergence.items():
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigValidationError(f"emergence.{key} must be a list of strings")


def _validate_feishu(feishu: dict) -> None:
    if not feishu:
        return
    if "enabled" in feishu and not isinstance(feishu["enabled"], bool):
        raise ConfigValidationError("feishu.enabled must be boolean")
    if feishu.get("enabled"):
        for field in ("agent_a_webhook", "agent_b_webhook"):
            value = feishu.get(field, "")
            if not isinstance(value, str) or not value.startswith("https://"):
                raise ConfigValidationError(f"feishu.{field} must be an https URL when enabled")
            if "XXXX" in value:
                raise ConfigValidationError(f"feishu.{field} still contains placeholder XXXX")


def _validate_model_roles(model_roles: dict) -> None:
    if not model_roles:
        return
    for field in ("helper_model", "judge_model"):
        if field in model_roles and model_roles[field] is not None and not isinstance(model_roles[field], str):
            raise ConfigValidationError(f"model_roles.{field} must be a string")


def _validate_anti_repetition(section: dict) -> None:
    if "enabled" in section and not isinstance(section["enabled"], bool):
        raise ConfigValidationError("anti_repetition.enabled must be boolean")
    if section.get("repeat_penalty") is not None:
        _range(section["repeat_penalty"], "anti_repetition.repeat_penalty", 0.5, 2.0)
    if "threshold" in section:
        _range(section["threshold"], "anti_repetition.threshold", 0.0, 1.0)
    if "action" in section and section["action"] not in ("nudge", "end_round", "none"):
        raise ConfigValidationError("anti_repetition.action must be nudge, end_round or none")
    if "max_consecutive" in section:
        _int_min(section["max_consecutive"], "anti_repetition.max_consecutive", 0)
    if "window" in section:
        _int_min(section["window"], "anti_repetition.window", 1)


def _validate_evaluation(section: dict) -> None:
    if "repetition_weight" in section:
        _range(section["repetition_weight"], "evaluation.repetition_weight", 0.0, 1.0)
    judge = section.get("llm_judge") or {}
    if "enabled" in judge and not isinstance(judge["enabled"], bool):
        raise ConfigValidationError("evaluation.llm_judge.enabled must be boolean")
    if "model" in judge and judge["model"] is not None and not isinstance(judge["model"], str):
        raise ConfigValidationError("evaluation.llm_judge.model must be a string")
    if "max_utterances" in judge:
        _int_min(judge["max_utterances"], "evaluation.llm_judge.max_utterances", 1)
    if "temperature" in judge:
        _range(judge["temperature"], "evaluation.llm_judge.temperature", 0.0, 2.0)


def _validate_prompt(section: dict) -> None:
    if "strip_instruction_echo" in section and not isinstance(section["strip_instruction_echo"], bool):
        raise ConfigValidationError("prompt.strip_instruction_echo must be boolean")


def _require(mapping: dict, key: str, path: str) -> None:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigValidationError(f"Missing required config field: {path}.{key}")


def _int_min(value, path: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigValidationError(f"{path} must be an integer >= {minimum}")


def _range(value, path: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (minimum <= float(value) <= maximum):
        raise ConfigValidationError(f"{path} must be between {minimum} and {maximum}")

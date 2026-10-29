import copy

import pytest
import yaml

from src.config_validator import ConfigValidationError, validate_config


def _config():
    with open("configs/low_resource_4g.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_low_resource_config_is_valid():
    validate_config(_config())


def test_missing_required_section_fails():
    cfg = _config()
    del cfg["agents"]
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_invalid_temperature_fails():
    cfg = _config()
    cfg["agents"]["agent_a"]["temperature"] = 4
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_enabled_feishu_placeholder_fails():
    cfg = copy.deepcopy(_config())
    cfg["feishu"]["enabled"] = True
    cfg["feishu"]["agent_a_webhook"] = "https://open.feishu.cn/open-apis/bot/v2/hook/XXXX"
    cfg["feishu"]["agent_b_webhook"] = "https://open.feishu.cn/open-apis/bot/v2/hook/abc"
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


@pytest.mark.parametrize("template", [
    "你好 {topic} 和 {unknown}",
    "JSON 示例 {\"a\": 1} 话题 {topic}",
    "未闭合 {topic",
])
def test_bad_prompt_template_placeholders_fail(template):
    cfg = _config()
    cfg["agents"]["agent_a"]["system_prompt_template"] = template
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_escaped_braces_and_partner_name_allowed():
    cfg = _config()
    cfg["agents"]["agent_a"]["system_prompt_template"] = "和 {partner_name} 聊 {topic}，格式 {{json}}"
    validate_config(cfg)


def test_non_string_prompt_template_fails():
    cfg = _config()
    cfg["agents"]["agent_a"]["system_prompt_template"] = ["{topic}"]
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_zero_metacognitive_interval_fails():
    cfg = _config()
    cfg["memory"]["metacognitive_reflection_interval"] = 0
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_too_few_depth_labels_fails():
    cfg = _config()
    cfg["memory"]["depth_labels"] = ["a", "b"]
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_invalid_log_level_fails():
    cfg = _config()
    cfg["logging"]["log_level"] = "LOUD"
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_bool_is_not_accepted_as_integer():
    cfg = _config()
    cfg["agents"]["agent_a"]["max_tokens"] = True
    with pytest.raises(ConfigValidationError):
        validate_config(cfg)


def test_default_config_is_valid():
    with open("config.yaml", "r", encoding="utf-8") as f:
        validate_config(yaml.safe_load(f))

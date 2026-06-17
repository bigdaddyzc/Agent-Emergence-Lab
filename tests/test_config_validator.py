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

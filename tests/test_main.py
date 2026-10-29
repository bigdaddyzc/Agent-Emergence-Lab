"""Tests for the CLI entry point's error handling (no Ollama needed)."""

import sys

import pytest

import src.main as main_mod
from src.ollama_client import OllamaConnectionError


def test_main_exits_cleanly_on_missing_config(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", "does/not/exist.yaml"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2
    assert "找不到配置文件" in capsys.readouterr().out


def test_main_exits_cleanly_on_invalid_config(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("agents: {}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(bad)])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2
    assert "配置文件无效" in capsys.readouterr().out


def test_verify_ollama_handles_model_list_failure(capsys):
    class Client:
        def is_running(self):
            return True

        def list_models(self):
            raise OllamaConnectionError("refused")

    config = {"agents": {"agent_a": {"model": "a"}, "agent_b": {"model": "b"}}}
    with pytest.raises(SystemExit) as exc:
        main_mod.verify_ollama(Client(), config)
    assert exc.value.code == 1
    assert "无法获取" in capsys.readouterr().out

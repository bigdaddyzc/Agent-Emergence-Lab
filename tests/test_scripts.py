import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from analyze_session import analyze, format_summary, load_jsonl
from generate_session_report import generate_report
from install_models import load_models_from_config


def test_install_models_loads_config_models():
    models = load_models_from_config(Path("configs/low_resource_4g.yaml"))
    assert models == ["qwen2.5:0.5b", "llama3.2:1b"]


def test_analyze_session_summarizes_jsonl(tmp_path):
    path = tmp_path / "session.jsonl"
    rows = [
        {"type": "session_start", "session_id": "s1", "topic": "测试"},
        {"type": "turn", "turn_number": 0, "agent_name": "Nova", "token_count_response": 10, "content": "hello"},
        {"type": "metric", "name": "novel_concepts_nova", "value": 1, "turn_number": 0},
        {"type": "metric", "name": "cross_references", "value": 2, "turn_number": 0},
        {"type": "session_end", "session_id": "s1", "topic": "测试", "total_turns": 1},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    loaded = load_jsonl(path)
    summary = analyze(loaded)
    assert summary["session_id"] == "s1"
    assert summary["signal_totals"]["novel_concepts"] == 1
    assert "Session: s1" in format_summary(summary)


def test_analyze_session_accepts_utf8_bom(tmp_path):
    path = tmp_path / "session_bom.jsonl"
    path.write_text('\ufeff{"type": "session_start", "session_id": "s1", "topic": "测试"}\n', encoding="utf-8")
    summary = analyze(load_jsonl(path))
    assert summary["session_id"] == "s1"


def test_generate_report_contains_warning():
    rows = [
        {"type": "session_start", "session_id": "s1", "topic": "测试"},
        {"type": "turn", "turn_number": 0, "agent_name": "Nova", "token_count_response": 10, "content": "【新概念：X】"},
        {"type": "metric", "name": "novel_concepts_nova", "value": 1, "turn_number": 0},
        {"type": "session_end", "session_id": "s1", "topic": "测试", "total_turns": 1},
    ]
    report = generate_report(analyze(rows), rows)
    assert "Session Report: s1" in report
    assert "heuristic signals" in report

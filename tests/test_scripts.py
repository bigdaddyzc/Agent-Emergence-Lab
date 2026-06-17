import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from analyze_session import analyze, format_summary, load_jsonl
from benchmark_models import _format_markdown
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
        {"type": "metric", "name": "concept_events", "value": 1, "turn_number": 0},
        {"type": "metric", "name": "repetition_score_nova", "value": 0.5, "turn_number": 0},
        {"type": "metric", "name": "cross_references", "value": 2, "turn_number": 0},
        {"type": "session_end", "session_id": "s1", "topic": "测试", "total_turns": 1},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    loaded = load_jsonl(path)
    summary = analyze(loaded)
    assert summary["session_id"] == "s1"
    assert summary["signal_totals"]["novel_concepts"] == 1
    assert summary["concept_events"] == 1
    assert summary["quality_totals"]["repetition_score"] == 0.5
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


def test_benchmark_markdown_formatter():
    markdown = _format_markdown([{
        "model": "m",
        "grade": "A",
        "avg_tokens_per_second": 4.2,
        "max_elapsed_s": 10,
        "recommendation": "recommended",
        "runs": [{"prompt": "p", "error": "", "tokens_per_second": 4.2, "sample": "ok"}],
    }])
    assert "| m | A | 4.2 | 10s | recommended |" in markdown

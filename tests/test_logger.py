import json

from src.logger import Logger, SessionLog, TurnLog


def _config(tmp_path):
    return {
        "logging": {
            "base_dir": str(tmp_path),
            "conversation_dir": str(tmp_path / "conversations"),
            "memory_dir": str(tmp_path / "memory"),
            "log_level": "INFO",
        }
    }


def test_logger_writes_jsonl_markdown_and_snapshot(tmp_path):
    logger = Logger(_config(tmp_path))
    session = SessionLog(
        session_id=logger.session_id,
        topic="测试",
        config_snapshot={},
        agents=[{"name": "Nova", "model": "m", "role": ""}, {"name": "Riven", "model": "m", "role": ""}],
        start_time="2026-01-01T00:00:00",
        end_time="",
        total_turns=1,
        total_tokens=0,
        total_duration_ms=0,
    )
    logger.log_session_metadata(session)
    logger.log_turn(TurnLog(
        session_id=logger.session_id,
        turn_number=0,
        agent_name="Nova",
        agent_role="",
        model="m",
        temperature=0.7,
        content="hello",
        token_count_prompt=1,
        token_count_response=2,
        duration_ms=3,
        tokens_per_second=4,
        memory_count=0,
        timestamp="2026-01-01T00:00:01",
    ))
    logger.log_metric("novel_concepts_nova", 1, 0)
    snap = logger.write_memory_snapshot([{"content": "memory"}], 0)
    logger.write_session_end(session)

    rows = [json.loads(line) for line in open(logger._jsonl_path, encoding="utf-8")]
    assert rows[0]["type"] == "session_start"
    assert any(row["type"] == "turn" for row in rows)
    assert any(row["type"] == "metric" for row in rows)
    assert "hello" in open(logger._md_path, encoding="utf-8").read()
    assert json.load(open(snap, encoding="utf-8"))[0]["content"] == "memory"


def test_logger_supports_single_agent_header(tmp_path):
    logger = Logger(_config(tmp_path))
    session = SessionLog(
        session_id=logger.session_id,
        topic="测试",
        config_snapshot={},
        agents=[{"name": "Nova", "model": "m", "role": ""}],
        start_time="2026-01-01T00:00:00",
        end_time="",
        total_turns=1,
        total_tokens=0,
        total_duration_ms=0,
    )
    logger.log_session_metadata(session)
    header = open(logger._md_path, encoding="utf-8").read()
    assert "Nova (m)" in header
    assert "()" not in header and "(m, )" not in header


def test_sessions_started_in_same_second_get_distinct_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(Logger, "_generate_session_id", lambda self: "emergence_fixed")
    config = _config(tmp_path)
    first = Logger(config)
    first.log_metric("x", 1, 0)
    second = Logger(config)
    assert first.session_id == "emergence_fixed"
    assert second.session_id == "emergence_fixed_2"
    assert first._jsonl_path != second._jsonl_path

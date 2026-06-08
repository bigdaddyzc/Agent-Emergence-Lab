"""Logging system: JSON-Lines, Markdown, memory snapshots, metrics."""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnLog:
    session_id: str
    turn_number: int
    agent_name: str
    agent_role: str
    model: str
    temperature: float
    content: str
    token_count_prompt: int
    token_count_response: int
    duration_ms: int
    tokens_per_second: float
    memory_count: int
    timestamp: str


@dataclass
class SessionLog:
    session_id: str
    topic: str
    config_snapshot: dict
    agents: list[dict]
    start_time: str
    end_time: str
    total_turns: int
    total_tokens: int
    total_duration_ms: int


@dataclass
class MetricLog:
    name: str
    value: float
    turn_number: int
    timestamp: str


class Logger:
    """Logs conversation turns, session metadata, memory snapshots, and metrics."""

    def __init__(self, config: dict):
        self.config = config
        self.base_dir = config["logging"]["base_dir"]
        self.conv_dir = config["logging"]["conversation_dir"]
        self.mem_dir = config["logging"]["memory_dir"]
        self.session_id = self._generate_session_id()
        self._jsonl_path = os.path.join(self.conv_dir, f"{self.session_id}.jsonl")
        self._md_path = os.path.join(self.conv_dir, f"{self.session_id}.md")
        self._metrics: list[MetricLog] = []

        os.makedirs(self.conv_dir, exist_ok=True)
        os.makedirs(self.mem_dir, exist_ok=True)

        log_level = getattr(logging, config["logging"].get("log_level", "INFO"))
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def _generate_session_id(self) -> str:
        return f"emergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_turn(self, turn: TurnLog) -> None:
        """Log a single conversation turn to JSONL and Markdown."""
        self._append_jsonl({"type": "turn", **asdict(turn)})
        self._append_md_turn(turn)

    def log_session_metadata(self, session: SessionLog) -> None:
        """Write session metadata as first JSONL entry and at MD file top."""
        self._prepend_jsonl({"type": "session_start", **asdict(session)})
        self._write_md_header(session)

    def log_metric(self, name: str, value: float, turn_number: int) -> None:
        entry = MetricLog(
            name=name,
            value=value,
            turn_number=turn_number,
            timestamp=datetime.now().isoformat(),
        )
        self._metrics.append(entry)
        self._append_jsonl({"type": "metric", **asdict(entry)})

    def write_memory_snapshot(self, entries: list, turn_number: int) -> str:
        """Write memory snapshot JSON and return path."""
        path = os.path.join(
            self.mem_dir,
            f"{self.session_id}_turn{turn_number}_memory.json",
        )
        data = []
        for e in entries:
            if hasattr(e, "__dict__"):
                data.append(e.__dict__ if hasattr(e, "__dict__") else e)
            elif isinstance(e, dict):
                data.append(e)
            else:
                data.append({"content": str(e)})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Memory snapshot saved: %s", path)
        return path

    def write_session_end(self, session: SessionLog) -> None:
        self._append_jsonl({"type": "session_end", **asdict(session)})
        self._append_md_session_end(session, self._metrics)

    # ---- Internal helpers ----

    def _append_jsonl(self, data: dict) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _prepend_jsonl(self, data: dict) -> None:
        """Write as first line by reading existing and rewriting."""
        existing = ""
        if os.path.exists(self._jsonl_path):
            with open(self._jsonl_path, "r", encoding="utf-8") as f:
                existing = f.read()
        with open(self._jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.write(existing)

    def _write_md_header(self, session: SessionLog) -> None:
        lines = [
            f"# Agent Emergence Lab - Session {session.session_id}",
            "",
            f"**Topic:** {session.topic}",
            f"**Started:** {session.start_time}",
            f"**Agents:** {session.agents[0]['name']} ({session.agents[0]['model']}, {session.agents[0]['role']})"
            f" x {session.agents[1]['name']} ({session.agents[1]['model']}, {session.agents[1]['role']})",
            "",
            "---",
            "",
        ]
        with open(self._md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _append_md_turn(self, turn: TurnLog) -> None:
        time_str = turn.timestamp.split("T")[1][:8] if "T" in turn.timestamp else ""
        lines = [
            f"## Turn {turn.turn_number + 1}",
            "",
            f"### {turn.agent_name} ({turn.agent_role}) - {time_str}",
            "",
            turn.content,
            "",
            f"*({turn.token_count_response} tokens, "
            f"{turn.tokens_per_second} tok/s, "
            f"{turn.memory_count} memories injected)*",
            "",
            "---",
            "",
        ]
        with open(self._md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _append_md_session_end(self, session: SessionLog,
                                metrics: list[MetricLog]) -> None:
        lines = [
            "## Session Summary",
            "",
            f"- **Total turns:** {session.total_turns}",
            f"- **Total tokens:** {session.total_tokens}",
            f"- **Total duration:** {self._format_duration(session.total_duration_ms)}",
            f"- **End time:** {session.end_time}",
            "",
        ]
        if metrics:
            lines.append("### Metrics")
            lines.append("")
            lines.append("| Metric | Value | Turn |")
            lines.append("|---|---|---|")
            for m in metrics:
                lines.append(f"| {m.name} | {m.value} | {m.turn_number} |")
            lines.append("")

        with open(self._md_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _format_duration(ms: int) -> str:
        seconds = ms // 1000
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m {seconds}s"

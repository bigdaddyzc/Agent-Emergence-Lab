"""Analyze an Agent Emergence Lab JSONL conversation log."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


SIGNAL_PREFIXES = (
    "novel_concepts_",
    "cross_domain_analogies_",
    "metacognitive_count_",
    "reasoning_steps_",
    "knowledge_generation_",
    "critical_challenges_",
    "concept_elaboration_",
    "question_propagation_",
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze(rows: list[dict]) -> dict:
    turns = [r for r in rows if r.get("type") == "turn"]
    metrics = [r for r in rows if r.get("type") == "metric"]
    session_start = next((r for r in rows if r.get("type") == "session_start"), {})
    session_end = next((r for r in rows if r.get("type") == "session_end"), {})

    agent_turns = Counter(t["agent_name"] for t in turns)
    agent_tokens = defaultdict(int)
    for turn in turns:
        agent_tokens[turn["agent_name"]] += turn.get("token_count_response", 0)

    metric_totals = Counter()
    metric_by_turn = defaultdict(Counter)
    for metric in metrics:
        name = metric["name"]
        value = metric.get("value", 0)
        metric_totals[name] += value
        metric_by_turn[metric.get("turn_number", 0)][name] += value

    signal_totals = Counter()
    for name, value in metric_totals.items():
        for prefix in SIGNAL_PREFIXES:
            if name.startswith(prefix):
                signal_totals[prefix.rstrip("_")] += value

    high_signal_turns = []
    for turn_number, counter in metric_by_turn.items():
        score = sum(v for k, v in counter.items() if any(k.startswith(p) for p in SIGNAL_PREFIXES))
        if score:
            high_signal_turns.append({
                "turn_number": turn_number,
                "score": score,
                "metrics": dict(counter),
            })
    high_signal_turns.sort(key=lambda item: item["score"], reverse=True)

    return {
        "session_id": session_start.get("session_id") or session_end.get("session_id", ""),
        "topic": session_start.get("topic") or session_end.get("topic", ""),
        "total_turn_logs": len(turns),
        "conversation_rounds": session_end.get("total_turns", _infer_rounds(turns)),
        "total_tokens": sum(t.get("token_count_response", 0) for t in turns),
        "agent_turns": dict(agent_turns),
        "agent_response_tokens": dict(agent_tokens),
        "metric_totals": dict(metric_totals),
        "signal_totals": dict(signal_totals),
        "topic_switches": metric_totals.get("topic_switch", 0),
        "memories_extracted": metric_totals.get("memories_extracted", 0),
        "high_signal_turns": high_signal_turns[:10],
    }


def _infer_rounds(turns: list[dict]) -> int:
    if not turns:
        return 0
    return max(t.get("turn_number", 0) for t in turns) + 1


def format_summary(summary: dict) -> str:
    lines = [
        f"Session: {summary['session_id']}",
        f"Topic: {summary['topic']}",
        f"Rounds: {summary['conversation_rounds']}",
        f"Turn logs: {summary['total_turn_logs']}",
        f"Response tokens: {summary['total_tokens']}",
        f"Agent turns: {summary['agent_turns']}",
        f"Signal totals: {summary['signal_totals']}",
        f"Memories extracted: {summary['memories_extracted']}",
        f"Topic switches: {summary['topic_switches']}",
        "High signal turns:",
    ]
    for item in summary["high_signal_turns"]:
        lines.append(f"  - turn {item['turn_number']}: score={item['score']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a conversation JSONL log")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    summary = analyze(load_jsonl(args.jsonl))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

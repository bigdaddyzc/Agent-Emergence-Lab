"""Generate a Markdown report from a conversation JSONL log."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_session import analyze, load_jsonl


def generate_report(summary: dict, rows: list[dict]) -> str:
    turns = [r for r in rows if r.get("type") == "turn"]
    snippets = _high_signal_snippets(summary, turns)
    lines = [
        f"# Session Report: {summary['session_id']}",
        "",
        f"Topic: {summary['topic']}",
        "",
        "## Overview",
        "",
        f"- Conversation rounds: {summary['conversation_rounds']}",
        f"- Turn logs: {summary['total_turn_logs']}",
        f"- Response tokens: {summary['total_tokens']}",
        f"- Memories extracted: {summary['memories_extracted']}",
        f"- Topic switches: {summary['topic_switches']}",
        "",
        "## Agent Activity",
        "",
    ]
    for agent, count in summary["agent_turns"].items():
        tokens = summary["agent_response_tokens"].get(agent, 0)
        avg = round(tokens / max(count, 1), 1)
        lines.append(f"- {agent}: {count} turns, {tokens} tokens, avg {avg} tokens/turn")

    lines.extend([
        "",
        "## Signal Totals",
        "",
    ])
    if summary["signal_totals"]:
        for name, value in sorted(summary["signal_totals"].items()):
            lines.append(f"- {name}: {value}")
    else:
        lines.append("- No emergence-adjacent signals were logged.")

    lines.extend([
        "",
        "## High Signal Turns",
        "",
    ])
    if snippets:
        lines.extend(snippets)
    else:
        lines.append("No high-signal turn snippets were available.")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "These metrics are heuristic signals, not proof of true emergent capability. "
        "Use them to locate interesting dialogue segments, then compare against "
        "baseline runs and human or judge-model review.",
        "",
    ])
    return "\n".join(lines)


def _high_signal_snippets(summary: dict, turns: list[dict]) -> list[str]:
    by_round = {}
    for turn in turns:
        by_round.setdefault(turn.get("turn_number", 0), []).append(turn)

    lines = []
    for item in summary["high_signal_turns"][:5]:
        turn_number = item["turn_number"]
        lines.append(f"### Turn {turn_number} | score {item['score']}")
        lines.append("")
        for turn in by_round.get(turn_number, []):
            content = turn.get("content", "").strip().replace("\n", " ")
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"**{turn.get('agent_name', '?')}**: {content}")
            lines.append("")
        metric_names = ", ".join(sorted(item["metrics"].keys()))
        lines.append(f"Metrics: {metric_names}")
        lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown session report")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    rows = load_jsonl(args.jsonl)
    summary = analyze(rows)
    output = args.output
    if output is None:
        output = Path("logs/reports") / f"{summary['session_id'] or args.jsonl.stem}_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_report(summary, rows), encoding="utf-8")
    print(f"Wrote report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

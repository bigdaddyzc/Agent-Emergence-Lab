"""Baseline / control experiment runner for comparing conversation modes.

The point of this module is to make the "emergence" claim falsifiable: run the
same topic through a control condition and through the dual-agent system, then
score every transcript with the *same* heuristic evaluator so the numbers can be
compared side by side.

Modes
-----
single_agent   one model, one persona, talking to itself (monologue). Gets the
               same utterance budget as a dual run (2 utterances per round).
self_dialogue  the dual-agent loop, but agent_b runs agent_a's model: same
               model, two personas. Isolates the effect of *model diversity*.
dual_basic     two different models, no long-term memory, no depth progression.
dual_memory    dual_basic + long-term memory.
dual_depth     dual_basic + depth progression / topic switching.
full           the system exactly as configured.

single_agent, self_dialogue and dual_basic share the same memory/depth settings,
so they form a controlled trio:

* dual_basic vs single_agent  -> effect of interaction itself
* dual_basic vs self_dialogue -> effect of using two different models
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import Agent, AgentConfig
from src.evaluator import ConceptTracker, DialogueQualityEvaluator, EmergenceSignalEvaluator
from src.logger import Logger, SessionLog, TurnLog
from src.main import load_config, run_conversation, verify_ollama
from src.memory import MemorySystem
from src.ollama_client import OllamaClient, OllamaError


EXPERIMENT_MODES = (
    "single_agent", "self_dialogue", "dual_basic", "dual_memory", "dual_depth", "full",
)
CONTROL_MODES = ("single_agent", "self_dialogue")
DEFAULT_MODES = ["single_agent", "self_dialogue", "dual_basic"]
# Utterances per round in a dual conversation; the monologue baseline gets the
# same budget so totals are comparable.
UTTERANCES_PER_ROUND = 2

# Metrics shown in the comparison report, in order.
REPORT_METRICS = (
    "utterances",
    "signals_per_utterance",
    "signals_per_100_tokens",
    "novel_concepts",
    "unique_concepts",
    "concept_uptake",
    "cross_domain_analogies",
    "critical_challenges",
    "reasoning_steps",
    "metacognitive_count",
    "repetition_mean",
    "topic_drift_mean",
)


def build_mode_config(base_config: dict, mode: str) -> dict:
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unknown experiment mode: {mode}")
    config = copy.deepcopy(base_config)
    memory = config.setdefault("memory", {})
    topic = config.setdefault("topic", {})

    if mode in ("single_agent", "self_dialogue"):
        memory["long_term_enabled"] = False
        memory["max_memories_per_prompt"] = 0
        topic["turns_per_depth"] = 999999
        topic["max_turns_before_switch"] = 999999
        if mode == "self_dialogue":
            agents = config["agents"]
            agents["agent_b"]["model"] = agents["agent_a"]["model"]
    elif mode == "dual_basic":
        memory["long_term_enabled"] = False
        memory["max_memories_per_prompt"] = 0
        topic["turns_per_depth"] = 999999
        topic["max_turns_before_switch"] = 999999
    elif mode == "dual_memory":
        memory["long_term_enabled"] = True
        topic["turns_per_depth"] = 999999
        topic["max_turns_before_switch"] = 999999
    elif mode == "dual_depth":
        memory["long_term_enabled"] = False
        memory["max_memories_per_prompt"] = 0
    elif mode == "full":
        pass
    return config


def run_experiment(config_path: Path, topic: str, turns: int,
                   modes: list[str], runs: int, seed: int | None = None,
                   *, client=None) -> list[dict]:
    """Run every mode ``runs`` times and return one scored result per session.

    All modes in the same run index share one topic, so differences between
    modes are not confounded by different topics. ``client`` can be injected
    (tests pass a fake); otherwise a real OllamaClient is built from config.
    """
    if turns < 1:
        raise ValueError("turns must be >= 1 for an experiment")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    unknown = [m for m in modes if m not in EXPERIMENT_MODES]
    if unknown:
        raise ValueError(f"Unknown experiment mode(s): {', '.join(unknown)}")
    if seed is not None:
        random.seed(seed)

    base_config = load_config(str(config_path))
    if client is None:
        client = OllamaClient(
            base_url=base_config["ollama"]["host"],
            default_timeout=base_config["ollama"]["timeout"],
            keep_alive=base_config["ollama"].get("keep_alive", "5m"),
        )
    verify_ollama(client, base_config)

    results = []
    experiment_id = f"exp_{int(time.time())}"
    for run_idx in range(runs):
        run_topic = topic or client.generate_topic(base_config["agents"]["agent_a"]["model"])
        for mode in modes:
            config = build_mode_config(base_config, mode)
            config.setdefault("experiment", {})
            config["experiment"].update({
                "experiment_id": experiment_id,
                "mode": mode,
                "run_index": run_idx,
                "seed": seed,
                "platform": platform.platform(),
                "python": sys.version,
            })
            agent_a = Agent(AgentConfig(**config["agents"]["agent_a"]),
                            config["agents"]["agent_b"]["name"], client)
            agent_b = Agent(AgentConfig(**config["agents"]["agent_b"]),
                            config["agents"]["agent_a"]["name"], client)
            memory = MemorySystem(config, client)
            logger = Logger(config)
            print(f"\n=== [{experiment_id}] mode={mode} run={run_idx + 1}/{runs} topic={run_topic}")
            if mode == "single_agent":
                run_single_agent(config, agent_a, memory, logger, run_topic, turns)
            else:
                run_conversation(config, agent_a, agent_b, memory, logger, run_topic, turns)
            results.append({
                "experiment_id": experiment_id,
                "mode": mode,
                "run_index": run_idx,
                "session_id": logger.session_id,
                "jsonl_path": logger._jsonl_path,
                "md_path": logger._md_path,
                "topic": run_topic,
                "models": [config["agents"]["agent_a"]["model"]]
                if mode == "single_agent"
                else [config["agents"]["agent_a"]["model"], config["agents"]["agent_b"]["model"]],
                "scores": score_session(Path(logger._jsonl_path), base_config.get("emergence", {})),
            })
    return results


def run_single_agent(config: dict, agent: Agent, memory: MemorySystem,
                     logger: Logger, topic: str, max_turns: int) -> None:
    """Run a single-agent monologue baseline with the same logging shape.

    ``max_turns`` counts *rounds*, like run_conversation. Each round has
    UTTERANCES_PER_ROUND utterances so the baseline gets the same generation
    budget as a dual-agent session. The agent sees its own full history (no
    own-turn truncation), otherwise the baseline would be crippled.
    """
    import datetime

    agent.compile_system_prompt(topic)
    started = time.time()
    started_dt = datetime.datetime.fromtimestamp(started)
    logger.log_session_metadata(SessionLog(
        session_id=logger.session_id,
        topic=topic,
        config_snapshot=config,
        agents=[{"name": agent.name, "model": agent.config.model, "role": agent.role}],
        start_time=started_dt.isoformat(),
        end_time="",
        total_turns=max_turns,
        total_tokens=0,
        total_duration_ms=0,
    ))

    total_tokens = 0
    completed_rounds = 0
    for utterance in range(max_turns * UTTERANCES_PER_ROUND):
        turn = utterance // UTTERANCES_PER_ROUND
        summary, history = memory.get_compressed_context()
        context = agent.build_context(
            conversation_history=history,
            memories=[],
            compressed_summary=summary,
            depth_level=1,
            depth_label="single-agent baseline",
            truncate_own_turns=False,
        )
        try:
            response, stats = agent.think(context)
        except OllamaError as e:
            print(f"\n  ❌ {agent.name} 出错: {e}")
            break
        total_tokens += stats.tokens_generated
        memory.add_turn(agent.name, response, stats.tokens_generated, turn)
        memory.update_prompt_tokens(stats.tokens_prompt)
        if memory.should_compress():
            memory.compress_context(turn)
        logger.log_turn(TurnLog(
            session_id=logger.session_id,
            turn_number=turn,
            agent_name=agent.name,
            agent_role=agent.role,
            model=agent.config.model,
            temperature=agent.config.temperature,
            content=response,
            token_count_prompt=stats.tokens_prompt,
            token_count_response=stats.tokens_generated,
            duration_ms=stats.total_duration_ms,
            tokens_per_second=stats.tokens_per_second,
            memory_count=0,
            timestamp=datetime.datetime.now().isoformat(),
        ))
        if utterance % UTTERANCES_PER_ROUND == UTTERANCES_PER_ROUND - 1:
            completed_rounds = turn + 1

    logger.write_session_end(SessionLog(
        session_id=logger.session_id,
        topic=topic,
        config_snapshot=config,
        agents=[{"name": agent.name, "model": agent.config.model, "role": agent.role}],
        start_time=started_dt.isoformat(),
        end_time=datetime.datetime.now().isoformat(),
        total_turns=completed_rounds,
        total_tokens=total_tokens,
        total_duration_ms=int((time.time() - started) * 1000),
    ))


# ── Scoring ──────────────────────────────────────────────────────────────

def load_turns(jsonl_path: Path) -> tuple[dict, list[dict]]:
    """Return (session_start row, turn rows in log order) from a session JSONL."""
    start: dict = {}
    turns: list[dict] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "session_start" and not start:
            start = row
        elif row.get("type") == "turn":
            turns.append(row)
    return start, turns


def score_transcript(turns: list[dict], topic: str,
                     emergence_config: dict | None = None) -> dict:
    """Score a transcript with the project's heuristic evaluators.

    Every mode is re-scored here from the logged text, so the control and the
    dual-agent runs are measured by exactly the same code. Repetition is
    measured against the *previous utterance in the transcript* (whoever said
    it), which treats a monologue and a dialogue the same way.
    """
    signal_eval = EmergenceSignalEvaluator(emergence_config)
    tracker = ConceptTracker()
    similarity = DialogueQualityEvaluator._similarity

    totals: dict[str, int] = {}
    tokens = 0
    repetition: list[float] = []
    drift: list[float] = []
    previous = ""
    for idx, turn in enumerate(turns):
        text = turn.get("content", "") or ""
        for name, value in signal_eval.evaluate_dict(text).items():
            totals[name] = totals.get(name, 0) + value
        tracker.observe(text, turn_number=idx, agent_name=turn.get("agent_name", "?"))
        tokens += int(turn.get("token_count_response") or 0) or len(text)
        if previous:
            repetition.append(similarity(text, previous))
        drift.append(max(0.0, min(1.0, 1.0 - similarity(text, topic))))
        previous = text

    utterances = len(turns)
    signal_total = sum(totals.values())
    concept_summary = tracker.summary()
    # A concept is "taken up" when it is mentioned again in a later utterance.
    uptake = sum(1 for item in concept_summary.values()
                 if item["mentions"] > 1 and item["last_turn"] > item["first_turn"])
    scores = {
        "utterances": utterances,
        "response_tokens": tokens,
        "signal_total": signal_total,
        "signals_per_utterance": round(signal_total / utterances, 3) if utterances else 0.0,
        "signals_per_100_tokens": round(100 * signal_total / tokens, 3) if tokens else 0.0,
        "unique_concepts": len(concept_summary),
        "concept_uptake": uptake,
        "repetition_mean": round(sum(repetition) / len(repetition), 3) if repetition else 0.0,
        "topic_drift_mean": round(sum(drift) / len(drift), 3) if drift else 0.0,
    }
    scores.update(totals)
    return scores


def score_session(jsonl_path: Path, emergence_config: dict | None = None) -> dict:
    start, turns = load_turns(jsonl_path)
    return score_transcript(turns, start.get("topic", ""), emergence_config)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def compare_modes(results: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate scored results into {mode: {metric: {mean, std, n}}}."""
    by_mode: dict[str, list[dict]] = {}
    for item in results:
        by_mode.setdefault(item["mode"], []).append(item.get("scores", {}))
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for mode, score_list in by_mode.items():
        metrics = sorted({k for s in score_list for k in s})
        summary[mode] = {}
        for metric in metrics:
            mean, std = _mean_std([float(s.get(metric, 0)) for s in score_list])
            summary[mode][metric] = {"mean": round(mean, 3), "std": round(std, 3),
                                     "n": len(score_list)}
    return summary


def pick_baseline(modes: list[str], requested: str | None = None) -> str | None:
    if requested:
        return requested if requested in modes else None
    for mode in CONTROL_MODES:
        if mode in modes:
            return mode
    return None


def format_comparison(results: list[dict], baseline: str | None = None) -> str:
    """Render a Markdown comparison table (mean ± std per mode)."""
    summary = compare_modes(results)
    modes = [m for m in EXPERIMENT_MODES if m in summary]
    baseline = pick_baseline(modes, baseline)
    topics = sorted({r.get("topic", "") for r in results})
    run_counts = ", ".join(
        "{}={}".format(m, summary[m]["utterances"]["n"]) for m in modes if "utterances" in summary[m]
    )
    lines = [
        "# Baseline Comparison",
        "",
        f"- Topics: {', '.join(topics) or '(none)'}",
        f"- Runs per mode: {run_counts}",
        f"- Baseline: {baseline or '(none - add single_agent or self_dialogue)'}",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for metric in REPORT_METRICS:
        cells = []
        for mode in modes:
            stat = summary[mode].get(metric)
            cells.append("-" if stat is None else f"{stat['mean']:g} ± {stat['std']:g}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    if baseline:
        base = summary[baseline].get("signals_per_utterance", {}).get("mean", 0.0)
        lines += ["", f"## signals_per_utterance vs {baseline}", ""]
        for mode in modes:
            if mode == baseline:
                continue
            value = summary[mode].get("signals_per_utterance", {}).get("mean", 0.0)
            ratio = f"{value / base:.2f}x" if base else "n/a"
            lines.append(f"- {mode}: {value:g} vs {base:g} ({ratio})")

    lines += [
        "",
        "> These are regex-based heuristic signals (see src/evaluator.py), not proof",
        "> of emergence. Treat a gap as meaningful only if it holds across several",
        "> topics and runs and survives reading the transcripts. Lower repetition is",
        "> better; topic drift is measured against the starting topic.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline/control experiments and compare them with the evaluator")
    parser.add_argument("--config", type=Path, default=Path("configs/low_resource_4g.yaml"))
    parser.add_argument("--topic", default="",
                        help="Shared topic for every mode (omit to generate one per run)")
    parser.add_argument("--turns", type=int, default=5,
                        help="Rounds per session (one round = 2 utterances)")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=EXPERIMENT_MODES)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline", choices=EXPERIMENT_MODES,
                        help="Mode to compare against (default: single_agent, else self_dialogue)")
    parser.add_argument("--output", type=Path, default=Path("logs/experiments/latest.json"))
    args = parser.parse_args(argv)

    results = run_experiment(args.config, args.topic, args.turns, args.modes, args.runs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = format_comparison(results, args.baseline)
    report_path = args.output.with_suffix(".md")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote experiment manifest: {args.output}")
    print(f"Wrote comparison report:   {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

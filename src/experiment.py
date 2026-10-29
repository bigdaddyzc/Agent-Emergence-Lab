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

from src.agent import Agent, instruction_corpus_from_config
from src.anti_repetition import RepetitionGuard
from src.evaluator import (
    ConceptTracker,
    DialogueQualityEvaluator,
    EmergenceSignalEvaluator,
    build_reference_ngrams,
    drop_echo_lines,
    overall_score,
    repetition_score,
)
from src.llm_judge import LLMJudge, judge_settings
from src.logger import Logger, SessionLog, TurnLog
from src.main import build_agents, load_config, run_conversation, verify_ollama
from src.memory import MemorySystem
from src.ollama_client import OllamaClient, OllamaError
from src.stats import compare_samples, has_scipy


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
    "overall_score",
    "signals_per_utterance",
    "raw_signals_per_utterance",
    "template_echo_ratio",
    "signals_per_100_tokens",
    "novel_concepts",
    "unique_concepts",
    "concept_uptake",
    "cross_domain_analogies",
    "critical_challenges",
    "reasoning_steps",
    "metacognitive_count",
    "repetition_mean",
    "self_repetition_mean",
    "repetition_combined_mean",
    "topic_drift_mean",
    "judge_score",
)
# Metrics that get a significance test against the baseline.
SIGNIFICANCE_METRICS = ("overall_score", "signals_per_utterance", "repetition_combined_mean",
                        "judge_score")


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


def load_topics(topic: str = "", topics: list[str] | None = None,
                topics_file: Path | None = None) -> list[str]:
    """Merge --topic, --topics and --topics-file (one per line, # comments).

    Returns ``[""]`` when nothing was given, meaning "generate one per run".
    """
    merged: list[str] = []
    if topic:
        merged.append(topic)
    merged.extend(t for t in (topics or []) if t)
    if topics_file:
        for line in Path(topics_file).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                merged.append(line)
    seen: set[str] = set()
    unique = [t for t in merged if not (t in seen or seen.add(t))]
    return unique or [""]


def run_experiment(config_path: Path, topic: str, turns: int,
                   modes: list[str], runs: int, seed: int | None = None,
                   *, topics: list[str] | None = None, client=None,
                   judge: bool | None = None, judge_model: str = "") -> list[dict]:
    """Run every topic x run x mode and return one scored result per session.

    All modes in the same (topic, run) cell share one topic, so differences
    between modes are not confounded by different topics. ``topics`` extends
    ``topic``; an empty topic means one is generated per run. ``client`` can be
    injected (tests pass a fake); otherwise a real OllamaClient is built.
    ``judge`` overrides ``evaluation.llm_judge.enabled``.
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
    topic_list = load_topics(topic, topics)

    base_config = load_config(str(config_path))
    if client is None:
        client = OllamaClient(
            base_url=base_config["ollama"]["host"],
            default_timeout=base_config["ollama"]["timeout"],
            keep_alive=base_config["ollama"].get("keep_alive", "5m"),
        )
    verify_ollama(client, base_config)

    jset = judge_settings(base_config)
    if judge is not None:
        jset["enabled"] = judge
    if judge_model:
        jset["model"] = judge_model
        jset["enabled"] = True if judge is None else judge
    llm_judge = LLMJudge(client, jset["model"], temperature=float(jset["temperature"])) \
        if jset["enabled"] and jset["model"] else None

    results = []
    experiment_id = f"exp_{int(time.time())}"
    total_cells = len(topic_list) * runs
    for topic_idx, requested_topic in enumerate(topic_list):
        for run_idx in range(runs):
            run_topic = requested_topic or client.generate_topic(
                base_config["agents"]["agent_a"]["model"])
            for mode in modes:
                config = build_mode_config(base_config, mode)
                config.setdefault("experiment", {})
                config["experiment"].update({
                    "experiment_id": experiment_id,
                    "mode": mode,
                    "topic_index": topic_idx,
                    "run_index": run_idx,
                    "seed": seed,
                    "platform": platform.platform(),
                    "python": sys.version,
                })
                agent_a, agent_b = build_agents(config, client)
                memory = MemorySystem(config, client)
                logger = Logger(config)
                cell = topic_idx * runs + run_idx + 1
                print(f"\n=== [{experiment_id}] mode={mode} cell={cell}/{total_cells} "
                      f"run={run_idx + 1}/{runs} topic={run_topic}")
                if mode == "single_agent":
                    run_single_agent(config, agent_a, memory, logger, run_topic, turns)
                else:
                    run_conversation(config, agent_a, agent_b, memory, logger, run_topic, turns)
                scores = score_session(Path(logger._jsonl_path), base_config.get("emergence", {}))
                if llm_judge is not None:
                    _, logged_turns = load_turns(Path(logger._jsonl_path))
                    scores.update(llm_judge.score_transcript(
                        logged_turns, topic=run_topic,
                        max_utterances=int(jset.get("max_utterances", 6))))
                results.append({
                    "experiment_id": experiment_id,
                    "mode": mode,
                    "topic_index": topic_idx,
                    "run_index": run_idx,
                    "session_id": logger.session_id,
                    "jsonl_path": logger._jsonl_path,
                    "md_path": logger._md_path,
                    "topic": run_topic,
                    "models": [config["agents"]["agent_a"]["model"]]
                    if mode == "single_agent"
                    else [config["agents"]["agent_a"]["model"], config["agents"]["agent_b"]["model"]],
                    "scores": scores,
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
    # Same depth label the dual modes see at depth 1 -- the baseline must not
    # be told it is a "baseline" (that text leaked into its utterances).
    depth_labels = config.get("memory", {}).get("depth_labels") or [""]
    guard = RepetitionGuard(config)
    echo_refs = build_reference_ngrams(agent.instruction_corpus() + list(depth_labels))
    strip_echo = config.get("prompt", {}).get("strip_instruction_echo", True)
    end_round_at = -1
    for utterance in range(max_turns * UTTERANCES_PER_ROUND):
        turn = utterance // UTTERANCES_PER_ROUND
        if turn == end_round_at:
            continue
        summary, history = memory.get_compressed_context()
        nudge = guard.take_nudge()
        messages = agent.build_messages(
            conversation_history=history,
            memories=[],
            compressed_summary=summary,
            depth_level=1,
            depth_label=depth_labels[0],
            truncate_own_turns=False,
            nudge=nudge,
        )
        try:
            raw, stats = agent.think(messages)
        except OllamaError as e:
            print(f"\n  ❌ {agent.name} 出错: {e}")
            break
        response = drop_echo_lines(raw, echo_refs) if strip_echo else raw
        verdict = guard.observe(response)
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
            repetition_score=verdict.score,
            raw_content=raw if raw != response else None,
            nudged=bool(nudge),
        ))
        if verdict.end_session:
            logger.log_metric("ended_early_repetition", 1, turn)
            break
        if verdict.end_round and utterance % UTTERANCES_PER_ROUND != UTTERANCES_PER_ROUND - 1:
            logger.log_metric("round_ended_early_repetition", 1, turn)
            end_round_at = turn
            completed_rounds = turn + 1
            continue
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
                     emergence_config: dict | None = None,
                     references=None, *, repetition_weight: float = 1.0) -> dict:
    """Score a transcript with the project's heuristic evaluators.

    Every mode is re-scored here from the logged text, so the control and the
    dual-agent runs are measured by exactly the same code.

    * Signals are counted on the model's *raw* output after stripping text
      that copies the instruction corpus (``references``) and collapsing
      repeated sentences/steps (``signals_per_utterance``). The old, unfiltered
      count is kept as ``raw_signals_per_utterance`` for comparison.
    * ``repetition_mean`` is overlap with the previous utterance (whoever said
      it); ``self_repetition_mean`` is repetition inside an utterance;
      ``repetition_combined_mean`` is the per-utterance max of the two.
    * ``overall_score = signals_per_utterance * (1 - weight * repetition_combined_mean)``.
    """
    signal_eval = EmergenceSignalEvaluator(emergence_config)
    tracker = ConceptTracker()
    similarity = DialogueQualityEvaluator._similarity
    from src.evaluator import internal_repetition

    grams = build_reference_ngrams(references) if references and not isinstance(references, set) \
        else references

    totals: dict[str, int] = {}
    raw_total = 0
    tokens = 0
    repetition: list[float] = []
    self_rep: list[float] = []
    combined: list[float] = []
    drift: list[float] = []
    echo: list[float] = []
    previous = ""
    for idx, turn in enumerate(turns):
        content = turn.get("content", "") or ""
        text = turn.get("raw_content") or content
        clean_signals, echo_ratio = signal_eval.evaluate_clean(text, grams)
        for name, value in clean_signals.as_dict().items():
            totals[name] = totals.get(name, 0) + value
        raw_total += sum(signal_eval.evaluate_dict(text).values())
        echo.append(echo_ratio)
        tracker.observe(text, turn_number=idx, agent_name=turn.get("agent_name", "?"))
        tokens += int(turn.get("token_count_response") or 0) or len(text)
        self_rep.append(internal_repetition(text))
        if previous:
            repetition.append(similarity(text, previous))
        combined.append(repetition_score(text, [previous] if previous else []))
        drift.append(max(0.0, min(1.0, 1.0 - similarity(text, topic))))
        previous = text

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    utterances = len(turns)
    signal_total = sum(totals.values())
    concept_summary = tracker.summary()
    # A concept is "taken up" when it is mentioned again in a later utterance.
    uptake = sum(1 for item in concept_summary.values()
                 if item["mentions"] > 1 and item["last_turn"] > item["first_turn"])
    spu = round(signal_total / utterances, 3) if utterances else 0.0
    scores = {
        "utterances": utterances,
        "response_tokens": tokens,
        "signal_total": signal_total,
        "signals_per_utterance": spu,
        "raw_signals_per_utterance": round(raw_total / utterances, 3) if utterances else 0.0,
        "signals_per_100_tokens": round(100 * signal_total / tokens, 3) if tokens else 0.0,
        "template_echo_ratio": avg(echo),
        "unique_concepts": len(concept_summary),
        "concept_uptake": uptake,
        "repetition_mean": avg(repetition),
        "self_repetition_mean": avg(self_rep),
        "repetition_combined_mean": avg(combined),
        "topic_drift_mean": avg(drift),
    }
    scores["overall_score"] = overall_score(spu, scores["repetition_combined_mean"],
                                            repetition_weight=repetition_weight)
    scores.update(totals)
    return scores


def score_session(jsonl_path: Path, emergence_config: dict | None = None) -> dict:
    """Re-score a session log; the instruction corpus comes from its config snapshot."""
    start, turns = load_turns(jsonl_path)
    snapshot = start.get("config_snapshot") or {}
    references = instruction_corpus_from_config(snapshot)
    weight = float(((snapshot.get("evaluation") or {}).get("repetition_weight", 1.0)))
    return score_transcript(turns, start.get("topic", ""), emergence_config, references,
                            repetition_weight=weight)


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

    if baseline:
        lines += ["", f"## Significance vs {baseline}", ""]
        lines.append(_significance_note())
        lines.append("")
        lines.append("| Mode | Metric | mean diff | n (mode/base) | test | p | Cliff's δ |")
        lines.append("|---|---|---|---|---|---|---|")
        base_rows = [r.get("scores", {}) for r in results if r["mode"] == baseline]
        any_test = False
        for mode in modes:
            if mode == baseline:
                continue
            rows = [r.get("scores", {}) for r in results if r["mode"] == mode]
            for metric in SIGNIFICANCE_METRICS:
                a = [float(s[metric]) for s in rows if s.get(metric) is not None]
                b = [float(s[metric]) for s in base_rows if s.get(metric) is not None]
                if not a or not b:
                    continue
                cmp = compare_samples(a, b)
                any_test = any_test or cmp["p"] is not None
                p = "-" if cmp["p"] is None else f"{cmp['p']:g}"
                delta = "-" if cmp["cliffs_delta"] is None else f"{cmp['cliffs_delta']:g}"
                lines.append(f"| {mode} | {metric} | {cmp['mean_diff']:g} | "
                             f"{cmp['n_a']}/{cmp['n_b']} | {cmp['test']} | {p} | {delta} |")
        if not any_test:
            lines += ["", "> Not enough sessions for a significance test (need >= 2 per mode;"
                          " recommended >= 3 topics x 5 runs). Differences above are anecdotal."]

    lines += [
        "",
        "> signals_per_utterance counts signals only after removing text copied from the",
        "> prompt instructions and collapsing repeated sentences; raw_signals_per_utterance",
        "> is the old unfiltered count. overall_score = signals_per_utterance x",
        "> (1 - repetition_combined_mean).",
        "",
        "> These are regex-based heuristic signals (see src/evaluator.py), not proof",
        "> of emergence. Treat a gap as meaningful only if it holds across several",
        "> topics and runs and survives reading the transcripts. Lower repetition is",
        "> better; topic drift is measured against the starting topic.",
    ]
    return "\n".join(lines) + "\n"


def _significance_note() -> str:
    if has_scipy():
        return ("Welch's t-test (scipy; Mann-Whitney p in the JSON). Each session is one sample; "
                "p-values are not corrected for multiple comparisons.")
    return ("Two-sided permutation test on the difference of means (scipy not installed). "
            "Each session is one sample; p-values are not corrected for multiple comparisons.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline/control experiments and compare them with the evaluator")
    parser.add_argument("--config", type=Path, default=Path("configs/low_resource_4g.yaml"))
    parser.add_argument("--topic", default="",
                        help="Shared topic for every mode (omit to generate one per run)")
    parser.add_argument("--topics", nargs="+", default=[],
                        help="Several topics; every mode runs --runs times on each")
    parser.add_argument("--topics-file", type=Path,
                        help="Text file with one topic per line (# comments allowed)")
    parser.add_argument("--judge", action="store_true",
                        help="Also score with the optional LLM judge (evaluation.llm_judge)")
    parser.add_argument("--judge-model", default="",
                        help="Judge model (implies --judge); default: config / model_roles.judge_model")
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

    topics = load_topics(args.topic, args.topics, args.topics_file)
    results = run_experiment(args.config, "", args.turns, args.modes, args.runs, args.seed,
                             topics=topics, judge=True if args.judge else None,
                             judge_model=args.judge_model)
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

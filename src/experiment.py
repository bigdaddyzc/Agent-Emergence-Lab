"""Baseline experiment runner for comparing conversation modes."""

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
from src.logger import Logger, SessionLog, TurnLog
from src.main import load_config, run_conversation, verify_ollama
from src.memory import MemorySystem
from src.ollama_client import OllamaClient


EXPERIMENT_MODES = ("single_agent", "dual_basic", "dual_memory", "dual_depth", "full")


def build_mode_config(base_config: dict, mode: str) -> dict:
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unknown experiment mode: {mode}")
    config = copy.deepcopy(base_config)
    memory = config.setdefault("memory", {})
    topic = config.setdefault("topic", {})

    if mode == "single_agent":
        memory["long_term_enabled"] = False
        memory["max_memories_per_prompt"] = 0
        topic["turns_per_depth"] = 999999
        topic["max_turns_before_switch"] = 999999
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
                   modes: list[str], runs: int, seed: int | None = None) -> list[dict]:
    if seed is not None:
        random.seed(seed)

    base_config = load_config(str(config_path))
    client = OllamaClient(
        base_url=base_config["ollama"]["host"],
        default_timeout=base_config["ollama"]["timeout"],
        keep_alive=base_config["ollama"].get("keep_alive", "5m"),
    )
    verify_ollama(client, base_config)

    results = []
    experiment_id = f"exp_{int(time.time())}"
    for mode in modes:
        for run_idx in range(runs):
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
            run_topic = topic or client.generate_topic(config["agents"]["agent_a"]["model"])
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
            })
    return results


def run_single_agent(config: dict, agent: Agent, memory: MemorySystem,
                     logger: Logger, topic: str, max_turns: int) -> None:
    """Run a single-agent baseline with the same logging shape as conversations."""
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
    for turn in range(max_turns):
        summary, history = memory.get_compressed_context()
        context = agent.build_context(
            conversation_history=history,
            memories=[],
            compressed_summary=summary,
            depth_level=1,
            depth_label="single-agent baseline",
        )
        response, stats = agent.think(context)
        total_tokens += stats.tokens_generated
        memory.add_turn(agent.name, response, stats.tokens_generated, turn)
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

    logger.write_session_end(SessionLog(
        session_id=logger.session_id,
        topic=topic,
        config_snapshot=config,
        agents=[{"name": agent.name, "model": agent.config.model, "role": agent.role}],
        start_time=started_dt.isoformat(),
        end_time=datetime.datetime.now().isoformat(),
        total_turns=max_turns,
        total_tokens=total_tokens,
        total_duration_ms=int((time.time() - started) * 1000),
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run baseline experiment modes")
    parser.add_argument("--config", type=Path, default=Path("configs/low_resource_4g.yaml"))
    parser.add_argument("--topic", default="")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=["dual_basic", "full"],
                        choices=EXPERIMENT_MODES)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("logs/experiments/latest.json"))
    args = parser.parse_args(argv)

    results = run_experiment(args.config, args.topic, args.turns, args.modes, args.runs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote experiment manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

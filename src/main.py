"""Agent Emergence Lab - Main orchestrator.

Runs the conversation loop with real-time terminal output.
"""

import argparse
import datetime
import logging
import os
import random
import sys
import threading
import time

import yaml

# Ensure project root is on sys.path so `from src.xxx` works
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.agent import Agent, AgentConfig
from src.logger import Logger, TurnLog, SessionLog
from src.memory import MemorySystem
from src.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_stop_event = threading.Event()


# ── Terminal UI ──────────────────────────────────────────────────────────

_SEP = "=" * 56
_SUB_SEP = "-" * 56


def _print_banner():
    print()


def _print_turn_start(agent: str, turn: int, is_first_in_round: bool = True):
    if turn == 0 and is_first_in_round:
        print(f"\n  {agent} 先说：")
    else:
        print(f"\n  {agent} 说：")


def _print_turn_complete(agent: str, content: str,
                         tokens: int):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"  [{time_str}]")
    for line in content.strip().split("\n"):
        print(f"  {line}")
    sys.stdout.flush()


def _print_memory(entries: list):
    for e in entries:
        print(f"  🧠 [{e.get('source_agent', '?')}] {e.get('content', '')[:120]}")
    sys.stdout.flush()


def _print_summary(metrics: dict, log_path: str, snap_path: str):
    print(f"\n{_SEP}")
    print(f"  聊完了！共 {metrics['total_turns']} 轮，{metrics['total_tokens']} tokens")
    print(f"  日志: {log_path}")
    print(f"{_SEP}\n")


# ── Helpers ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> str:
    return _project_root


def verify_ollama(client, config: dict) -> None:
    if not client.is_running():
        print("错误: Ollama 未运行。请先启动: ollama serve")
        sys.exit(1)
    models = client.list_models()
    required = [
        config["agents"]["agent_a"]["model"],
        config["agents"]["agent_b"]["model"],
    ]
    for m in required:
        if m not in models:
            print(f"错误: 模型 '{m}' 未找到。请先拉取: ollama pull {m}")
            sys.exit(1)


def init_feishu(config: dict) -> tuple:
    """Initialize Feishu bots from config. Returns (feishu_a, feishu_b) — both None if disabled."""
    feishu_config = config.get("feishu", {})
    if not feishu_config.get("enabled", False):
        return None, None

    webhook_a = feishu_config.get("agent_a_webhook", "")
    webhook_b = feishu_config.get("agent_b_webhook", "")
    if not webhook_a or not webhook_b:
        logger.warning("Feishu enabled but webhook URLs missing in config")
        return None, None
    # Guard against the placeholder URLs shipped in the example config
    if "XXXX" in webhook_a or "XXXX" in webhook_b:
        logger.warning("Feishu enabled but webhook URLs are still placeholders; disabling")
        return None, None

    from src.feishu import FeishuBot
    return (
        FeishuBot(webhook_a, config["agents"]["agent_a"]["name"]),
        FeishuBot(webhook_b, config["agents"]["agent_b"]["name"]),
    )


# ── Conversation Loop ────────────────────────────────────────────────────

def run_conversation(config: dict, agent_a: Agent, agent_b: Agent,
                     memory: MemorySystem, logger_inst: Logger,
                     topic: str, max_turns: int,
                     feishu_a=None, feishu_b=None) -> None:
    """Run the conversation loop with real-time terminal output."""
    global _stop_event
    _stop_event.clear()

    agent_a.compile_system_prompt(topic)
    agent_b.compile_system_prompt(topic)
    session_start = time.time()
    session_start_dt = datetime.datetime.fromtimestamp(session_start)

    logger_inst.log_session_metadata(SessionLog(
        session_id=logger_inst.session_id, topic=topic, config_snapshot=config,
        agents=[{"name": agent_a.name, "model": agent_a.config.model, "role": agent_a.role},
                {"name": agent_b.name, "model": agent_b.config.model, "role": agent_b.role}],
        start_time=session_start_dt.isoformat(), end_time="",
        total_turns=max_turns, total_tokens=0, total_duration_ms=0,
    ))

    _print_banner()
    print(f"  {agent_a.name} 和 {agent_b.name} 在聊：「{topic}」")
    print(f"  日志: {logger_inst._md_path}")
    print()

    total_tokens = 0
    total_duration_ms = 0
    total_refs = 0
    completed_turns = 0
    current_topic = topic
    topic_switch_count = 0

    # Depth progression state
    depth_labels = config.get("memory", {}).get("depth_labels",
        ["具体的例子和观察", "模式与连接", "原理与框架", "应用与推演", "跨领域综合"])
    turns_per_depth = config.get("topic", {}).get("turns_per_depth", 4)
    max_turns_before_switch = config.get("topic", {}).get("max_turns_before_switch", 14)
    depth_level = 1
    turns_at_current_depth = 0
    topic_turns = 0

    # Engagement tracking
    consecutive_low_refs = 0

    # Emergence config
    emergence_config = config.get("emergence", {})
    meta_reflection_interval = config.get("memory", {}).get(
        "metacognitive_reflection_interval", 6)

    turn = 0
    while True:
        if _stop_event.is_set():
            print(f"\n  ⛔ 用户已停止。")
            break

        # Safety limit when --turns is set
        if max_turns > 0 and turn >= max_turns:
            break

        # Compute transition/metacognition flags before building context
        transition_to_next = (
            turns_at_current_depth >= turns_per_depth - 1
            and depth_level < 5
        )
        meta_due = (
            turn > 0
            and turn % meta_reflection_interval == 0
        )

        # ── Agent A ──
        compressed_summary, history = memory.get_compressed_context()
        last_content = history[-1]["content"] if history else topic
        memories_result = memory.retrieve_relevant(last_content, current_topic=current_topic)
        memories = memories_result[0]
        depth_info = memories_result[1] if len(memories_result) > 1 else {}

        context = agent_a.build_context(
            conversation_history=history, memories=memories,
            compressed_summary=compressed_summary,
            depth_level=depth_level,
            depth_label=depth_labels[depth_level - 1] if depth_labels else "",
            refs_last_turn=max(consecutive_low_refs + 1, 1),
            consecutive_low_refs=consecutive_low_refs,
            turns_at_current_depth=turns_at_current_depth,
            turns_per_depth=turns_per_depth,
            transition_to_next_depth=transition_to_next,
            meta_reflection_due=meta_due,
        )
        _print_turn_start(agent_a.name, turn, is_first_in_round=True)

        try:
            response_a, stats_a = agent_a.think(context)
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_a.name, e)
            print(f"\n  ❌ {agent_a.name} 出错: {e}")
            break

        refs_a = agent_a.extract_references(response_a)
        total_refs += refs_a
        memory.add_turn(agent_a.name, response_a, stats_a.tokens_generated, turn)
        memory.update_prompt_tokens(stats_a.tokens_prompt)

        logger_inst.log_turn(TurnLog(
            session_id=logger_inst.session_id, turn_number=turn,
            agent_name=agent_a.name, agent_role=agent_a.role,
            model=agent_a.config.model, temperature=agent_a.config.temperature,
            content=response_a, token_count_prompt=stats_a.tokens_prompt,
            token_count_response=stats_a.tokens_generated,
            duration_ms=stats_a.total_duration_ms,
            tokens_per_second=stats_a.tokens_per_second,
            memory_count=len(memories),
            timestamp=datetime.datetime.now().isoformat(),
        ))
        total_tokens += stats_a.tokens_generated
        total_duration_ms += stats_a.total_duration_ms
        _print_turn_complete(agent_a.name, response_a,
                             stats_a.tokens_generated)

        # ── Emergence signal tracking (Agent A) ──
        signals_a = agent_a.extract_emergence_signals(
            response_a, emergence_config=emergence_config)
        for signal_name, signal_value in signals_a.items():
            if signal_value > 0:
                logger_inst.log_metric(
                    f"{signal_name}_{agent_a.name.lower()}",
                    signal_value, turn,
                )

        if feishu_a:
            feishu_a.send_message(
                f"🗣 {agent_a.name} · 第 {turn + 1} 轮\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{response_a}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ {stats_a.tokens_generated} tokens"
            )

        if _stop_event.is_set():
            break

        # ── Agent B ──
        compressed_summary, history = memory.get_compressed_context()
        memories_result_b = memory.retrieve_relevant(response_a, current_topic=current_topic)
        memories_b = memories_result_b[0]

        context = agent_b.build_context(
            conversation_history=history, memories=memories_b,
            compressed_summary=compressed_summary,
            depth_level=depth_level,
            depth_label=depth_labels[depth_level - 1] if depth_labels else "",
            refs_last_turn=refs_a,
            consecutive_low_refs=consecutive_low_refs,
            turns_at_current_depth=turns_at_current_depth,
            turns_per_depth=turns_per_depth,
            transition_to_next_depth=transition_to_next,
            meta_reflection_due=meta_due,
        )
        _print_turn_start(agent_b.name, turn, is_first_in_round=False)

        try:
            response_b, stats_b = agent_b.think(context)
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_b.name, e)
            print(f"\n  ❌ {agent_b.name} 出错: {e}")
            break

        refs_b = agent_b.extract_references(response_b)
        total_refs += refs_b
        memory.add_turn(agent_b.name, response_b, stats_b.tokens_generated, turn)
        memory.update_prompt_tokens(stats_b.tokens_prompt)

        logger_inst.log_turn(TurnLog(
            session_id=logger_inst.session_id, turn_number=turn,
            agent_name=agent_b.name, agent_role=agent_b.role,
            model=agent_b.config.model, temperature=agent_b.config.temperature,
            content=response_b, token_count_prompt=stats_b.tokens_prompt,
            token_count_response=stats_b.tokens_generated,
            duration_ms=stats_b.total_duration_ms,
            tokens_per_second=stats_b.tokens_per_second,
            memory_count=len(memories_b),
            timestamp=datetime.datetime.now().isoformat(),
        ))
        total_tokens += stats_b.tokens_generated
        total_duration_ms += stats_b.total_duration_ms
        _print_turn_complete(agent_b.name, response_b,
                             stats_b.tokens_generated)

        # ── Emergence signal tracking (Agent B) ──
        signals_b = agent_b.extract_emergence_signals(
            response_b, emergence_config=emergence_config)
        for signal_name, signal_value in signals_b.items():
            if signal_value > 0:
                logger_inst.log_metric(
                    f"{signal_name}_{agent_b.name.lower()}",
                    signal_value, turn,
                )

        if feishu_b:
            feishu_b.send_message(
                f"🗣 {agent_b.name} · 第 {turn + 1} 轮\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{response_b}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ {stats_b.tokens_generated} tokens"
            )

        completed_turns = turn + 1

        # ── Context compression (triggered when approaching context window limit) ──
        if memory.should_compress():
            summary = memory.compress_context(turn)
            print(f"  📦 上下文压缩完成（第 {turn + 1} 轮）")

        # ── Memory extraction ──
        if memory.should_extract(turn) and config["memory"]["long_term_enabled"]:
            new_memories = []
            for text, source in [(response_a, agent_a.name), (response_b, agent_b.name)]:
                extracted = memory.extract_memories(text, source, turn, topic=current_topic)
                memory.long_term.extend(extracted)
                new_memories.extend(extracted)
            if new_memories:
                print(f"  📝 提取了 {len(new_memories)} 条记忆:")
                for m in new_memories:
                    print(f"     [{m.source_agent}] {m.content[:100]}")
                logger_inst.log_metric("memories_extracted", len(new_memories), turn)

        # ── Consolidation ──
        if memory.should_consolidate(turn):
            memory.consolidate()
            snap_path = logger_inst.write_memory_snapshot(memory.long_term, turn)
            print(f"  🔗 记忆合并完成 → {snap_path}")

        # ── Periodic memory emergence stats ──
        if turn > 0 and turn % 5 == 0:
            conn_density = memory.calculate_cross_connection_density()
            logger_inst.log_metric("memory_connection_density", conn_density, turn)
            novel_count = memory.get_novel_memory_count()
            logger_inst.log_metric("memory_novel_count", novel_count, turn)
            type_dist = memory.get_memory_type_distribution()
            for t, c in type_dist.items():
                logger_inst.log_metric(f"memory_type_{t}", c, turn)
            synth_count = memory.get_synthetic_memory_count()
            logger_inst.log_metric("synthetic_memory_count", synth_count, turn)

        logger_inst.log_metric("cross_references", refs_a + refs_b, turn)

        # ── Depth progression + engagement tracking ──
        refs_this_round = refs_a + refs_b
        if refs_this_round <= 1:
            consecutive_low_refs += 1
        else:
            consecutive_low_refs = 0

        topic_turns += 1
        turns_at_current_depth += 1

        # Depth nudge: after enough turns at current level, push deeper
        if turns_at_current_depth >= turns_per_depth and depth_level < 5:
            depth_level += 1
            turns_at_current_depth = 0
            print(f"\n  📊 深入探索：第{depth_level}层——{depth_labels[depth_level - 1]}")
            logger_inst.log_metric("depth_level", depth_level, turn)

        # Safety switch: if stuck at any depth too long, change topic
        # Use adaptive threshold: more time at mid-depth (3-4) for deeper exploration,
        # normal threshold at surface or max depth
        if depth_level == 5 or depth_level <= 2:
            switch_threshold = max_turns_before_switch
        else:
            switch_threshold = max_turns_before_switch + 4

        if topic_turns >= switch_threshold:
            depth_level = 1
            turns_at_current_depth = 0
            topic_turns = 0
            consecutive_low_refs = 0
            topic_switch_count += 1

            # Dynamically generate next topic
            new_topic = agent_b.client.generate_topic(
                model=config["agents"]["agent_a"]["model"]
            )
            # Avoid identical consecutive topics
            retries = 0
            while new_topic.lower() == current_topic.lower() and retries < 3:
                new_topic = agent_b.client.generate_topic(
                    model=config["agents"]["agent_a"]["model"]
                )
                retries += 1

            print(f"\n  --- 聊到了别的：{new_topic} ---")

            if feishu_a and feishu_b:
                msg = f"🔄 聊到了别的：\n「{current_topic}」→「{new_topic}」"
                feishu_a.send_message(msg)
                feishu_b.send_message(msg)

            current_topic = new_topic
            agent_a.compile_system_prompt(current_topic)
            agent_b.compile_system_prompt(current_topic)

            memory.short_term.append({
                "agent": agent_b.name if turn % 2 == 0 else agent_a.name,
                "content": f"诶换个话题，我在想——{current_topic}",
                "token_count": 0,
                "turn_number": turn,
                "timestamp": time.time(),
            })
            logger_inst.log_metric("topic_switch", topic_switch_count, turn)

        turn += 1

    # ── Session end ──
    total_duration = int((time.time() - session_start) * 1000)
    session_end_log = SessionLog(
        session_id=logger_inst.session_id, topic=topic, config_snapshot=config,
        agents=[{"name": agent_a.name, "model": agent_a.config.model, "role": agent_a.role},
                {"name": agent_b.name, "model": agent_b.config.model, "role": agent_b.role}],
        start_time=session_start_dt.isoformat(),
        end_time=datetime.datetime.now().isoformat(),
        total_turns=completed_turns, total_tokens=total_tokens,
        total_duration_ms=total_duration,
    )
    logger_inst.write_session_end(session_end_log)
    snap_path = logger_inst.write_memory_snapshot(memory.long_term, "final")
    logger_inst.log_metric("final_memory_count", len(memory.long_term), completed_turns)

    avg_tok = total_tokens / max(completed_turns, 1)
    metrics = {
        "total_turns": completed_turns,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration,
        "avg_tokens_per_turn": round(avg_tok, 1),
        "total_cross_references": total_refs,
        "memory_count": len(memory.long_term),
        "avg_tokens_per_second": round(total_tokens / max(total_duration / 1000, 1), 2),
    }
    _print_summary(metrics, logger_inst._md_path, snap_path)


# ── Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agent Emergence Lab")
    parser.add_argument("--topic", default=None,
                        help="Conversation topic (omit to auto-generate)")
    parser.add_argument("--turns", type=int, default=0,
                        help="Number of turns (0 = infinite, Ctrl+C to stop)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)

    # Init core components
    ollama_client = OllamaClient(
        base_url=config["ollama"]["host"],
        default_timeout=config["ollama"]["timeout"],
        keep_alive=config["ollama"].get("keep_alive", "5m"),
    )
    verify_ollama(ollama_client, config)

    # Auto-generate topic if not specified
    if not args.topic:
        print("  正在随机生成话题...")
        args.topic = ollama_client.generate_topic(
            model=config["agents"]["agent_a"]["model"]
        )
        print(f"  今日话题：{args.topic}\n")

    agent_a = Agent(AgentConfig(**config["agents"]["agent_a"]),
                    config["agents"]["agent_b"]["name"], ollama_client)
    agent_b = Agent(AgentConfig(**config["agents"]["agent_b"]),
                    config["agents"]["agent_a"]["name"], ollama_client)
    memory = MemorySystem(config, ollama_client)
    logger_inst = Logger(config)

    feishu_a, feishu_b = init_feishu(config)

    try:
        run_conversation(config, agent_a, agent_b, memory, logger_inst,
                        args.topic, args.turns,
                        feishu_a=feishu_a, feishu_b=feishu_b)
    except KeyboardInterrupt:
        _stop_event.set()
        print("\n\n  已中断。部分日志已保存。")
        sys.exit(0)


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (required before any Python commands)
source venv/bin/activate

# Run conversation (terminal mode only; web UI was removed)
python3 src/main.py --topic "your topic" --turns 0
# --turns 0  = infinite (Ctrl+C to stop)
# --turns N  = stop after N rounds
# --topic    omit to auto-generate
# --config   path to config (default: config.yaml)

# Run tests
python3 -m pytest tests/ -v

# Set up from scratch
bash setup.sh

# Start Ollama server
ollama serve

# Check available models
ollama list

# View running conversation (live tail)
tail -f /tmp/emergence_*.log
```

## Project Overview

**Agent Emergence Lab** — a dual-AI conversation system where two differently-prompted agents (Nova + Riven) converse autonomously through local Ollama models. The goal is to observe **emergent capabilities** from their extended interaction: concept creation, cross-domain analogies, structured reasoning, critical thinking, and knowledge synthesis.

## Architecture

### Data flow
```
config.yaml → main.py (orchestrator) → agent_a (Nova) ↔ agent_b (Riven)
                                       → memory.py (dual-store: short-term + long-term)
                                       → logger.py (JSONL + MD files)
                                       → feishu.py (optional webhook push)
                                       → Ollama API (via ollama_client.py)
```

### Source layout (src/)

- **`main.py`** — Entry point. Parses args (`--topic`, `--turns`, `--config`), loads config, initializes all components, runs the conversation turn loop. Each turn: agent_a speaks → agent_b responds → memory/logging/bookkeeping. Handles topic depth progression and automatic topic switching.

- **`agent.py`** — `Agent` class with persona config. `build_context()` assembles prompt from memories + compressed summary + dialogue history. `think()` calls Ollama chat API. `extract_references()` counts partner-name mentions (emergence metric). `extract_emergence_signals()` detects novelty/analogy/metacognition/reasoning patterns in text.

- **`memory.py`** — `MemorySystem` with `MemoryEntry` dataclass. Short-term: sliding window (configurable N turns). Long-term: extracted insights with keyword indexing, Jaccard-overlap retrieval with recency/topic/novelty boosts. Key operations: `extract_memories()` (periodic insight extraction), `consolidate()` (dedup + knowledge synthesis via analyst model), `compress_context()` (summarize old turns when approaching context limit). Automatically tracks cross-connections between co-retrieved memories.

- **`ollama_client.py`** — Wrapper around Ollama REST API (`/api/chat`, `/api/generate`, `/api/embed`). Custom exception hierarchy, 3-attempt retry, `OllamaResponse` dataclass with token/timing stats.

- **`logger.py`** — `Logger` class writes to JSON-Lines (programmatic analysis), Markdown (human reading), and memory snapshot JSONs to `logs/memory/`.

- **`feishu.py`** — `FeishuBot` for pushing agent messages to Feishu/Lark group chat via webhook (optional, enabled in config).

### What was removed / doesn't exist
- **No web UI** — `webui.py` source and `static/index.html` were removed. Only terminal output remains.
- **No `--port` or `--resume` CLI flags** — only `--topic`, `--turns`, `--config`.
- **No Flask dependency in requirements.txt** (only requests, PyYAML, numpy, jieba, pytest).

### Tests (tests/)

- `test_agent.py` — 22 tests covering `clean_response`, `extract_references`, `build_context`, `compile_system_prompt`
- `test_memory.py` — 25 tests covering keyword extraction, similarity, retrieval, depth classification, compression scheduling
- Run with: `python3 -m pytest tests/ -v`
- Missing coverage: `main.py`, `logger.py`, `ollama_client.py`, `feishu.py`

## Agent Personas (current config.yaml)

| Agent | Model | Temp | max_tokens | context_window | Style |
|-------|-------|------|-------------|----------------|-------|
| Nova (agent_a) | qwen2.5:3b | 0.70 | 512 | 5120 | Analytical, structured reasoning, pattern finding |
| Riven (agent_b) | llama3.2:3b | 0.80 | 512 | 5120 | Playful, lateral thinking, analogy-driven |

Key prompt instructions (both agents): concept naming via `【新概念：名称】`, cross-domain analogy, multi-step reasoning chains, critical questioning via `【质疑：问题】`, metacognitive reflection, structured summaries via `【阶段性总结】`.

## Configuration (config.yaml)

Controls everything: agent prompts/params, memory system (extraction_interval, consolidation_interval, min_group_size_for_synthesis, compression_threshold), topic depth progression (turns_per_depth, max_turns_before_switch), emergence signal patterns (regex lists for novelty/analogy/metacognition/reasoning/challenge detection), Ollama connection settings, Feishu webhook URLs.

**Key tunables** (optimized after v1→v2 experiments):
- `extraction_interval: 2` — extract memories every 2 turns
- `consolidation_interval: 3` — merge/dedup every 3 extraction cycles (~6 turns)
- `min_group_size_for_synthesis: 2` — synthesize from 2 related memories
- `turns_per_depth: 6` — minimum turns before depth progression
- `max_turns_before_switch: 20` — forced topic switch threshold
- `metacognitive_reflection_interval: 4` — prompt metacognition every 4 turns

## Emergence Metrics & Analysis

The system automatically tracks (logged as metrics in JSONL):
- **Cross-references**: partner name + linguistic pattern matches per turn
- **Novelty patterns**: `【新概念】`, `Insight:`, etc.
- **Analogy patterns**: `就像`, `类比`, `联想`, etc.
- **Metacognitive patterns**: `我不确定`, `我们是不是`, `反思`, etc.
- **Reasoning patterns**: `第一步…第二步…`, `如果…那么…因为…`, etc.
- **Challenge patterns**: `【质疑】`, `有没有可能`, `反例`, etc.
- **Depth level**: 1-5 progression tracking
- **Memory stats**: connection density, novelty count, type distribution, synthetic count

**Analysis scripts** (ad-hoc, used in context):
```python
# Load memory snapshot
import json
with open('logs/memory/<session>_turn<N>_memory.json') as f:
    mems = json.load(f)

# Load conversation log
with open('logs/conversations/<session>.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
turns = [l for l in lines if l.get('type') == 'turn']
metrics = [l for l in lines if l.get('type') == 'metric']
```

## Logs

- **Conversations**: `logs/conversations/<session_id>.jsonl` (programmatic) + `.md` (human-readable)
- **Memory snapshots**: `logs/memory/<session_id>_turn<N>_memory.json`
- Session ID format: `emergence_YYYYMMDD_HHMMSS`
- **No auto-cleanup**: logs accumulate indefinitely; clean manually.

## Known Project Discrepancies

| README says | Reality |
|---|---|
| Nova model: qwen2.5:7b | Uses **qwen2.5:3b** (intentional, faster on CPU) |
| Web UI at :5000 | **Removed** — only terminal mode |
| Riven temp: 0.9 | Config: **0.80** (tuned for coherence) |

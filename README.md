<div align="center">

# 🧪 Agent Emergence Lab

**Two small AI models talk freely. Something unexpected happens.**

*A dual-agent dialogue system designed to observe emergent capabilities through continuous free conversation*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![No GPU Required](https://img.shields.io/badge/GPU-not%20required-brightgreen)](https://github.com/bigdaddyzc/Agent-Emergence-Lab)
[![Models](https://img.shields.io/badge/Models-Qwen2.5--3B%20%7C%20Llama3.2--3B-orange)](https://ollama.ai/library)

[English](#why-this-project) · [中文](#中文说明) · [Quick Start](#quick-start) · [How It Works](#how-it-works)

</div>

---

> **What is "emergence"?** Abilities that appear in a system that were not present in any of its individual components. This project asks: can two 3B-parameter models, talking freely, produce reasoning neither could achieve alone?

---

## Demo

```
[Turn 12] Nova → Riven
"...这让我想到一个新的框架：如果把记忆本身视为一种认知压缩算法，
那么遗忘不是失败，而是系统在主动优化信息密度——"

[Turn 12] 📊 Emergence signals detected:
  ✦ Novel concept        : 【新概念】认知压缩算法
  ✦ Cross-domain analogy : 记忆 ↔ 压缩算法
  ✦ Metacognition        : 对自身推理过程的反思

[Turn 13] Riven → Nova
"等等，你这个类比有个漏洞——遗忘是无损压缩还是有损压缩？
如果是有损的，那被压缩掉的部分去哪里了？"

[Turn 13] 📊 Emergence signals detected:
  ✦ Critical challenge   : 【质疑】漏洞识别
  ✦ Reasoning depth      : 追问第二层因果
```

> *Actual output from a conversation on the topic "the nature of memory"*

---

## Why This Project

Most multi-agent systems route tasks between specialized agents. This project does something different: **it lets two models with contrasting personalities talk freely**, without a task, without a human in the loop, and watches what happens.

The hypothesis: sustained dialogue between complementary cognitive styles — one analytical, one creative — can surface reasoning patterns that neither model produces in isolation.

---

## Agents

| | **Nova** | **Riven** |
|---|---|---|
| **Model** | `qwen2.5:3b` | `llama3.2:3b` |
| **Cognitive style** | Rational · Systematic · Structured | Creative · Divergent · Intuitive |
| **Strengths** | Logical deduction, step-by-step reasoning | Analogy generation, lateral thinking |
| **Temperature** | 0.70 | 0.80 |

The tension between these two styles is the engine of emergence.

---

## How It Works

### Conversation Architecture

```
Topic Input
    │
    ▼
┌─────────────────────────────────────────┐
│           Dialogue Orchestrator          │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  Nova (A)    │◄─►│   Riven (B)      │ │
│  │  Qwen2.5-3B  │  │   Llama3.2-3B    │ │
│  └──────┬───────┘  └────────┬─────────┘ │
│         │    Dialogue Loop  │           │
│         └────────┬──────────┘           │
│                  │                      │
│         ┌────────▼────────┐             │
│         │  Memory System  │             │
│         │ ┌─────────────┐ │             │
│         │ │ Short-term  │ │             │
│         │ │  (12 turns) │ │             │
│         │ └─────────────┘ │             │
│         │ ┌─────────────┐ │             │
│         │ │  Long-term  │ │             │
│         │ │ extraction  │ │             │
│         │ │  retrieval  │ │             │
│         │ │ consolidate │ │             │
│         │ └─────────────┘ │             │
│         └────────┬────────┘             │
│                  │                      │
│         ┌────────▼────────┐             │
│         │ Emergence Meter │             │
│         └─────────────────┘             │
└─────────────────────────────────────────┘
```

### Two-Tier Memory System

**Short-term (working memory)**
- Sliding window of the last 12 turns
- Full verbatim context for recent exchanges

**Long-term (knowledge base)**
- Extracts 1–3 structured memories every 2 turns via Ollama
- Memory types: `insight` · `fact` · `analogy` · `question` · `metacognitive`
- Retrieval scoring: keyword overlap + topic match + recency bonus + novelty boost
- Consolidation: clusters related memories → synthesizes higher-order insights
- Context compression: auto-summarizes old turns when token budget is exceeded

### Depth Progression

Conversations advance through 5 depth levels automatically:

```
Level 1 → Surface exploration
Level 2 → Concept clarification
Level 3 → Mechanism & principles
Level 4 → Cross-domain connections
Level 5 → Meta-cognitive reflection
```

### Emergence Detection

The system tracks 6 signal types in real time:

| Signal | Example Pattern |
|--------|----------------|
| Novel concept | `【新概念】`, `关键洞见` |
| Cross-domain analogy | `就像`, `类比`, `类似于` |
| Metacognition | `我不确定`, `知识盲区`, `我意识到` |
| Deep reasoning | `第一步`, `因为…所以`, `推导出` |
| Knowledge synthesis | `关键发现`, `我们发现`, `综合来看` |
| Critical challenge | `【质疑】`, `反例`, `这里有个漏洞` |

---

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed

### Installation

```bash
git clone https://github.com/bigdaddyzc/Agent-Emergence-Lab.git
cd Agent-Emergence-Lab

# One-command setup (pulls models + installs dependencies)
bash setup.sh
```

<details>
<summary>Manual setup</summary>

```bash
# Pull models
ollama pull qwen2.5:3b
ollama pull llama3.2:3b

# Create virtualenv
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
</details>

### Run

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start conversation (0 = unlimited turns)
source venv/bin/activate
python3 src/main.py --topic "consciousness and subjective experience" --turns 0

# With turn limit
python3 src/main.py --topic "the nature of creativity" --turns 30
```

---

## Project Structure

```
Agent-Emergence-Lab/
├── config.yaml          # Agent personas, memory params, emergence patterns
├── setup.sh             # One-command environment setup
├── requirements.txt     # Python dependencies
├── src/
│   ├── main.py          # Entry point & dialogue loop orchestration
│   ├── agent.py         # Agent class — persona, context building, inference
│   ├── memory.py        # Two-tier memory system (extraction, retrieval, consolidation)
│   ├── logger.py        # JSONL / Markdown / snapshot logging
│   ├── ollama_client.py # Ollama HTTP API wrapper
│   └── feishu.py        # Optional Feishu/Lark webhook integration
└── tests/               # Unit tests (pytest)
```

---

## Configuration

Key settings in `config.yaml`:

```yaml
# Automatic topic switching
topic:
  topic_transition_enabled: true
  max_turns_before_switch: 20

# Memory tuning
memory:
  short_term_window: 12        # turns kept in working memory
  extraction_interval: 2       # extract memories every N turns
  max_memories_per_prompt: 6   # memories injected per turn
  novelty_boost_weight: 0.25   # reward novel memories in retrieval

# Optional: push each turn to Feishu (Lark) for real-time monitoring
feishu:
  enabled: false
  agent_a_webhook: ""
  agent_b_webhook: ""
```

---

## Example Log Output

Each session produces three log formats:

**`logs/conversations/session_*.md`** — human-readable dialogue  
**`logs/conversations/session_*.jsonl`** — structured data with emergence scores per turn  
**`logs/memory/snapshot_*.json`** — full memory state snapshot

```json
{
  "turn": 14,
  "agent": "Nova",
  "content": "...",
  "emergence": {
    "novel_concepts": 1,
    "analogies": 2,
    "metacognitive": 0,
    "reasoning_steps": 3,
    "challenges": 1
  },
  "memory_retrieved": 4
}
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | **Not required** | Optional (speeds up inference) |
| Storage | 5 GB (models) | 10 GB |
| CPU | Any modern x86/ARM | — |

Runs on: MacBook Air M2 (8 GB), Windows 11 with integrated graphics, standard Linux VMs.

---

## Roadmap

- [ ] Web UI for real-time dialogue visualization
- [ ] Emergence score dashboard (charts over time)
- [ ] Support for more model pairs (Mistral, Phi-3, Gemma)
- [ ] Exportable "emergence report" per session
- [ ] Three-agent configuration

---

## Research Context

Inspired by the ongoing debate in AI research:

- **Wei et al. (2022)** — *Emergent Abilities of Large Language Models*: defined emergence as capabilities absent in small models but present at scale
- **Schaeffer et al. (2023)** — *Are Emergent Abilities of Large Language Models a Mirage?*: argued emergence may be a metric artifact
- **This project's angle**: can emergence be induced at small scale through *interaction structure* rather than parameter count alone?

---

## Contributing

Contributions welcome. Areas of interest:

- New emergence signal patterns (especially for non-Chinese conversations)
- Alternative retrieval methods (vector embeddings, BM25)
- New agent persona configurations
- Evaluation benchmarks for inter-agent emergence

```bash
python3 -m pytest tests/ -v
```

---

## License

MIT © [bigdaddyzc](https://github.com/bigdaddyzc)

---

<div align="center">

**If you find this interesting, leave a ⭐ — it helps others find the project.**

*Questions? Open an [issue](https://github.com/bigdaddyzc/Agent-Emergence-Lab/issues) or start a [discussion](https://github.com/bigdaddyzc/Agent-Emergence-Lab/discussions).*

</div>

---

## 中文说明

**Agent Emergence Lab** 是一个双 Agent 自由对话实验系统。

两个不同人格的小模型（Nova 理性分析型 / Riven 创意发散型）在无人工干预的情况下持续交流，系统自动追踪对话中出现的涌现行为——那些超越单个模型能力边界的推理模式。

**核心特性：**
- 双层记忆（短期工作记忆 + 长期知识提取与检索）
- 5 个深度层级的自动进阶
- 6 类涌现信号实时检测
- 纯本地部署，基于 Ollama，无需 GPU，无需外部 API
- 支持飞书 Webhook 实时推送对话

**快速开始：**
```bash
git clone https://github.com/bigdaddyzc/Agent-Emergence-Lab.git
cd Agent-Emergence-Lab
bash setup.sh
ollama serve  # 另开终端
source venv/bin/activate
python3 src/main.py --topic "意识与主观体验" --turns 0
```

<div align="center">

# 🧪 Agent Emergence Lab

**Two small AI models talk freely. Something unexpected happens.**

*A dual-agent dialogue system designed to observe emergent capabilities through continuous free conversation*

[![CI](https://github.com/bigdaddyzc/Agent-Emergence-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/bigdaddyzc/Agent-Emergence-Lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
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

- Python 3.10+ (CI runs 3.11)
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

### Baseline / Control Experiment

A dual-agent transcript full of `【新概念】` tags proves little on its own: a single
model prompted the same way might produce just as many. `src/experiment.py` runs the
**same topic** through control conditions and the dual-agent system, then re-scores
every transcript with the **same** evaluator (`src/evaluator.py`).

| Mode | What runs | Compare with |
|---|---|---|
| `single_agent` | One model, one persona, monologue (sees its own full history) | — control |
| `self_dialogue` | Dual loop, but both personas run agent_a's model | — control |
| `dual_basic` | Two different models, no long-term memory, no depth progression | vs `single_agent`: effect of interaction · vs `self_dialogue`: effect of model diversity |
| `dual_memory` / `dual_depth` / `full` | dual_basic + memory / + depth / everything as configured | ablations |

`single_agent`, `self_dialogue` and `dual_basic` share identical memory/depth settings, and
every mode gets the same budget: `--turns N` = N rounds = 2N utterances.

```bash
# Default modes: single_agent + self_dialogue + dual_basic (here 6 rounds each, 3 runs)
python3 src/experiment.py --topic "the nature of memory" --turns 6 --runs 3

# Add the full system and pick the baseline explicitly
python3 src/experiment.py --topic "the nature of memory" --turns 6 --runs 3 \
    --modes single_agent dual_basic full --baseline single_agent

# Re-score an old experiment (e.g. after editing emergence patterns)
python3 scripts/compare_experiments.py logs/experiments/latest.json --config config.yaml
```

Outputs `logs/experiments/latest.json` (manifest + per-session scores) and
`logs/experiments/latest.md` (mean ± std per mode, plus `signals_per_utterance` as a
ratio to the baseline). Reported metrics include signals per utterance / per 100 tokens,
concept count and *uptake* (a concept mentioned again in a later utterance), analogies,
challenges, reasoning steps, repetition vs the previous utterance, and drift from the topic.

> The scores are regex heuristics, not proof of emergence. Use several topics and
> `--runs ≥ 3`, and read the transcripts before trusting a gap.

#### Evaluation v2: echo-free signals, repetition penalty, significance

- **Template echo does not score.** Small models often copy the instructions
  ("先提到对方刚说的一个具体观点", "【步骤1】"). Before counting signals, every span that
  shares a 6-character n-gram with the instruction corpus (system prompts + all turn
  instructions; the topic itself is excluded) is removed, and repeated sentences /
  empty or duplicated reasoning steps are collapsed. `signals_per_utterance` is this clean
  count; the old count is kept as `raw_signals_per_utterance`, and `template_echo_ratio`
  reports how much of the output was copied.
- **Repetition is a penalty.** `repetition_mean` (overlap with the previous utterance),
  `self_repetition_mean` (loops inside one utterance) and `repetition_combined_mean`
  (per-utterance max of both). The headline metric is
  `overall_score = signals_per_utterance × (1 − w × repetition_combined_mean)`
  (`evaluation.repetition_weight`, default 1).
- **Optional LLM judge** (off by default): `--judge` or `--judge-model qwen2.5:3b`, or
  `evaluation.llm_judge` in config. Rates sampled utterances 1–5 on novelty,
  responsiveness, coherence and non-repetition → `judge_score`.
- **Human calibration:** `python3 scripts/human_calibration.py export logs/experiments/latest.json --n 30`
  writes a blind CSV (mode kept in a separate key file); fill in `human_score` (1–5,
  optionally `human_score_2` for a second rater), then
  `python3 scripts/human_calibration.py report logs/calibration/labels.csv` prints the
  Spearman correlation of human vs `auto_score` / `judge_score` and inter-rater agreement.
- **Several topics × runs + significance:** `--topics A B C` and/or `--topics-file topics.txt`
  (one per line). The report adds a significance table vs the baseline (Welch's t-test via
  scipy when installed, Mann–Whitney p in the JSON; otherwise an exact/Monte-Carlo
  permutation test in pure Python) with Cliff's δ as effect size. With one session per mode
  it says plainly that no test is possible.

#### Prompt hygiene and anti-repetition

- Instructions, depth hints, memories and summaries go **only in the system message**; the
  conversation is passed as real chat turns (own turns = `assistant`, partner = `user`).
  Mode descriptions (e.g. "single-agent baseline") are never shown to the model, topic
  switches are announced in the system message instead of a fake utterance, and lines in a
  reply that mostly copy the instructions are dropped before the reply enters the shared
  history (`prompt.strip_instruction_echo`; the raw output stays in the JSONL as `raw_content`
  and is what gets scored).
- `anti_repetition.repeat_penalty` (or per-agent `repeat_penalty`) is sent to Ollama.
  After each utterance a repetition score is computed; above `anti_repetition.threshold`
  the next speaker gets a "change angle" nudge in its system message (`action: nudge`), or
  the round ends early (`action: end_round`); `max_consecutive` repetitive utterances in a
  row end the session.
- Markdown logs show one `## Round N` header per round, `### #k Name (role)` per utterance
  (no empty `()`), and the per-utterance repetition score.

#### Recommended settings

The 0.5B/1B models in `configs/low_resource_4g.yaml` are for smoke tests: they copy
instructions and loop, and any metric gap between modes is mostly noise. For results worth
reading use **3B+ models** (e.g. `qwen2.5:3b` + `llama3.2:3b`, as in `config.yaml`) and
**≥ 3 topics × ≥ 5 runs** with **≥ 6 rounds** per session:

```bash
python3 -m src.experiment --config config.yaml --topics-file configs/topics_example.txt --runs 5 --turns 6
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
│   ├── evaluator.py     # Heuristic emergence signals & dialogue quality
│   ├── experiment.py    # Baseline/control experiments (topics × runs) + comparison report
│   ├── anti_repetition.py # Runtime repetition guard (nudge / end round / end session)
│   ├── stats.py         # Welch t / Mann–Whitney (scipy) or permutation test
│   ├── llm_judge.py     # Optional LLM-as-judge scoring (off by default)
│   ├── config_validator.py # Config checks run at startup
│   └── feishu.py        # Optional Feishu/Lark webhook integration
├── scripts/             # Session analysis, reports, model benchmark, experiment comparison,
│                        # human_calibration.py (blind labelling + agreement report)
├── .github/workflows/   # CI: pytest on Python 3.11
└── tests/               # Unit tests (pytest, fake Ollama client — no server needed)
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

**对照实验（基线）：**

光看双 Agent 对话里出现了多少【新概念】并不能说明“涌现”——同样提示词下的单个模型也许一样多。
`src/experiment.py` 让**同一个话题**分别跑对照组和双 Agent 组，再用**同一个评估器**给每份对话打分：

| 模式 | 内容 | 对比意义 |
|---|---|---|
| `single_agent` | 单模型单人格自言自语（能看到自己的完整历史） | 对照组 |
| `self_dialogue` | 双 Agent 流程，但两个人格都用 agent_a 的模型 | 对照组 |
| `dual_basic` | 两个不同模型，无长期记忆、无深度推进 | 对比 `single_agent` 看“交互”本身的作用；对比 `self_dialogue` 看“模型差异”的作用 |
| `dual_memory` / `dual_depth` / `full` | 在 dual_basic 基础上加记忆 / 加深度推进 / 完整系统 | 消融实验 |

`single_agent`、`self_dialogue`、`dual_basic` 的记忆和深度设置完全相同；所有模式生成预算一致：`--turns N` = N 轮 = 2N 次发言。

```bash
# 默认运行 single_agent + self_dialogue + dual_basic
python3 src/experiment.py --topic "记忆的本质" --turns 6 --runs 3

# 加入完整系统并指定基线
python3 src/experiment.py --topic "记忆的本质" --turns 6 --runs 3 \
    --modes single_agent dual_basic full --baseline single_agent

# 修改涌现规则后重新给旧实验打分
python3 scripts/compare_experiments.py logs/experiments/latest.json --config config.yaml
```

结果写入 `logs/experiments/latest.json`（清单 + 每场得分）和 `logs/experiments/latest.md`（各模式均值 ± 标准差，以及相对基线的 `signals_per_utterance` 倍数）。

> 这些分数是基于正则的启发式信号，不是涌现的证明。请用多个话题、`--runs ≥ 3`，并在相信差距之前读一读对话原文。

**评估 v2：去掉模板复述、重复惩罚、显著性检验**

- **复述提示词不再得分。** 小模型常把指令原样抄进回答（"先提到对方刚说的一个具体观点"、"【步骤1】"）。计数前先删除与指令语料（系统提示 + 各轮指令，不含话题本身）共享 6 字 n-gram 的片段，并合并重复句子和空的/重复的推理步骤。`signals_per_utterance` 是清洗后的计数，旧计数保留为 `raw_signals_per_utterance`，`template_echo_ratio` 表示输出中有多少是抄的。
- **重复作为惩罚。** `repetition_mean`（与上一句的重合）、`self_repetition_mean`（单句内部打转）、`repetition_combined_mean`（两者逐句取最大）。总分 `overall_score = signals_per_utterance × (1 − w × repetition_combined_mean)`（`evaluation.repetition_weight`，默认 1）。
- **可选 LLM 评审**（默认关闭）：`--judge` 或 `--judge-model qwen2.5:3b`，或在配置 `evaluation.llm_judge` 中开启；对抽样发言按新颖性、回应性、连贯性、不重复打 1–5 分 → `judge_score`。
- **人工校准：** `python3 scripts/human_calibration.py export logs/experiments/latest.json --n 30` 导出盲评 CSV（模式写在单独的 key 文件里）；填写 `human_score`（1–5，可选第二位评分者 `human_score_2`）后运行 `python3 scripts/human_calibration.py report logs/calibration/labels.csv`，输出人工分与 `auto_score` / `judge_score` 的 Spearman 相关以及评分者间一致性。
- **多话题 × 多次运行 + 显著性：** `--topics A B C` 和/或 `--topics-file topics.txt`（每行一个）。报告新增相对基线的显著性表（装了 scipy 用 Welch t 检验，JSON 中附 Mann–Whitney p；否则用纯 Python 精确/蒙特卡洛置换检验），并给出 Cliff's δ 效应量。每个模式只有一场时会明确说明无法检验。

**提示词卫生与防重复**

- 指令、深度提示、记忆和摘要**只放在 system 消息**里；对话历史以真实的聊天轮次传入（自己说的是 `assistant`，对方说的是 `user`）。模式描述（如 "single-agent baseline"）不再出现在提示中；换话题通过 system 消息告知，不再伪造一句发言；回答中主要在抄指令的行在进入共享历史前被删除（`prompt.strip_instruction_echo`；原始输出保存在 JSONL 的 `raw_content` 中，打分用的是原始输出）。
- `anti_repetition.repeat_penalty`（或单个 agent 的 `repeat_penalty`）会传给 Ollama。每句发言后计算重复度，超过 `anti_repetition.threshold` 时，下一位发言者的 system 消息里会加入“换个角度”的提醒（`action: nudge`），或提前结束本轮（`action: end_round`）；连续 `max_consecutive` 句重复则提前结束对话。
- Markdown 日志每轮一个 `## Round N` 标题，每句发言 `### #k 名字 (角色)`（不再有空的 `()`），并显示逐句重复度。

**推荐设置**

`configs/low_resource_4g.yaml` 里的 0.5B/1B 模型只适合冒烟测试：它们会抄指令、会打转，模式之间的差距大多是噪声。想得到值得看的结果，请用 **3B 及以上模型**（如 `config.yaml` 中的 `qwen2.5:3b` + `llama3.2:3b`），**至少 3 个话题 × 每个 5 次运行**，每场 **≥ 6 轮**：

```bash
python3 -m src.experiment --config config.yaml --topics-file configs/topics_example.txt --runs 5 --turns 6
```

**运行测试：** `python3 -m pytest -q`（使用假的 Ollama 客户端，无需启动 Ollama；CI 在 Python 3.11 上自动运行）。

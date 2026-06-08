# Agent Emergence Lab

双Agent自由对话系统，通过两个不同人格的AI Agent持续对话实现**能力涌现**（Emergent Capabilities），配备记忆体系统和实时Web展示界面。

## 概述

两个具有不同特长的AI Agent（理性分析型 **Nova** + 创意发散型 **Riven**）通过不间断自由对话，在互动中产生超出单个模型能力的涌现行为。系统完整记录对话日志，并通过Web界面实时展示。

### 核心特性

- **双Agent对话**: Nova (Qwen2.5-3B, 分析师) × Riven (Llama 3.2-3B, 创造者)
- **记忆系统**: 双层架构 — 工作记忆（短期）+ 长期记忆（知识提取与检索）
- **完整日志**: JSON-Lines（程序分析）+ Markdown（人工阅读）+ 记忆快照
- **涌现测量**: 自动追踪新概念创造、跨域类比、批判性质疑等指标
- **纯本地部署**: 通过Ollama运行，无需GPU，无需外部API

## 快速开始

### 一键安装

```bash
bash setup.sh
```

### 手动安装

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull qwen2.5:3b
ollama pull llama3.2:3b

# 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 运行

```bash
# 终端1: 启动Ollama
ollama serve

# 终端2: 启动系统
source venv/bin/activate
python3 src/main.py --topic "你的主题" --turns 0
```

## 项目结构

```
├── config.yaml          # 系统配置（Agent人格、模型、记忆参数）
├── CLAUDE.md            # 开发指南
├── requirements.txt     # Python依赖
├── setup.sh             # 一键安装脚本
├── src/
│   ├── main.py          # 入口，对话循环编排
│   ├── agent.py         # Agent类（人格、上下文构建）
│   ├── memory.py        # 记忆系统（工作记忆+长期记忆）
│   ├── logger.py        # 日志系统（JSONL/MD/快照）
│   └── ollama_client.py # Ollama HTTP API封装
└── tests/               # 测试用例
```

## 涌现指标

| 指标 | 含义 |
|---|---|
| 新概念创造 | 原创概念命名频率 |
| 跨域类比 | 不同领域间的类比映射 |
| 推理步骤 | 结构化分步推理的深度 |
| 批判性质疑 | 挑战假设和追问证据 |
| 元认知反思 | 对自身思维的觉察与审视 |

---

### 👋 About @bigdaddyzc

- 👀 I’m interested in reinforcement learning, NLP, and human-AI interaction
- 🌱 I’m currently learning agentic systems and emergent capabilities
- 💞️ I’m looking to collaborate on AI/ML research
- 📫 How to reach me: [GitHub](https://github.com/bigdaddyzc)

<!---
bigdaddyzc/bigdaddyzc is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
>>>>>>> origin/main

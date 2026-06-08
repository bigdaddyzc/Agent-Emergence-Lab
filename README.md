# Agent Emergence Lab

双Agent自由对话系统，通过两个不同人格的AI Agent持续对话实现**能力涌现**（Emergent Capabilities），配备记忆体系统和实时Web展示界面。

## 概述

两个具有不同特长的AI Agent（理性分析型 **Nova** + 创意发散型 **Riven**）通过不间断自由对话，在互动中产生超出单个模型能力的涌现行为。系统完整记录对话日志，并通过Web界面实时展示。

### 核心特性

- **双Agent对话**: Nova (Qwen2.5-7B, 分析师) × Riven (Llama 3.2-3B, 创造者)
- **记忆系统**: 双层架构 — 工作记忆（短期）+ 长期记忆（知识提取与检索）
- **实时Web UI**: 浏览器端SSE实时推送，双列聊天布局
- **完整日志**: JSON-Lines（程序分析）+ Markdown（人工阅读）+ 记忆快照
- **涌现测量**: 自动追踪响应长度、交叉引用、新概念引入等指标
- **纯本地部署**: 通过Ollama运行，无需GPU，无需外部API

## 快速开始

### 1. 一键安装

```bash
bash setup.sh
```

脚本将自动：
- 安装Ollama
- 下载所需模型（qwen2.5:7b + llama3.2:3b）
- 创建Python虚拟环境并安装依赖

### 2. 手动安装

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

### 3. 运行

```bash
# 终端1: 启动Ollama
ollama serve

# 终端2: 启动系统
source venv/bin/activate
python3 src/main.py --topic "人工智能意识的可能性" --turns 10
```

打开浏览器访问 **http://localhost:5000** 实时观看对话。

## 项目结构

```
├── config.yaml          # 系统配置（Agent人格、模型、记忆参数）
├── requirements.txt     # Python依赖
├── setup.sh             # 一键安装脚本
├── README.md
├── src/
│   ├── main.py          # 入口，对话循环编排
│   ├── agent.py         # Agent类（人格、上下文构建）
│   ├── memory.py        # 记忆系统（工作记忆+长期记忆）
│   ├── logger.py        # 日志系统（JSONL/MD/快照）
│   ├── webui.py         # Flask Web服务器 + SSE推送
│   └── ollama_client.py # Ollama HTTP API封装
├── static/
│   └── index.html       # Web前端（嵌入式CSS/JS）
└── logs/
    ├── conversations/   # 对话记录
    └── memory/          # 记忆快照
```

## 命令行参数

```bash
python3 src/main.py [--topic TOPIC] [--turns N] [--port PORT] [--config FILE]

--topic   对话主题（默认: "the nature of consciousness"）
--turns   对话轮数（默认: 10）
--port    Web UI端口（默认: 5000）
--config  配置文件路径（默认: config.yaml）
--resume  从之前的session恢复
```

## 系统架构

```
                          +-----------+
                          | config.yaml|
                          +-----+-----+
                                |
        +----------+     +-----v----+     +----------+     +-----------+
        | Ollama   |<--->| main.py  |<--->| logger.py|     | Browser   |
        | :11434   |     | 编排器    |     +----------+     | (SSE)    |
        |          |     +----------+                      +-----^-----+
        | qwen2.5  |       |      |                               |
        | :4b      |    +--+    +--+                      +-------+---+
        |          |    |         |                       | webui.py  |
        | llama3.2 |  agent_a  agent_b                    | Flask/SSE |
        | :3b      |  (Nova)   (Riven)                    +-----------+
        +----------+    |         |
                     +--v---------v--+
                     |  memory.py   |
                     |  记忆系统     |
                     +--------------+
```

## Agent人格

| | Nova (分析师) | Riven (创造者) |
|---|---|---|
| 模型 | qwen2.5:7b | llama3.2:3b |
| 角色 | 逻辑推理、结构化思考 | 横向思维、类比联想 |
| Temperature | 0.7 | 0.9 |

## 记忆系统

- **工作记忆**: 最近10轮对话历史，注入Agent上下文窗口
- **长期记忆**: 周期性从对话中提取insight，通过关键词检索
- **检索**: 基于Jaccard相似度的关键词匹配，返回top-5相关记忆
- **合并**: 定期去重合并，删除琐碎条目

## 涌现评估指标

| 指标 | 含义 |
|---|---|
| 响应长度变化 | 正斜率表示讨论深化 |
| 交叉引用次数 | 对方观点被引用的频率 |
| 新概念引入率 | 关键词多样性的增长速度 |
| 主题持续深度 | 非相邻轮次的话题连贯性 |
| 记忆利用率 | 长期记忆被引用的比例 |

## 系统要求

- Linux x86_64 (推荐8GB+ RAM)
- Python 3.10+
- 无需GPU（CPU推理使用Q4量化模型）
- 约6GB磁盘空间（用于模型文件）

## License

MIT

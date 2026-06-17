# 4G CPU 本地模型策略

生成日期：2026-06-17  
运行约束：4GB 内存、CPU 推理、本地 Ollama、免费模型、可自动安装。

---

## 1. 结论

4G CPU 环境下，项目应优先使用 0.5B-1B 级模型。当前默认的 `qwen2.5:3b` + `llama3.2:3b` 对 4GB 内存并不友好，虽然可能能运行，但长对话、双模型切换、上下文注入和记忆提取会造成明显性能压力。

推荐默认低配组合：

```text
Nova  -> qwen2.5:0.5b
Riven -> llama3.2:1b
```

更保守组合：

```text
Nova  -> qwen2.5:0.5b
Riven -> smollm2:360m
```

所有新增模型必须通过自动安装和性能基准测试后，才能进入推荐配置。

---

## 2. 模型准入原则

硬性要求：

1. 必须免费、本地可运行。
2. 必须能通过 Ollama 自动安装，例如 `ollama pull qwen2.5:0.5b`。
3. 必须在 4GB CPU 环境完成性能测试。
4. 默认候选模型参数规模不超过 1.5B。
5. 默认 `context_window` 不超过 2048-3072。
6. 默认 `max_tokens` 不超过 192-256。
7. 未通过基准测试的模型不能写入默认配置。

---

## 3. 推荐候选模型

| 模型 | 安装命令 | 推荐用途 | 备注 |
|---|---|---|---|
| `qwen2.5:0.5b` | `ollama pull qwen2.5:0.5b` | 中文 Agent、记忆提取 | 低配首选 |
| `llama3.2:1b` | `ollama pull llama3.2:1b` | 对话 Agent | 质量/速度平衡候选 |
| `smollm2:360m` | `ollama pull smollm2:360m` | 极低配对话、链路测试 | 中文能力需实测 |
| `smollm2:135m` | `ollama pull smollm2:135m` | smoke test | 不建议用于严肃对话 |
| `gemma3:1b` | `ollama pull gemma3:1b` | 可选通用模型 | 需测中文稳定性 |
| `tinyllama` | `ollama pull tinyllama` | 速度优先实验 | 中文与推理质量谨慎 |
| `deepseek-r1:1.5b` | `ollama pull deepseek-r1:1.5b` | 轻量推理实验 | 可能输出偏长，需严格限 token |

不推荐作为 4G 默认：

- `qwen2.5:3b`
- `llama3.2:3b`
- `phi3`
- `gemma:2b`
- 任何 4B、7B 或更大模型

---

## 4. 推荐低配配置

建议新增：

`configs/low_resource_4g.yaml`

核心参数：

```yaml
agents:
  agent_a:
    name: "Nova"
    model: "qwen2.5:0.5b"
    temperature: 0.65
    max_tokens: 192
    context_window: 2048

  agent_b:
    name: "Riven"
    model: "llama3.2:1b"
    temperature: 0.75
    max_tokens: 192
    context_window: 2048

memory:
  short_term_window: 6
  extraction_interval: 3
  consolidation_interval: 3
  max_memories_per_prompt: 3
  compression_threshold: 900
  turns_kept_verbatim: 2

ollama:
  timeout: 600
  keep_alive: "5m"
```

运行：

```bash
python src/main.py --config configs/low_resource_4g.yaml --topic "记忆的本质" --turns 5
```

---

## 5. 自动安装方案

新增脚本：

`scripts/install_models.py`

功能：

1. 读取 config 中的模型名。
2. 调用 `ollama list` 检查是否已安装。
3. 对缺失模型执行 `ollama pull <model>`。
4. 安装后再次验证。

命令：

```bash
python scripts/install_models.py --config configs/low_resource_4g.yaml
```

或：

```bash
python scripts/install_models.py --models qwen2.5:0.5b llama3.2:1b
```

验收：

- Ollama 未运行时提示 `ollama serve`；
- 已安装模型不重复下载；
- 拉取失败时退出码非 0；
- 成功后输出模型清单。

---

## 6. 性能基准方案

新增脚本：

`scripts/benchmark_models.py`

命令：

```bash
python scripts/benchmark_models.py --models qwen2.5:0.5b llama3.2:1b --rounds 3
```

测试任务：

1. 中文解释：`用三句话解释什么是长期记忆。`
2. 类比任务：`把记忆系统类比成图书馆，并指出类比哪里不成立。`
3. 对话接续：`回应对方观点，并提出一个具体质疑。`
4. 结构输出：`按 类型|新鲜度|内容 输出两条观察。`

记录指标：

- 总耗时；
- tokens/sec；
- prompt tokens；
- generated tokens；
- 是否超时；
- 输出是否可解析；
- 简短质量评级；
- 是否适合 Agent、memory extraction 或 judge。

最低准入：

| 指标 | 要求 |
|---|---|
| 128 token 生成 | 不超过 90 秒 |
| tokens/sec | 不低于 1.5 |
| 连续 3 轮 | 不崩溃、不超时 |
| 结构输出 | 至少 2/3 可解析 |
| 系统响应 | 不明显卡死 |

模型等级：

| 等级 | 说明 | 用法 |
|---|---|---|
| A | 速度和质量都可接受 | 推荐配置 |
| B | 可运行但有明显短板 | 可选实验 |
| C | 只能跑通链路 | smoke test |
| D | 超时、崩溃或输出不可用 | 移除候选 |

---

## 7. 新增模型流程

1. 在候选列表中添加模型。
2. 用 `scripts/install_models.py` 自动安装。
3. 用 `scripts/benchmark_models.py` 测试。
4. 将结果写入 `docs/model_benchmarks.md`。
5. A/B 级可进入推荐配置。
6. C 级只用于 smoke test。
7. D 级不再推荐。

---

## 8. 对当前项目的直接建议

短期建议：

1. 新建 `configs/low_resource_4g.yaml`。
2. 新建 `scripts/install_models.py`。
3. 新建 `scripts/benchmark_models.py`。
4. 将 `setup.sh` 改为读取配置并安装对应模型，而不是硬编码拉取 3B 模型。
5. README 中增加“4G CPU 推荐运行方式”。

默认命令建议：

```bash
python scripts/install_models.py --config configs/low_resource_4g.yaml
python scripts/benchmark_models.py --config configs/low_resource_4g.yaml
python src/main.py --config configs/low_resource_4g.yaml --turns 5
```

这样既满足免费本地模型要求，也避免 4GB CPU 环境被 3B 双模型压垮。

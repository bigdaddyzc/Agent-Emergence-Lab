# Agent Emergence Lab 优化方案

生成日期：2026-06-17  
适用项目：`D:\claude-workspace\agent-emergence-lab`  
关联报告：`docs/project_evaluation_report.md`

---

## 1. 优化总目标

本项目下一阶段不应只追求“更多功能”，而应优先把它从一个有趣的多 Agent 对话实验，升级为一个更可信、可复现、可分析、可持续迭代的本地多 Agent 研究平台。

总体优化目标分为四层：

1. 提升可信度：让“涌现信号”不只是正则命中，而能被对照实验、评分器和报告支持。
2. 提升可复现性：让每次实验的配置、模型、随机性、日志和结果可以复盘比较。
3. 提升工程可维护性：拆分过重主循环，建立清晰模块边界。
4. 提升使用体验：自动生成会话分析报告，让用户能看懂一场对话到底发生了什么。

核心策略：

先建立评估闭环，再增强模型能力；先让结果可信，再让系统变复杂。

---

## 2. 优先级总览

| 优先级 | 优化方向 | 目标 | 预期收益 |
|---|---|---|---|
| P0 | 文档与配置修复 | 修复乱码、不一致、配置不可校验问题 | 提升可信度和可用性 |
| P1 | 会话分析与报告 | 从 JSONL 自动生成统计报告 | 让实验结果可读、可比较 |
| P1 | 评估模块抽离 | 将涌现信号检测从 Agent 中拆出 | 降低耦合，便于升级指标 |
| P1 | 对照实验框架 | 支持 single/dual/no-memory/full-system 对照 | 支撑研究结论 |
| P2 | 记忆系统升级 | 引入 embedding 检索和记忆质量控制 | 提升长对话连续性 |
| P2 | E2E 测试与 Mock 模型 | 测完整对话链路 | 降低主循环回归风险 |
| P3 | 可视化/报告导出 | 趋势图、概念生命周期、精选片段 | 提升展示与传播价值 |
| P3 | 多 Agent 扩展 | 支持第三 Agent 或 Judge Agent | 增强实验能力 |

---

## 2.1 4G CPU 环境模型准入策略

运行约束：目标环境为 4GB 内存、CPU 推理、本地 Ollama、免费模型。任何新增第三方本地模型都必须满足以下准入规则。

硬性原则：

1. 默认不新增 3B 以上模型。
2. 默认不同时常驻两个 3B 模型。
3. 新模型必须能通过 `ollama pull <model>` 自动安装。
4. 新模型必须先经过本机性能基准测试，再允许写入默认配置。
5. 新模型必须是免费可本地运行模型，并在文档中记录来源、license/使用限制链接。
6. 对 4G CPU，默认上下文窗口不超过 2048-3072，默认回复长度不超过 192-256 tokens。
7. 若模型基准低于最低阈值，只能作为可选实验模型，不能作为默认模型。

推荐候选模型池：

| 用途 | 候选模型 | Ollama 安装命令 | 适配判断 |
|---|---|---|---|
| 中文/多语言轻量 Agent | `qwen2.5:0.5b` | `ollama pull qwen2.5:0.5b` | 优先候选，适合作为 4G CPU 默认轻量模型 |
| 英文/多语言轻量 Agent | `llama3.2:1b` | `ollama pull llama3.2:1b` | 可选候选，官方 1B 小模型 |
| 极轻量实验模型 | `smollm2:360m` | `ollama pull smollm2:360m` | 适合压力很大的机器，但中文能力需实测 |
| 极轻量/速度优先 | `smollm2:135m` | `ollama pull smollm2:135m` | 仅建议用于管线测试，不建议用于严肃对话质量评估 |
| 小型通用模型 | `gemma3:1b` | `ollama pull gemma3:1b` | 可选候选，需测试中文稳定性和内存占用 |
| 轻量对话模型 | `tinyllama` | `ollama pull tinyllama` | 可选候选，速度友好，但中文和推理质量需谨慎评估 |
| 轻量推理模型 | `deepseek-r1:1.5b` | `ollama pull deepseek-r1:1.5b` | 可选实验，不建议默认；推理输出可能偏长，需限制 tokens |

不推荐作为 4G CPU 默认模型：

- `qwen2.5:3b`
- `llama3.2:3b`
- `phi3`
- `gemma:2b`
- 任何 4B、7B 或更大模型

原因：模型文件本身可能能下载，但双 Agent 长对话、上下文、Ollama 运行时和系统内存叠加后，4GB 环境容易出现明显换页、极慢、超时或直接失败。

建议默认低配组合：

```yaml
agents:
  agent_a:
    model: "qwen2.5:0.5b"
    max_tokens: 192
    context_window: 2048
  agent_b:
    model: "llama3.2:1b"
    max_tokens: 192
    context_window: 2048

ollama:
  keep_alive: "5m"
  timeout: 600

memory:
  short_term_window: 6
  max_memories_per_prompt: 3
  compression_threshold: 900
  turns_kept_verbatim: 2
```

更稳妥的 4G 极低配组合：

```yaml
agents:
  agent_a:
    model: "qwen2.5:0.5b"
  agent_b:
    model: "smollm2:360m"
```

注意：极低配组合适合验证系统链路和低成本观察，不适合用来支撑强研究结论。

### 2.1.1 自动安装要求

建议新增脚本：

`scripts/install_models.py`

功能：

1. 读取 `config.yaml` 中的 `agents.*.model`。
2. 调用 `ollama list` 检查模型是否已安装。
3. 未安装则自动执行 `ollama pull <model>`。
4. 安装后再次验证。
5. 输出安装报告。

CLI 示例：

```bash
python scripts/install_models.py --config config.yaml
```

建议同时支持手动指定：

```bash
python scripts/install_models.py --models qwen2.5:0.5b llama3.2:1b
```

验收标准：

- Ollama 未运行时给出明确提示；
- 模型不存在或拉取失败时退出码非 0；
- 成功安装后 `ollama list` 能看到对应模型；
- 不重复下载已存在模型。

### 2.1.2 性能基准要求

建议新增脚本：

`scripts/benchmark_models.py`

基准命令：

```bash
python scripts/benchmark_models.py --models qwen2.5:0.5b llama3.2:1b --rounds 3
```

每个模型测试：

- 冷启动耗时；
- 首次生成总耗时；
- tokens/sec；
- 生成 tokens；
- prompt tokens；
- 是否超时；
- 进程峰值内存，能测则记录；
- 中文短回答质量样例；
- 是否适合 Agent A / Agent B / memory extraction / judge。

建议测试 prompt：

1. 中文短问答：`用三句话解释什么是长期记忆。`
2. 类比任务：`把记忆系统类比成一个图书馆，指出类比哪里不成立。`
3. 对话接续：`回应对方观点，并提出一个具体质疑。`
4. 结构输出：`按 类型|新鲜度|内容 输出两条观察。`

最低准入阈值：

| 指标 | 4G CPU 最低要求 |
|---|---|
| 单次 128 token 生成 | 不超过 90 秒 |
| tokens/sec | 不低于 1.5 |
| 连续 3 轮 | 不崩溃、不超时 |
| 输出可解析性 | 结构化任务至少 2/3 成功 |
| 内存表现 | 不导致系统明显失去响应 |

通过后的模型分级：

| 等级 | 含义 | 用法 |
|---|---|---|
| A | 速度和质量都可接受 | 可进入推荐配置 |
| B | 可运行但慢或质量一般 | 可选实验模型 |
| C | 只能跑通链路 | 仅用于 smoke test |
| D | 超时/崩溃/输出不可用 | 不纳入模型池 |

### 2.1.3 配置文件建议

建议新增低配配置：

`configs/low_resource_4g.yaml`

用途：

- 给 4G CPU 用户提供安全默认值；
- 不影响当前 `config.yaml`；
- 便于对比默认配置和低配配置。

运行方式：

```bash
python src/main.py --config configs/low_resource_4g.yaml --topic "记忆的本质" --turns 5
```

### 2.1.4 模型新增流程

任何新增本地模型必须走以下流程：

1. 在 `docs/models.md` 记录候选模型、Ollama 页面、license/使用限制、参数规模和预期用途。
2. 使用 `scripts/install_models.py` 自动安装。
3. 使用 `scripts/benchmark_models.py` 在 4G CPU 机器上测试。
4. 将测试结果写入 `docs/model_benchmarks.md`。
5. 只有 A/B 级模型可以写入推荐配置。
6. C 级模型只能用于 smoke test。
7. D 级模型从候选池移除。

---

## 3. 第一阶段：基础可信度修复

周期建议：1-2 天  
目标：让项目文档、配置和基础运行状态更可信。

### 3.1 修复 README 与 config 编码问题

当前 README 和 config 在终端读取时出现大量乱码，影响开源传播和使用信任。

建议动作：

1. 检查文件实际编码，统一保存为 UTF-8。
2. 修复 README 中损坏的中文段落、表格和示例输出。
3. 删除或更新 README 中与当前现实不一致的内容，例如已移除 Web UI 但 roadmap 或结构描述仍容易误导的部分。
4. 保留英文简介，但中文说明应完整可读。

验收标准：

- `Get-Content -Encoding UTF8 README.md` 中文正常显示；
- `Get-Content -Encoding UTF8 config.yaml` 注释正常显示；
- README 中不再出现已删除功能的使用说明。

### 3.2 增加配置校验

当前 `config.yaml` 直接被 `yaml.safe_load` 读取，缺少 schema 校验。配置错误可能在运行中才暴露。

建议新增：

`src/config_validator.py`

校验内容：

- 必需字段：`agents`、`memory`、`emergence`、`logging`、`ollama`；
- Agent 必需字段：`name`、`model`、`system_prompt_template`、`temperature`、`max_tokens`、`context_window`；
- 数值范围：temperature、max_tokens、context_window、interval；
- Feishu webhook placeholder 检测；
- emergence patterns 必须是字符串列表。

验收标准：

- 配置缺字段时给出清晰错误；
- 单元测试覆盖合法配置和错误配置；
- `main.py` 启动时先校验 config。

### 3.3 清理历史计划文档

`docs/superpowers/plans/2026-05-29-feishu-integration-and-topic-fix.md` 中包含过期路径和未完成 checklist，容易让维护者困惑。

建议动作：

- 移到 `docs/archive/`；
- 或在顶部标注“历史实现计划，当前代码已部分偏离”；
- 将仍有价值的内容合并进正式 docs。

---

## 4. 第二阶段：会话分析与自动报告

周期建议：2-4 天  
目标：把“对话日志”升级成“实验结果”。

### 4.1 新增会话分析脚本

建议新增：

`scripts/analyze_session.py`

输入：

```bash
python scripts/analyze_session.py logs/conversations/<session>.jsonl
```

输出指标：

- 总轮数；
- 总 tokens；
- 每个 Agent 发言次数；
- 每个 Agent 平均回复长度；
- cross references 总数与均值；
- novel concepts 总数；
- analogies 总数；
- metacognition 总数；
- reasoning steps 总数；
- critical challenges 总数；
- memory extraction 次数；
- topic switch 次数；
- 每 5 轮指标趋势；
- 高信号轮次 Top 10。

验收标准：

- 能读取现有 JSONL；
- 输出终端摘要；
- 可选导出 JSON。

### 4.2 新增自动实验报告生成器

建议新增：

`scripts/generate_session_report.py`

输入：

```bash
python scripts/generate_session_report.py logs/conversations/<session>.jsonl
```

输出：

`logs/reports/<session>_report.md`

报告结构：

1. 会话基本信息；
2. Agent 与模型配置；
3. 指标总览；
4. 指标趋势；
5. 关键对话片段；
6. 新概念列表；
7. 质疑与回应片段；
8. 记忆系统表现；
9. 初步结论；
10. 风险提示。

验收标准：

- 每次会话可自动生成 Markdown 报告；
- 报告中包含至少 3 个高信号片段；
- 报告中明确标注“启发式指标，不等同于严格证明”。

### 4.3 增加日志 schema 版本

建议在 session_start 中加入：

```json
{
  "schema_version": "1.0"
}
```

原因：

- 便于后续分析脚本兼容旧日志；
- 避免日志字段变化导致脚本崩溃。

---

## 5. 第三阶段：评估体系升级

周期建议：4-7 天  
目标：让“涌现”从口号变成可比较的实验指标。

### 5.1 抽离 Evaluator 模块

当前 `extract_emergence_signals()` 放在 `Agent` 内，不利于扩展。

建议新增：

`src/evaluator.py`

包含：

- `EmergenceSignalEvaluator`
- `SignalResult`
- `ConceptTracker`
- `ChallengeTracker`
- `RepetitionDetector`

第一版先迁移现有正则逻辑，不改变行为。

验收标准：

- `agent.py` 不再负责评估；
- 现有测试迁移或新增到 `tests/test_evaluator.py`；
- 所有测试通过。

### 5.2 建立概念生命周期追踪

当前系统只统计出现了多少 `【新概念】`，但不知道概念后续是否被使用。

建议追踪：

- concept_name；
- first_mentioned_turn；
- source_agent；
- initial_definition；
- later_references；
- challenged_count；
- refined_count；
- abandoned_or_active；
- last_seen_turn。

指标：

- 概念存活率；
- 概念复用率；
- 被质疑后仍保留的概念数；
- 单次出现后消失的概念数。

这比简单统计“新概念数量”更能反映对话是否真的积累了知识。

### 5.3 增加重复率与漂移率检测

建议指标：

- self repetition：Agent 是否重复自己上一轮观点；
- partner repetition：是否只复述对方；
- topic drift：当前回复关键词与当前 topic 的重合/语义距离；
- stagnation：连续多轮没有新信号。

用途：

- 识别对话是否陷入空转；
- 支持自动切换话题或触发反思 prompt；
- 让系统不仅统计“好信号”，也统计“坏信号”。

### 5.4 引入 Judge 评分

第一版可以使用本地模型作为裁判，后续再支持人工标注。

评分维度：

| 维度 | 说明 |
|---|---|
| Novelty | 是否提出新视角 |
| Coherence | 是否与上下文连贯 |
| Grounding | 是否回应对方具体观点 |
| Reasoning | 推理是否有因果结构 |
| Challenge Quality | 质疑是否具体有效 |
| Synthesis | 是否综合多轮信息 |

输出格式：

```text
novelty|1-5|理由
coherence|1-5|理由
reasoning|1-5|理由
...
```

注意：

Judge 评分不能代替严格研究结论，但比纯正则更有解释力。

---

## 6. 第四阶段：对照实验框架

周期建议：5-10 天  
目标：让项目可以回答“这个结构到底有没有提升效果”。

### 6.1 新增 ExperimentRunner

建议新增：

`src/experiment.py`

支持实验模式：

| 模式 | 描述 |
|---|---|
| `single_agent` | 单 Agent 对同一话题连续思考 |
| `dual_basic` | 双 Agent，无长期记忆，无深度推进 |
| `dual_memory` | 双 Agent，有记忆，无深度推进 |
| `dual_depth` | 双 Agent，无长期记忆，有深度推进 |
| `full` | 当前完整系统 |

CLI 示例：

```bash
python src/experiment.py --topic "记忆的本质" --turns 20 --mode full --runs 5
```

### 6.2 固定随机性与配置快照

建议每次实验记录：

- experiment_id；
- run_id；
- mode；
- topic；
- seed；
- config hash；
- model names；
- Ollama host；
- start/end time；
- git commit hash；
- Python version；
- platform info。

这样未来才能比较不同版本的实验结果。

### 6.3 实验对比报告

建议新增：

`scripts/compare_experiments.py`

比较指标：

- full vs dual_basic；
- memory on/off；
- depth on/off；
- agent pair A/B；
- model pair A/B。

输出：

- 每组均值；
- 标准差；
- 趋势图数据；
- 高信号片段对比；
- 简短结论。

---

## 7. 第五阶段：记忆系统升级

周期建议：4-8 天  
目标：提升长期对话的真实连续性，减少记忆污染。

### 7.1 增加 embedding 检索

`OllamaClient` 已有 `embed()` 方法，但当前 memory 检索主要是关键词。

建议：

- 为每条 `MemoryEntry` 增加 `embedding` 字段；
- 写入 memory 时生成 embedding；
- query 时生成 query embedding；
- 使用 cosine similarity；
- 结合关键词得分、topic bonus、recency、novelty；
- config 支持 `retrieval_method: keyword | embedding | hybrid`。

推荐第一版 scoring：

```text
final_score =
  0.55 * embedding_similarity
  + 0.20 * keyword_overlap
  + 0.10 * topic_score
  + 0.10 * recency_score
  + 0.05 * novelty_score
```

### 7.2 增加记忆质量过滤

当前模型提取 memory 后直接写入长期记忆。建议新增过滤：

- 过短内容拒绝；
- 纯复述拒绝；
- 与已有 memory 高重复拒绝；
- 缺少名词/关键词拒绝；
- novelty 低且无 topic relevance 的 memory 降权。

### 7.3 增加记忆衰减与归档

长期运行时 memory 会不断增长。建议：

- ref_count 长期为 0 的 memory 降权；
- 低质量 memory 进入 archive；
- synthetic memory 与原始 memory 分层管理；
- 每 N 轮生成 memory health metrics。

---

## 8. 第六阶段：工程结构优化

周期建议：3-6 天  
目标：降低 `main.py` 复杂度，便于后续扩展。

### 8.1 拆分主循环

建议目标结构：

```text
src/
  main.py                  # CLI only
  orchestrator.py          # 对话状态机
  topic_manager.py         # 话题生成与切换
  evaluator.py             # 指标检测
  experiment.py            # 批量实验
  report.py                # 报告生成核心逻辑
  agent.py
  memory.py
  logger.py
  ollama_client.py
  feishu.py
  config_validator.py
```

第一步不要大重构。建议按以下顺序：

1. 抽 `TopicManager`；
2. 抽 `Evaluator`；
3. 抽 `MetricsTracker`；
4. 最后再抽 `ConversationOrchestrator`。

### 8.2 增加类型与接口边界

建议增加 dataclass：

- `ConversationTurn`
- `AgentResponse`
- `RetrievalResult`
- `SignalResult`
- `ExperimentConfig`
- `SessionMetrics`

收益：

- 减少 dict 字段拼写错误；
- 让日志和分析脚本更稳定；
- 方便 IDE 和测试。

---

## 9. 第七阶段：测试体系增强

周期建议：2-5 天  
目标：覆盖核心链路，不依赖真实 Ollama。

### 9.1 新增 FakeOllamaClient

建议在 tests 中提供：

`tests/fakes.py`

能力：

- 固定返回 chat 内容；
- 固定返回 generate 内容；
- 模拟异常；
- 模拟 timeout；
- 模拟不同 token 数。

### 9.2 新增 E2E 测试

覆盖：

- 一轮完整双 Agent 对话；
- 记忆提取；
- 指标写入；
- topic switch；
- compression；
- session end；
- memory snapshot。

### 9.3 新增 Logger 测试

覆盖：

- JSONL 第一行 session_start；
- turn log schema；
- metric log schema；
- Markdown 文件生成；
- memory snapshot 可被 json.load。

---

## 10. 第八阶段：体验与展示优化

周期建议：可选，5-10 天  
目标：让项目更适合演示和传播。

### 10.1 终端体验优化

建议：

- 增加 `--quiet`；
- 增加 `--show-metrics`；
- 增加 `--no-feishu`；
- 增加 `--save-report`；
- 结束时打印报告路径；
- 出错时给出修复建议。

### 10.2 可选 TUI 或轻量 Web 回放

虽然 Web UI 已移除，但后续可以只做“日志回放/报告查看”，而不是实时复杂 UI。

推荐优先级：

1. 静态 HTML report；
2. Markdown report；
3. TUI dashboard；
4. 实时 Web UI。

### 10.3 README 传播优化

建议 README 聚焦三个卖点：

1. 本地运行的双小模型自主对话；
2. 记忆、深度推进与涌现信号追踪；
3. 自动实验报告和对照评估。

避免过强主张：

- 不说“证明涌现”；
- 改说“观察和评估多 Agent 互动中的涌现式行为信号”。

---

## 11. 推荐实施顺序

### 第 1 周

1. 修复 README/config 编码。
2. 增加 config validator。
3. 新增 session analyzer。
4. 新增 session report generator。
5. 抽离 evaluator 第一版。

预期结果：

项目从“能跑”升级为“能自动解释一场实验”。

### 第 2 周

1. 增加 concept lifecycle tracking。
2. 增加 repetition/drift 指标。
3. 增加 FakeOllamaClient。
4. 增加主流程 E2E 测试。
5. 增加 baseline experiment runner 第一版。

预期结果：

项目从“观察工具”升级为“可对照实验工具”。

### 第 3 周

1. 增加 embedding/hybrid memory retrieval。
2. 增加 memory quality filter。
3. 增加 experiment comparison report。
4. 拆分 TopicManager 和 MetricsTracker。

预期结果：

系统长对话质量和工程可维护性明显提升。

### 第 4 周

1. 增加 Judge scoring。
2. 增加静态 HTML/Markdown 高级报告。
3. README 重写。
4. 准备 demo 数据和示例报告。

预期结果：

项目具备对外展示、开源传播和进一步研究迭代基础。

---

## 12. 成功指标

优化完成后，建议用以下指标判断是否成功：

| 指标 | 目标 |
|---|---|
| 测试通过率 | 100% |
| E2E 测试 | 至少覆盖 1 个完整 session |
| 会话报告生成 | 100% 支持 JSONL -> Markdown |
| 对照实验 | 至少支持 3 种模式 |
| 配置校验 | 错误配置能在启动前失败 |
| 指标解释性 | 每个核心指标有定义和局限说明 |
| README 可读性 | 中英文无乱码，无过期功能 |
| 主循环复杂度 | `main.py` 行数明显下降，核心逻辑迁移到模块 |

---

## 13. 最小可行优化包

如果只做最小投入，推荐先做这 5 件事：

1. 修复 README/config 中文乱码。
2. 新增 `scripts/analyze_session.py`。
3. 新增 `scripts/generate_session_report.py`。
4. 抽离 `src/evaluator.py`。
5. 新增 FakeOllamaClient 和一个 E2E 测试。

这 5 项完成后，项目的可信度、可维护性和可展示性会立刻提升一个层级。

---

## 14. 结论

Agent Emergence Lab 的下一步优化重点应是“评估系统化”。当前项目已经能制造并记录有趣的多 Agent 对话现象，但要让这些现象变成可信结论，需要补齐分析、对照、评分、复现和测试。

推荐路线是：

先修复文档与配置，再自动生成会话报告；先抽离评估模块，再建立对照实验；先提升记忆质量，再考虑 UI 和多 Agent 扩展。

这样项目会从一个好玩的实验 demo，逐步成长为一个真正有研究说服力的本地多 Agent 交互实验平台。

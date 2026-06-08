# Feishu Integration & Topic Switching Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the broken topic switching mechanism, remove the Flask-based Web UI, and add Feishu bot integration so two bots push agent messages into a shared group chat in real time.

**Architecture:** Topic switching switches from LLM-generated topics (which always returns "consciousness") to a simple round-robin rotation through `config.yaml`'s `fallback_topics` list. The Web UI (Flask + SSE + static HTML) is deleted entirely. A new `feishu.py` module sends each agent's response as a text message via its own Feishu custom bot webhook, both posted into the same group chat.

**Tech Stack:** Python, requests (already in deps), Feishu custom bot webhooks (no SDK needed)

---

### Task 1: Fix topic switching in config.yaml and main.py

**Root cause:** `topic.max_turns_per_topic: 10` is too high (most sessions never reach it), and the LLM-based topic generator keeps returning consciousness-related topics because the model is saturated in the consciousness discussion. Fix: lower threshold to 5, replace LLM generation with fallback topic rotation.

**Files:**
- Modify: `config.yaml:8` — change `max_turns_per_topic` from 10 to 5
- Modify: `src/main.py:296-343` — replace LLM-based topic generation with fallback rotation

- [ ] **Step 1: Reduce topic switch threshold in config.yaml**

```yaml
# config.yaml line 8
topic:
  max_turns_per_topic: 5    # was 10
```

- [ ] **Step 2: Replace LLM topic generation with fallback rotation in main.py**

Replace the block starting from `# Generate new topic using the analyst model` through to the fallback assignment (`if fallback_topics else "科技与人文的冲突与融合"`), and the block's aftermath (`new_topic = new_topic.strip('"\'').strip("」").strip("「")` etc.).

Old code (lines ~303-323):

```python
# Generate new topic using the analyst model
topic_gen_prompt = (
    f"你刚才和 {agent_b.name} 讨论了「{current_topic}」。"
    f"现在需要换一个新话题。请根据刚才的讨论，提出一个相关但全新的讨论方向。"
    f"新话题要和之前的讨论有联系，但视角要不同。"
    f"只输出新话题的名称，不要解释，不要多余的字。"
)
try:
    new_topic_resp, _ = agent_a.think(
        f"{agent_a._system_prompt}\n\n额外指令：{topic_gen_prompt}"
    )
    new_topic = new_topic_resp.strip().strip('"').strip("'").strip("」").strip("「")
    if len(new_topic) > 100:
        new_topic = new_topic[:100]
except Exception:
    new_topic = fallback_topics[topic_switch_count % len(fallback_topics)] if fallback_topics else "科技与人文的冲突与融合"

new_topic = new_topic.strip('"\'').strip("」").strip("「")
if not new_topic or len(new_topic) < 2:
    new_topic = fallback_topics[topic_switch_count % len(fallback_topics)] if fallback_topics else "科技与人文的冲突与融合"
```

Replace with:

```python
# Rotate through fallback topics (more reliable than LLM-generated topics)
if fallback_topics:
    new_topic = fallback_topics[topic_switch_count % len(fallback_topics)]
else:
    new_topic = "科技与人文的冲突与融合"
```

- [ ] **Step 3: Update the topic switch print/log to log the actual new topic**

The print statement on line ~324 already prints the topic. Also log the topic value as a metric:

```python
logger_inst.log_metric("new_topic", topic_switch_count, turn, extra={"topic": current_topic})
```

But since `MetricLog` doesn't support extra fields, add the new topic to the `"topic_switch"` event metadata in a future task. For now, the print is sufficient:

```python
print(f"\n  📌 话题切换 ({topic_switch_count}): 「{current_topic}」→「{new_topic}」")
```

This line already works fine unmodified.

---

### Task 2: Remove Flask Web UI

**Files:**
- Delete: `src/webui.py`
- Delete: `static/index.html`
- Delete: `static/` directory (if empty after index.html removal)
- Modify: `src/main.py` — remove all WebUI references, `--web` flag, `--port` flag
- Modify: `requirements.txt` — remove flask

- [ ] **Step 1: Delete webui.py**

No code needed — just delete the file.

```bash
rm /workspace/programming-language-demo/src/webui.py
```

- [ ] **Step 2: Delete static/index.html (and static dir)**

```bash
rm /workspace/programming-language-demo/static/index.html
rmdir /workspace/programming-language-demo/static 2>/dev/null || true
```

- [ ] **Step 3: Remove Flask dependency from requirements.txt**

```diff
- flask>=3.0.0
```

- [ ] **Step 4: Remove all WebUI imports and references from main.py**

Remove these from main.py:

1. Remove the `--web` and `--port` argument parser arguments (lines ~386-391)
2. Remove the entire `if args.web:` block (lines ~413-465) and replace with simpler console-only fallback
3. Remove the `from src.webui import WebUI` import inside the web block
4. Remove all `webui.push_event(...)` and `webui.update_status(...)` calls throughout `run_conversation`

After removal, the main entry point simplifies to running the conversation directly:

```python
def main():
    parser = argparse.ArgumentParser(description="Agent Emergence Lab")
    parser.add_argument("--topic", default="the nature of consciousness",
                        help="Conversation topic")
    parser.add_argument("--turns", type=int, default=10,
                        help="Number of conversation turns")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)

    ollama_client = ...  # unchanged
    verify_ollama(ollama_client, config)

    agent_a = Agent(...)  # unchanged
    agent_b = Agent(...)  # unchanged
    memory = MemorySystem(config, ollama_client)
    logger_inst = Logger(config)

    feishu_a, feishu_b = _init_feishu(config)  # new, see Task 3

    try:
        run_conversation(config, agent_a, agent_b, memory, logger_inst,
                        args.topic, args.turns, feishu_a=feishu_a, feishu_b=feishu_b)
    except KeyboardInterrupt:
        _stop_event.set()
        print("\n\n  已中断。部分日志已保存。")
        sys.exit(0)
```

Remove all `webui.*` references inside `run_conversation()` function. The function signature needs a `feishu_a=None, feishu_b=None` parameter instead.

- [ ] **Step 5: Remove the `signal` and `threading` imports if no longer needed**

Check if `signal` and `threading` are still used after removing WebUI code. `signal` is not needed since we handle KeyboardInterrupt directly. Remove:

```python
import signal  # can remove
import threading  # can remove
```

Also remove the `_stop_event` usage via signal handlers (the global `_stop_event` and `sighandler` function).

Keep `_stop_event` and `threading.Event` since `run_conversation` checks `_stop_event.is_set()` for clean shutdown.

Actually, keep `threading` and `_stop_event` since they're used for the stop mechanism. Remove `signal`.

---

### Task 3: Create Feishu bot module

**Files:**
- Create: `src/feishu.py`

This module provides a simple client that POSTs text messages to a Feishu custom bot webhook.

- [ ] **Step 1: Create feishu.py**

```python
"""Feishu custom bot webhook client for sending agent messages to a group chat."""

import logging

import requests

logger = logging.getLogger(__name__)


class FeishuBot:
    """Sends messages to a Feishu group via a custom bot webhook."""

    def __init__(self, webhook_url: str, agent_name: str):
        self.webhook_url = webhook_url
        self.agent_name = agent_name

    def send_message(self, text: str) -> bool:
        """Send a plain text message. Returns True on success."""
        payload = {
            "msg_type": "text",
            "content": {
                "text": text,
            },
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != 0:
                logger.warning("Feishu API error (code=%s): %s",
                               body.get("code"), body.get("msg"))
                return False
            return True
        except requests.RequestException as e:
            logger.error("Feishu send failed for %s: %s", self.agent_name, e)
            return False
```

- [ ] **Step 2: Verify the module loads correctly**

```bash
cd /workspace/programming-language-demo && source venv/bin/activate && python3 -c "from src.feishu import FeishuBot; print('OK')"
```

---

### Task 4: Wire Feishu into main.py

**Files:**
- Modify: `src/main.py` — add feishu init, add webhook calls in the turn loop

- [ ] **Step 1: Add feishu initialization helper in main.py**

Add after the `verify_ollama` function:

```python
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

    from src.feishu import FeishuBot
    return (
        FeishuBot(webhook_a, config["agents"]["agent_a"]["name"]),
        FeishuBot(webhook_b, config["agents"]["agent_b"]["name"]),
    )
```

- [ ] **Step 2: Update run_conversation signature to accept feishu bots**

```python
def run_conversation(config: dict, agent_a: Agent, agent_b: Agent,
                     memory: MemorySystem, logger_inst: Logger,
                     topic: str, max_turns: int,
                     feishu_a=None, feishu_b=None) -> None:
```

- [ ] **Step 3: Add Feishu send calls after each agent turn**

After agent_a speaks (after line ~205 `if webui: webui.push_event(...)` block), add:

```python
if feishu_a:
    feishu_a.send_message(
        f"{agent_a.name}（{agent_a.role}）· 第 {turn + 1} 轮\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{response_a}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ {stats_a.tokens_generated} tokens · {stats_a.tokens_per_second} tok/s · {stats_a.total_duration_ms}ms"
    )
```

Similarly after agent_b speaks, send via `feishu_b`.

- [ ] **Step 4: Add Feishu notification on topic switch**

After topic switch (around line ~324 after the print), add:

```python
if feishu_a and feishu_b:
    msg = f"📌 话题切换：\n「{current_topic}」→「{new_topic}」"
    feishu_a.send_message(msg)
    feishu_b.send_message(msg)
```

- [ ] **Step 5: Pass feishu bots from main() into run_conversation()**

```python
feishu_a, feishu_b = init_feishu(config)
run_conversation(config, agent_a, agent_b, memory, logger_inst,
                 args.topic, args.turns, feishu_a=feishu_a, feishu_b=feishu_b)
```

---

### Task 5: Update config.yaml with Feishu settings

**Files:**
- Modify: `config.yaml` — add `feishu` section at bottom

- [ ] **Step 1: Append feishu config block**

```yaml
feishu:
  enabled: false
  agent_a_webhook: ""
  agent_b_webhook: ""
```

Disabled by default so the system still works in console-only mode without configuration.

---

### Task 6: Verify the system runs in console mode

- [ ] **Step 1: Install updated dependencies**

```bash
cd /workspace/programming-language-demo && source venv/bin/activate && pip install -r requirements.txt
```

- [ ] **Step 2: Dry-run import check**

```bash
cd /workspace/programming-language-demo && source venv/bin/activate && python3 -c "
from src.main import main
from src.agent import Agent, AgentConfig
from src.memory import MemorySystem
from src.logger import Logger
from src.ollama_client import OllamaClient
print('All imports OK')
"
```

- [ ] **Step 3: Quick smoke test (requires Ollama running)**

```bash
cd /workspace/programming-language-demo && source venv/bin/activate && python3 src/main.py --topic "测试" --turns 1
```

Verify the terminal output shows 1 turn of conversation and no Flask/WebUI errors.

---

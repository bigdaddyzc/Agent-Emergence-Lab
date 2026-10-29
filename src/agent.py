"""Agent persona and inference logic."""

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from src.evaluator import EmergenceSignalEvaluator

logger = logging.getLogger(__name__)

# Max attempts when response is too short
_MAX_REROLLS = 2
_MIN_RESPONSE_TOKENS = 50

# Per-call cues placed as the final user message. They are not stored in the
# shared conversation history. (A bare "继续。" made small models literally
# continue the previous, truncated sentence.)
_START_CUE = "请开始。"
_TURN_CUE = "轮到你发言。"

# Injected into the *system* message (never into history) when the previous
# utterance repeated earlier content too closely.
CHANGE_ANGLE_NUDGE = (
    "提醒：刚才的内容和前面高度重复。这一轮请换一个完全不同的角度"
    "（例如一个具体例子、一个反例或另一个领域的类比），不要复用前面的句子。"
)
TOPIC_SWITCH_NOTE = "话题已经换成：{topic}。从这个新话题自然地开始。"


@dataclass
class AgentConfig:
    name: str
    model: str
    role: str
    system_prompt_template: str
    temperature: float
    max_tokens: int
    context_window: int
    # Optional per-agent override; None falls back to the config-wide
    # anti_repetition.repeat_penalty, then to a model-based default.
    repeat_penalty: Optional[float] = None


class Agent:
    """An AI agent with a defined persona, capable of conversing with a partner."""

    def __init__(self, config: AgentConfig, partner_name: str, ollama_client, *,
                 default_repeat_penalty: Optional[float] = None):
        self.config = config
        self.name = config.name
        self.partner_name = partner_name
        self.role = config.role
        self.client = ollama_client
        self.default_repeat_penalty = default_repeat_penalty
        self._system_prompt = ""

    def compile_system_prompt(self, topic: str) -> str:
        """Render the system prompt template with partner name and topic."""
        self._system_prompt = self.config.system_prompt_template.format(
            partner_name=self.partner_name,
            topic=topic,
        )
        return self._system_prompt

    def clean_response(self, text: str) -> str:
        """Strip self-dialogue artifacts: 'Name:' prefixes and embedded partner lines."""
        if not text:
            return ""
        for name in [self.name, self.partner_name]:
            text = re.sub(rf'^{re.escape(name)}[：:]\s*', '', text, flags=re.MULTILINE)
        # Invented speaker labels ("你：", "我：") from a model writing a fake dialogue.
        text = re.sub(r'^\s*(?:你|我)\s*[：:]\s*', '', text, flags=re.MULTILINE)
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if re.match(rf'^(?:{re.escape(self.partner_name)})[：:]', stripped):
                continue
            if re.match(rf'^(?:{re.escape(self.name)})[：:]', stripped):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned).strip()

    def build_context(self, *, conversation_history: list[dict],
                      memories: list | None = None,
                      compressed_summary: str = "",
                      depth_level: int = 1,
                      depth_label: str = "",
                      refs_last_turn: int = 1,
                      consecutive_low_refs: int = 0,
                      turns_at_current_depth: int = 0,
                      turns_per_depth: int = 4,
                      transition_to_next_depth: bool = False,
                      meta_reflection_due: bool = False,
                      truncate_own_turns: bool = True) -> str:
        """Build the full input context for this agent's turn.

        ``truncate_own_turns`` shortens the agent's own earlier turns to
        discourage verbatim copying. A monologue baseline must pass False:
        there every history entry is the agent's own, so truncation would hide
        almost the whole conversation and unfairly handicap the baseline.
        """
        parts = []

        # Section 1: Shared insights with usage instruction
        if memories:
            memory_lines = ["- " + m.content for m in memories if hasattr(m, "content")]
            if memory_lines:
                parts.append("你们已经共同积累的见解（请在这些基础上延伸，不要重复已有的观点）：")
                parts.extend(memory_lines)
                parts.append("")

        # Section 2: Depth awareness
        if depth_label:
            parts.append(f"当前探索层次：第{depth_level}层——{depth_label}")
            parts.append("")

        # Section 3: Compressed historical summary
        if compressed_summary and len(conversation_history) >= 3:
            parts.append("前面聊的大致内容：")
            parts.append(compressed_summary)
            parts.append("")

        # Section 4: Recent conversation history with speaker labels
        if conversation_history:
            parts.append("最近聊了这些：")
            for entry in conversation_history:
                agent_label = entry.get("agent", "?")
                content = entry.get("content", "")
                # Truncate the agent's own prior turns to discourage verbatim copying
                if truncate_own_turns and agent_label == self.name and len(content) > 80:
                    content = content[:80] + "…"
                parts.append(f"  -{agent_label}- {content}")
                parts.append("")
        else:
            parts.append("随便聊点什么开始吧。")

        # Section 5: Structural instruction
        parts.append(self._build_instruction(
            depth_level, refs_last_turn, consecutive_low_refs,
            turns_at_current_depth, turns_per_depth,
            transition_to_next_depth, meta_reflection_due,
        ))

        return "\n".join(parts)

    def build_turn_instructions(self, *, memories: list | None = None,
                                compressed_summary: str = "",
                                depth_level: int = 1,
                                depth_label: str = "",
                                consecutive_low_refs: int = 0,
                                transition_to_next_depth: bool = False,
                                meta_reflection_due: bool = False,
                                nudge: str = "",
                                topic_note: str = "") -> str:
        """Everything the agent is *told* this turn, for the system message.

        Instructions, depth hints, background memories and summaries live here
        so they never appear in the shared conversation history, where small
        models would treat them as something a speaker said and copy them.
        """
        parts = []
        if topic_note:
            parts += [topic_note, ""]
        if memories:
            memory_lines = ["- " + m.content for m in memories if hasattr(m, "content")]
            if memory_lines:
                parts.append("背景：你们之前积累的见解（在此基础上延伸，不要复述）：")
                parts.extend(memory_lines)
                parts.append("")
        if compressed_summary:
            parts += ["背景：更早对话的概要（不要复述）：", compressed_summary, ""]
        if depth_label:
            parts += [f"当前探索层次：第{depth_level}层——{depth_label}", ""]
        parts.append(self._build_instruction(
            depth_level, 1, consecutive_low_refs, 0, 4,
            transition_to_next_depth, meta_reflection_due,
        ))
        if nudge:
            parts += ["", nudge]
        parts += ["", "以上是写作要求，不要在回答里复述或引用这些要求。"]
        return "\n".join(parts)

    def build_messages(self, *, conversation_history: list[dict],
                       memories: list | None = None,
                       compressed_summary: str = "",
                       depth_level: int = 1,
                       depth_label: str = "",
                       consecutive_low_refs: int = 0,
                       transition_to_next_depth: bool = False,
                       meta_reflection_due: bool = False,
                       truncate_own_turns: bool = True,
                       nudge: str = "",
                       topic_note: str = "",
                       **_ignored) -> list[dict]:
        """Build chat messages: instructions in system, history as plain turns.

        The agent's own earlier turns become ``assistant`` messages and the
        partner's become ``user`` messages, carrying only what was actually
        said. A short cue is appended when the agent must speak after itself
        (monologue baseline) or first.
        """
        system = self._system_prompt.strip()
        instructions = self.build_turn_instructions(
            memories=memories, compressed_summary=compressed_summary,
            depth_level=depth_level, depth_label=depth_label,
            consecutive_low_refs=consecutive_low_refs,
            transition_to_next_depth=transition_to_next_depth,
            meta_reflection_due=meta_reflection_due,
            nudge=nudge, topic_note=topic_note,
        )
        messages: list[dict] = [{"role": "system",
                                 "content": f"{system}\n\n{instructions}".strip()}]
        for entry in conversation_history:
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            own = entry.get("agent") == self.name
            if own and truncate_own_turns and len(content) > 80:
                content = content[:80] + "…"
            role = "assistant" if own else "user"
            if messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": role, "content": content})
        if messages[-1]["role"] == "system":
            messages.append({"role": "user", "content": _START_CUE})
        elif messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": _TURN_CUE})
        return messages

    def instruction_corpus(self) -> list[str]:
        """Every instruction text this agent can be shown (for echo detection).

        The topic is left out of the rendered system prompt so that merely
        naming the topic is never mistaken for template echo.
        """
        texts = [
            part.format(partner_name=self.partner_name)
            if "{partner_name}" in part else part
            for part in self.config.system_prompt_template.replace("{{", "{").replace("}}", "}")
            .split("{topic}")
        ]
        for depth in range(1, 6):
            for transition in (False, True):
                for meta in (False, True):
                    texts.append(self._build_instruction(depth, 1, 2, 0, 4, transition, meta))
        texts += [
            CHANGE_ANGLE_NUDGE, TOPIC_SWITCH_NOTE.replace("{topic}", ""),
            "背景：你们之前积累的见解（在此基础上延伸，不要复述）：",
            "背景：更早对话的概要（不要复述）：",
            "以上是写作要求，不要在回答里复述或引用这些要求。",
            "你们已经共同积累的见解（请在这些基础上延伸，不要重复已有的观点）：",
            _START_CUE, _TURN_CUE, "当前探索层次：第1层——", "前面聊的大致内容：", "最近聊了这些：", "随便聊点什么开始吧。",
        ]
        return texts

    def _build_instruction(self, depth_level: int,
                           refs_last_turn: int,
                           consecutive_low_refs: int,
                           turns_at_current_depth: int = 0,
                           turns_per_depth: int = 4,
                           transition_to_next_depth: bool = False,
                           meta_reflection_due: bool = False) -> str:
        """Build step-by-step instruction based on depth and engagement."""
        lines = []

        # Depth-transition summarization
        if transition_to_next_depth:
            lines.append("你们即将深入下一层探索。请先暂停一下，总结到目前为止的关键收获：")
            lines.append("  1. 我们目前最核心的洞见是什么？")
            lines.append("  2. 这些洞见中有哪些已经被【质疑】和回应过？哪些还没有？")
            lines.append("  3. 还有哪些没有答案的分歧？")
            lines.append("  4. 接下来应该往哪个方向挖掘？")
            lines.append("用【阶段性总结】标记你的回答。")
            lines.append("")

        # Periodic meta-reflection
        if meta_reflection_due:
            lines.append("暂停一下，反思当前的对话状态：")
            lines.append("  1. 我们是在接近共识还是越来越分散？")
            lines.append("  2. 有没有什么角度被我们忽略了？")
            lines.append("  3. 到目前为止，你自己的理解有什么变化？")
            lines.append("用【元认知】标记你的反思。")
            lines.append("")

        lines.append("请先接对方刚才的话，再提出你的新想法：")
        lines.append('  1. 先提到对方刚说的一个具体观点（用"你"来指对方）')
        lines.append("  2. 然后说你的新角度或新想法")

        # Add dialectical challenge step at depth 2+
        if depth_level >= 2:
            lines.append('  3. 如果对方的观点有漏洞或可质疑之处，一定要指出来——用"但是……"或"有没有可能……"')
            lines.append('     【质疑】过的观点经过回应后，才能算经过检验的结论。')
        else:
            lines.append("  3. 如果对方说的跟你的观察不符，说出来。")

        if depth_level >= 3:
            lines.append("  4. 最后简单总结你们俩目前在哪一点上达成共识，以及还有哪些分歧")

        if consecutive_low_refs >= 2:
            lines.append("")
            lines.append("注意：多回应对方刚说的具体内容，不要只说自己的。")

        lines.append("")
        depth_guides = {
            1: (
                "先关注具体的例子和观察。\n"
                "当你注意到某个有趣的现象或细节时，把它说出来。\n"
                "不用急着总结规律——先把素材摆出来。"
            ),
            2: (
                "试着把你的伙伴的例子联系起来——它们有没有共同之处？\n"
                "能不能把这些例子归几类？每一类有什么特点？\n"
                "寻找重复出现的模式，哪怕很细微。\n"
                "同时检验：这些模式真的成立吗？有没有反例？"
            ),
            3: (
                "尝试提炼一个更通用的原理或框架。\n"
                "用多步推理来推导：'如果X成立→那么Y会发生→因为Z机制→所以结果是……'\n"
                "问自己：'这背后的根本原因是什么？'\n"
                "做完推导后，反过来检验：如果前提是错的，结论还成立吗？"
            ),
            4: (
                "拿这个原理做思想实验：如果改变一个关键变量，会发生什么？\n"
                "用'如果……那么……'来做反事实推理。\n"
                "想想这个原理在现实中有哪些具体的应用场景。\n"
                "同时做压力测试：在极端情况下，这个结论还站得住吗？"
            ),
            5: (
                "试试把这个想法联系到一个完全不同的领域。\n"
                "问：'这在XX领域会是什么样子？'——XX可以是生物、艺术、工程、心理……\n"
                "生成一个跨领域类比，并解释它们为什么在结构上相似。\n"
                "但也要问：这个类比在哪些地方不成立？差异在哪里？"
            ),
        }
        guide = depth_guides.get(depth_level, "")
        if guide:
            lines.append(guide)

        # Knowledge gap identification (depth 3+)
        if depth_level >= 3:
            lines.append("")
            lines.append("思考一下：在这个话题上，还有什么是你不太确定的？")
            lines.append("如果给你一个机会问任何问题来填补知识空白，你会问什么？")

        # Chain-of-thought enforcement for all depths
        lines.append("")
        lines.append("推理要求：请用 [步骤1/2/3…] 标记你的推理链。即使简单的论点也至少展开两步。")
        if depth_level >= 3:
            lines.append("在写出最终观点之前，先展示你的推理过程，再给结论。")
            lines.append("用因果链展开：'因为X→所以Y（机制是Z）→这意味着W'")

        lines.append("")
        lines.append(f"只说{self.name}自己的话，不要替{self.partner_name}说话。")

        return "\n".join(lines)

    def _get_repeat_penalty(self) -> float:
        """repeat_penalty sent to Ollama.

        Per-agent ``repeat_penalty`` wins, then the config-wide
        ``anti_repetition.repeat_penalty``, then a model-based default (qwen
        needs a stronger penalty).
        """
        if self.config.repeat_penalty is not None:
            return float(self.config.repeat_penalty)
        if self.default_repeat_penalty is not None:
            return float(self.default_repeat_penalty)
        model_lower = self.config.model.lower()
        if "qwen" in model_lower:
            return 1.15
        return 1.10

    def think(self, context):
        """Run inference and return (response_text, OllamaResponse).

        ``context`` is either a message list from :meth:`build_messages`
        (preferred) or a legacy context string from :meth:`build_context`.
        """
        if isinstance(context, list):
            messages = context
        else:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": context},
            ]

        for attempt in range(_MAX_REROLLS + 1):
            result = self.client.chat(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                context_window=self.config.context_window,
                repeat_penalty=self._get_repeat_penalty(),
            )

            cleaned = self.clean_response(result.text)
            if not cleaned:
                cleaned = result.text

            if result.tokens_generated >= _MIN_RESPONSE_TOKENS:
                return cleaned, result

            logger.warning(
                "Response too short (%d tokens, attempt %d), re-rolling...",
                result.tokens_generated, attempt + 1
            )

        return self.clean_response(result.text), result

    def extract_references(self, text: str) -> int:
        """Count how many times the partner is referenced in text."""
        patterns = [
            rf"{self.partner_name}",
            # English patterns
            r"as you (?:mentioned|said|noted|pointed out)",
            r"building on your",
            r"your (?:point|analogy|insight|idea|question|suggestion|thought|perspective)",
            r"you (?:raised|brought up|suggested|proposed)",
            r"following your",
            r"to your point",
            r"expanding on",
            # Chinese patterns
            r"你(?:提到|说过|刚才说|的观点|的想法|的比喻|的看法|的思路)",
            r"(?:针对|基于|顺着|接着)你的",
            r"(?:正如|就像)你(?:说的|提到的|讲的)",
            r"让我想到你",
            r"回应你的",
            r"同意你的",
        ]
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def extract_emergence_signals(self, text: str,
                                   emergence_config: dict | None = None) -> dict:
        """Count emergence-related signals in response text.

        Returns dict with keys: novel_concepts, cross_domain_analogies,
                                metacognitive_count, reasoning_steps,
                                knowledge_generation.
        """
        return EmergenceSignalEvaluator(emergence_config).evaluate_dict(text)


def instruction_corpus_from_config(config: dict) -> list[str]:
    """Instruction corpus for both agents of a config (used by the scorer)."""
    agents = (config or {}).get("agents") or {}
    a, b = agents.get("agent_a"), agents.get("agent_b")
    texts: list[str] = []
    for own, partner in ((a, b), (b, a)):
        if not own:
            continue
        try:
            cfg = AgentConfig(**{k: own[k] for k in (
                "name", "model", "role", "system_prompt_template",
                "temperature", "max_tokens", "context_window") if k in own})
            agent = Agent(cfg, (partner or {}).get("name", ""), None)
            texts += agent.instruction_corpus()
        except (TypeError, KeyError, ValueError, IndexError):
            continue
    if not texts:
        texts = Agent(AgentConfig("A", "m", "", "{topic}", 0.5, 1, 512), "B", None).instruction_corpus()
    texts += list(((config or {}).get("memory") or {}).get("depth_labels") or [])
    return texts

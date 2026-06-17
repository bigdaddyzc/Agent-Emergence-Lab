"""Evaluation helpers for emergence-related dialogue signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SignalResult:
    """Counts of heuristic signals found in one response."""

    novel_concepts: int = 0
    cross_domain_analogies: int = 0
    metacognitive_count: int = 0
    reasoning_steps: int = 0
    knowledge_generation: int = 0
    concept_elaboration: int = 0
    question_propagation: int = 0
    critical_challenges: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "novel_concepts": self.novel_concepts,
            "cross_domain_analogies": self.cross_domain_analogies,
            "metacognitive_count": self.metacognitive_count,
            "reasoning_steps": self.reasoning_steps,
            "knowledge_generation": self.knowledge_generation,
            "concept_elaboration": self.concept_elaboration,
            "question_propagation": self.question_propagation,
            "critical_challenges": self.critical_challenges,
        }


@dataclass
class ConceptEvent:
    name: str
    turn_number: int
    agent_name: str
    snippet: str


@dataclass
class ConceptTracker:
    """Tracks named concept mentions across a session."""

    concepts: dict[str, list[ConceptEvent]] = field(default_factory=dict)

    def observe(self, text: str, *, turn_number: int, agent_name: str) -> list[ConceptEvent]:
        events = []
        for match in re.finditer(r"【新概念[:：]\s*([^】]+)】", text):
            name = match.group(1).strip()
            if not name:
                continue
            start = max(match.start() - 40, 0)
            end = min(match.end() + 80, len(text))
            event = ConceptEvent(
                name=name,
                turn_number=turn_number,
                agent_name=agent_name,
                snippet=text[start:end].strip(),
            )
            self.concepts.setdefault(name, []).append(event)
            events.append(event)
        return events

    def summary(self) -> dict[str, dict[str, int | str]]:
        return {
            name: {
                "mentions": len(events),
                "first_turn": events[0].turn_number,
                "last_turn": events[-1].turn_number,
                "source_agent": events[0].agent_name,
            }
            for name, events in self.concepts.items()
        }


class EmergenceSignalEvaluator:
    """Heuristic evaluator for emergence-adjacent textual signals."""

    def __init__(self, emergence_config: dict | None = None):
        self.config = emergence_config or {}

    def evaluate(self, text: str) -> SignalResult:
        ec = self.config
        result = SignalResult()

        for pattern in ec.get("novelty_patterns", ["【新概念"]):
            result.novel_concepts += self._count(pattern, text)

        for pattern in ec.get("analogy_patterns", ["就像", "类比", "联系到", "相通"]):
            result.cross_domain_analogies += self._count(pattern, text, ignore_case=True)

        for pattern in ec.get(
            "metacognitive_patterns",
            ["我在想", "我不确定", "我发现自己", "我们是不是"],
        ):
            result.metacognitive_count += self._count(pattern, text, ignore_case=True)

        for pattern in ec.get("reasoning_patterns", ["第一步", "第二步", "第1步", "第2步"]):
            result.reasoning_steps += self._count(pattern, text, ignore_case=True)

        for pattern in ec.get("knowledge_markers", ["新洞见", "关键发现", "Insight"]):
            result.knowledge_generation += self._count(pattern, text, ignore_case=True)

        for pattern in ec.get("challenge_patterns", ["【质疑", "有没有可能", "反例", "但是"]):
            result.critical_challenges += self._count(pattern, text, ignore_case=True)

        result.concept_elaboration = self._count(
            r"这个概念|这个想法|这个观点.*进一步|这个思路|基于这个概念|在这个基础上",
            text,
        )
        result.question_propagation = self._count(
            r"这让我想问|进一步的问题是|更深一层的问题是|如果.*那么.*是否意味着|进而追问",
            text,
        )
        return result

    def evaluate_dict(self, text: str) -> dict[str, int]:
        return self.evaluate(text).as_dict()

    @staticmethod
    def _count(pattern: str, text: str, *, ignore_case: bool = False) -> int:
        flags = re.IGNORECASE if ignore_case else 0
        return len(re.findall(pattern, text, flags))

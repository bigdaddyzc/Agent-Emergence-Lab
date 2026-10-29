"""Evaluation helpers for emergence-related dialogue signals."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# Character n-gram length used to detect copied instruction text.
DEFAULT_ECHO_NGRAM = 6


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

    def evaluate_clean(self, text: str, references=None, *,
                       ngram: int = DEFAULT_ECHO_NGRAM) -> tuple[SignalResult, float]:
        """Count signals only in the model's own words.

        Text that copies the instruction corpus (``references``) is stripped
        first, and repeated sentences / empty or duplicated reasoning steps are
        collapsed, so template echo and loops do not inflate the counts.
        Returns ``(signals, template_echo_ratio)``.
        """
        clean, echo_ratio = strip_template_echo(text, references, ngram)
        return self.evaluate(dedupe_for_scoring(clean)), echo_ratio

    @staticmethod
    def _count(pattern: str, text: str, *, ignore_case: bool = False) -> int:
        flags = re.IGNORECASE if ignore_case else 0
        return len(re.findall(pattern, text, flags))


class DialogueQualityEvaluator:
    """Tracks repetition, topic drift, and signal stagnation over turns."""

    def __init__(self, *, stagnation_window: int = 3):
        self.previous_by_agent: dict[str, str] = {}
        self.recent_signal_scores: list[int] = []
        self.stagnation_window = stagnation_window

    def evaluate(self, *, agent_name: str, text: str, topic: str,
                 signal_counts: dict[str, int]) -> dict[str, float]:
        repetition = self._similarity(text, self.previous_by_agent.get(agent_name, ""))
        drift = 1.0 - self._similarity(text, topic)
        signal_score = sum(signal_counts.values())
        self.recent_signal_scores.append(signal_score)
        self.recent_signal_scores = self.recent_signal_scores[-self.stagnation_window:]
        stagnation = 1.0 if (
            len(self.recent_signal_scores) == self.stagnation_window
            and sum(self.recent_signal_scores) == 0
        ) else 0.0
        self.previous_by_agent[agent_name] = text
        return {
            "repetition_score": round(repetition, 3),
            "topic_drift_score": round(max(0.0, min(drift, 1.0)), 3),
            "stagnation_flag": stagnation,
        }

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        tokens_a = _simple_tokens(a)
        tokens_b = _simple_tokens(b)
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _simple_tokens(text: str) -> set[str]:
    lowered = text.lower()
    english = set(re.findall(r"[a-z]{3,}", lowered))
    chinese = set(re.findall(r"[\u4e00-\u9fff]{2}", lowered))
    return english | chinese


# ── Template-echo stripping ──────────────────────────────────────────────
#
# Small models often copy the prompt's instruction scaffolding verbatim
# ("先提到对方刚说的一个具体观点", "【步骤1】", ...). That text matches the signal
# regexes but is not the model's own thinking, so it must not score. We mark
# every character of the response that sits inside a character n-gram shared
# with the instruction corpus and drop it before counting signals.



def _is_content_char(ch: str) -> bool:
    """Letters, digits and CJK count; whitespace and punctuation do not."""
    if ch.isspace():
        return False
    category = unicodedata.category(ch)
    return not (category.startswith("P") or category.startswith("S"))


def _content_chars(text: str) -> list[tuple[int, str]]:
    return [(i, ch.lower()) for i, ch in enumerate(text) if _is_content_char(ch)]


def build_reference_ngrams(references: Iterable[str], n: int = DEFAULT_ECHO_NGRAM) -> set[str]:
    """Character n-grams (punctuation and whitespace ignored) of the instructions."""
    grams: set[str] = set()
    for ref in references:
        if not ref:
            continue
        chars = "".join(ch for _, ch in _content_chars(ref))
        for i in range(len(chars) - n + 1):
            grams.add(chars[i:i + n])
    return grams


def strip_template_echo(text: str, references: Iterable[str] | set[str] | None,
                        n: int = DEFAULT_ECHO_NGRAM) -> tuple[str, float]:
    """Remove the parts of ``text`` that copy the instruction corpus.

    Returns ``(clean_text, echo_ratio)`` where ``echo_ratio`` is the share of
    content characters that were removed. ``references`` may be raw texts or
    a precomputed n-gram set from :func:`build_reference_ngrams`.
    """
    if not text:
        return "", 0.0
    if not references:
        return text, 0.0
    grams = references if isinstance(references, set) else build_reference_ngrams(references, n)
    if not grams:
        return text, 0.0
    content = _content_chars(text)
    if len(content) < n:
        return text, 0.0
    joined = "".join(ch for _, ch in content)
    echoed = [False] * len(content)
    for i in range(len(content) - n + 1):
        if joined[i:i + n] in grams:
            for j in range(i, i + n):
                echoed[j] = True
    removed = {content[j][0] for j, flag in enumerate(echoed) if flag}
    if not removed:
        return text, 0.0
    kept = "".join(ch for i, ch in enumerate(text) if i not in removed)
    # Tidy up punctuation/whitespace runs left behind by the removal.
    kept = re.sub(r"[ \t]+", " ", kept)
    kept = re.sub(r"\n\s*\n+", "\n", kept)
    kept = re.sub(r"^[\s，,。：:；;、]+", "", kept, flags=re.MULTILINE)
    return kept.strip(), round(len(removed) / len(content), 3)


# ── Repetition ───────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")
_STEP_MARKER = re.compile(r"[【\[]?\s*步骤\s*\d+\s*[】\]]?")


def internal_repetition(text: str) -> float:
    """How much an utterance repeats itself (0 = no repeats, 1 = all repeats).

    Sentences are compared after removing step markers; a sentence counts as a
    repeat when it is a near-duplicate (token Jaccard >= 0.7) of an earlier one.
    """
    sentences = [
        _STEP_MARKER.sub("", s).strip() for s in _SENTENCE_SPLIT.split(text or "")
    ]
    sentences = [s for s in sentences if len(s) >= 4]
    if len(sentences) < 2:
        return 0.0
    seen: list[set[str]] = []
    repeats = 0
    for sentence in sentences:
        tokens = _simple_tokens(sentence)
        if not tokens:
            continue
        if any(len(tokens & prev) / len(tokens | prev) >= 0.7 for prev in seen):
            repeats += 1
        seen.append(tokens)
    return round(repeats / len(sentences), 3)


def repetition_score(text: str, previous: Iterable[str] = ()) -> float:
    """Per-utterance repetition: max of overlap with earlier utterances and self-repetition."""
    cross = max((DialogueQualityEvaluator._similarity(text, p) for p in previous if p),
                default=0.0)
    return round(max(cross, internal_repetition(text)), 3)


def overall_score(signals_per_utterance: float, repetition_mean: float,
                  *, repetition_weight: float = 1.0) -> float:
    """Signals per utterance, discounted by repetition.

    ``overall = signals_per_utterance * max(0, 1 - weight * repetition_mean)``,
    so a transcript that loops on itself scores lower than a varied one with
    the same signal count, and a fully repetitive one scores 0.
    """
    factor = max(0.0, 1.0 - repetition_weight * max(0.0, repetition_mean))
    return round(signals_per_utterance * factor, 3)


def dedupe_for_scoring(text: str) -> str:
    """Drop repeated sentences and reasoning-step markers with no new content.

    ``【步骤1】X【步骤2】X`` keeps only the first step, and a sentence that is a
    near-duplicate of an earlier one is removed, so a model looping on the same
    sentence cannot multiply its signal count.
    """
    if not text:
        return ""
    pieces = re.split(r"(" + _STEP_MARKER.pattern + r")", text)
    out: list[str] = []
    seen: list[set[str]] = []

    def novel(fragment: str) -> str:
        kept = []
        for sentence in re.split(r"(?<=[。！？!?；;\n])", fragment):
            tokens = _simple_tokens(sentence)
            if tokens and any(len(tokens & prev) / len(tokens | prev) >= 0.8 for prev in seen):
                continue
            if tokens:
                seen.append(tokens)
            kept.append(sentence)
        return "".join(kept)

    pending_marker = ""
    for piece in pieces:
        if not piece:
            continue
        if _STEP_MARKER.fullmatch(piece):
            pending_marker = piece
            continue
        body = novel(piece)
        content = "".join(ch for ch in body if _is_content_char(ch))
        if pending_marker and len(content) >= 8:
            out.append(pending_marker)
        out.append(body)
        pending_marker = ""
    return "".join(out).strip()


def drop_echo_lines(text: str, references, *, threshold: float = 0.5,
                    ngram: int = DEFAULT_ECHO_NGRAM) -> str:
    """Remove whole lines that mostly copy the instructions.

    Used before an utterance enters the *shared* history, so one agent's
    copied scaffolding is not fed to its partner as if it were dialogue. Works
    line by line (unlike :func:`strip_template_echo`) to keep the remaining
    text readable. Falls back to the original text if nothing would be left.
    """
    if not text or not references:
        return text
    grams = references if isinstance(references, set) else build_reference_ngrams(references, ngram)
    kept = [line for line in text.split("\n")
            if strip_template_echo(line, grams, ngram)[1] < threshold]
    result = "\n".join(kept).strip()
    return result if len("".join(ch for ch in result if _is_content_char(ch))) >= 10 else text

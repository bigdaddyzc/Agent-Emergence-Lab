"""Runtime guard against repetitive dialogue.

After every utterance the guard scores how much it repeats the recent
conversation (and itself). Above ``threshold`` it either asks the next speaker
to change angle (a nudge placed in that speaker's *system* message, never in
shared history), ends the current round early, and/or ends the session after
too many repetitive utterances in a row.

Config (all optional)::

    anti_repetition:
      enabled: true
      repeat_penalty: 1.15      # sent to Ollama for every agent without its own value
      threshold: 0.6            # repetition score that counts as "repetitive"
      action: nudge             # nudge | end_round | none
      max_consecutive: 3        # end the session after N repetitive utterances in a row (0 = never)
      window: 4                 # how many previous utterances to compare against
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent import CHANGE_ANGLE_NUDGE
from src.evaluator import repetition_score

ACTIONS = ("nudge", "end_round", "none")

DEFAULTS = {
    "enabled": True,
    "repeat_penalty": None,
    "threshold": 0.6,
    "action": "nudge",
    "max_consecutive": 3,
    "window": 4,
}


def anti_repetition_settings(config: dict | None) -> dict:
    settings = dict(DEFAULTS)
    settings.update(((config or {}).get("anti_repetition") or {}))
    return settings


@dataclass
class RepetitionVerdict:
    score: float
    repetitive: bool
    nudge_next: bool = False
    end_round: bool = False
    end_session: bool = False


class RepetitionGuard:
    def __init__(self, config: dict | None = None):
        s = anti_repetition_settings(config)
        self.enabled = bool(s["enabled"])
        self.threshold = float(s["threshold"])
        self.action = s["action"] if s["action"] in ACTIONS else "nudge"
        self.max_consecutive = int(s["max_consecutive"] or 0)
        self.window = max(1, int(s["window"]))
        self.recent: list[str] = []
        self.consecutive = 0
        self.nudges_sent = 0
        self._pending_nudge = ""

    def observe(self, text: str) -> RepetitionVerdict:
        score = repetition_score(text, self.recent[-self.window:])
        self.recent.append(text)
        if not self.enabled:
            return RepetitionVerdict(score=score, repetitive=False)
        if score <= self.threshold:
            self.consecutive = 0
            return RepetitionVerdict(score=score, repetitive=False)
        self.consecutive += 1
        verdict = RepetitionVerdict(score=score, repetitive=True)
        if self.action in ("nudge", "end_round"):
            self._pending_nudge = CHANGE_ANGLE_NUDGE
            verdict.nudge_next = True
        verdict.end_round = self.action == "end_round"
        verdict.end_session = bool(self.max_consecutive) and self.consecutive >= self.max_consecutive
        return verdict

    def take_nudge(self) -> str:
        """Return (and clear) the nudge for the next speaker, if any."""
        nudge, self._pending_nudge = self._pending_nudge, ""
        if nudge:
            self.nudges_sent += 1
        return nudge

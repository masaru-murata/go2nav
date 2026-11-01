"""rationale: the mandatory decision record of every turn (record kind; never executed)."""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="rationale", kind="record", side="brain", needs_motion=False,
    description="毎回の応答に必ず含める判断記録。where_am_i=今どこにいると考えるか、goal_belief=目標（探し物・人・場所）はどちらと考えるか、"
                "options=検討した選択肢と却下理由、chosen=選んだ行動、why=理由、confidence=0〜1。",
    params={"where_am_i": {"type": "string"}, "goal_belief": {"type": "string"},
            "options": {"type": "array", "items": {"type": "object", "properties": {"action": {"type": "string"}, "rejected_because": {"type": "string"}},
                                                    "required": ["action", "rejected_because"], "additionalProperties": False}},
            "chosen": {"type": "string"}, "why": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
    required=["where_am_i", "goal_belief", "options", "chosen", "why", "confidence"])

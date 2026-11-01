"""remember: append a note to the task memory (record kind)."""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="remember", kind="record", side="brain", needs_motion=False,
    description="仮説・証拠・行き止まり・指示のノートを記憶に追記する。",
    params={"note": {"type": "string"}, "kind": {"type": "string", "enum": ["hypothesis", "evidence", "dead_end", "instruction"]}},
    required=["note", "kind"])

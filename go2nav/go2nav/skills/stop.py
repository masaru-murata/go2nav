"""stop: immediate stop and end of the task (control kind, brain side)."""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="stop", kind="control", side="brain", alone=True, needs_motion=False,
    description="即時停止して終了（単独で返す）。", params={"reason": {"type": "string"}}, required=["reason"])


def run_brain(ex, call, turn: int, images: list) -> dict:
    ex.finish("stopped", {"reason": call.args.get("reason", "")})
    return {"tool": "stop", "status": "ok"}

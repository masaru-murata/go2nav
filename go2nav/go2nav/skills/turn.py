"""turn: in-place rotation to a named bearing (robot side)."""
from __future__ import annotations

import math

from ..common import wrap_rad
from .registry import SkillSpec

TURN_BEARINGS = ["left_45", "left_90", "right_45", "right_90", "around"]

SPEC = SkillSpec(
    name="turn", kind="action", description="その場で旋回する。",
    params={"bearing": {"type": "string", "enum": TURN_BEARINGS}}, required=["bearing"])


def run(runner, job) -> None:
    cfg = runner.cfg["turn"]
    bearing = job.args.get("bearing")
    deg = cfg["bearings"].get(bearing) if isinstance(bearing, str) else None
    if deg is None:
        try:
            deg = float(bearing)
        except (TypeError, ValueError):
            job.status, job.result = "failed", {"reason": f"bad_bearing:{bearing}"}
            return
    runner.phase(job, "turning")
    y0 = runner.yaw()
    a = runner.turn_to(job, y0 + math.radians(deg), cfg)
    job.result["turned_deg"] = round(math.degrees(wrap_rad(runner.yaw() - y0)), 1)
    if a is None:
        job.status = "ok"

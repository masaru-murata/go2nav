"""go_forward_until: drive straight until an opening / blockage / distance (robot side)."""
from __future__ import annotations

import time

from .registry import SkillSpec

SPEC = SkillSpec(
    name="go_forward_until", kind="action",
    description="前進し、事象で止まる。stop_on の事象（opening=左右が開けた、blocked=前方1m未満、distance=max_m到達）のいずれかで停止する。1回最大15m・60秒。",
    params={"stop_on": {"type": "array", "items": {"type": "string", "enum": ["opening", "blocked", "distance"]}},
            "max_m": {"type": "number", "minimum": 0.5, "maximum": 15}},
    required=["stop_on", "max_m"], makes_node=True)


def run(runner, job) -> None:
    cfg = runner.cfg["go_forward_until"]
    stop_on = set(job.args.get("stop_on") or ["opening", "blocked", "distance"])
    req = float(job.args.get("max_m") or cfg["max_m"])
    max_m = min(req, float(cfg["max_m"]))
    if max_m < req:
        job.result["truncated"] = True
    runner.phase(job, "forward")
    a, why = runner.drive_forward(job, max_m, float(cfg["timeout_s"]), stop_on, float(cfg["cruise_vx"]))
    runner.core.set_cmd(0.0, 0.0, 0.0, job.tool)
    job.result["stopped_because"] = why
    if a is None and why == "stalled":
        job.status, job.result["reason"] = "failed", "stalled: 機体が前進指令に反応しない（側方や斜め前の障害物に機体側の回避が反応している可能性）"
    elif a is None:
        job.status = "ok"
        runner.snapshot(job, 0, hires=False, settle_s=float(runner.cfg["turn"].get("settle_s", 0.3)))
    job.result["free_space"] = runner.core.free_space()

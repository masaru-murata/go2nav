"""approach: face a bearing and close in until the range ahead equals stop_m (robot side)."""
from __future__ import annotations

import math

from .registry import SkillSpec

BEARINGS = ["ahead", "ahead_left", "left", "behind_left", "behind", "behind_right", "right", "ahead_right"]

SPEC = SkillSpec(
    name="approach", kind="action",
    description="指定方位へ向き直り、その方位の距離が stop_m になるまで（最大5m）前進する。目標らしきものに寄るときに使う。物は stop_m=0.5 まで、人は 1.0 を目安。",
    params={"bearing": {"type": "string", "enum": BEARINGS}, "stop_m": {"type": "number", "minimum": 0.5, "maximum": 3.0}},
    required=["bearing", "stop_m"])


def run(runner, job) -> None:
    cfg = runner.cfg["approach"]
    deg = runner.bearing_deg(job.args.get("bearing", "ahead"))
    if deg is None:
        job.status, job.result = "failed", {"reason": f"bad_bearing:{job.args.get('bearing')}"}
        return
    stop_m = max(float(cfg["min_stop_m"]), float(job.args.get("stop_m") or cfg["default_stop_m"]))
    runner.phase(job, "turning")
    if abs(deg) > 1.0:
        if runner.turn_to(job, runner.yaw() + math.radians(deg), runner.cfg["turn"]):
            return
        runner.wait_fresh_lidar(job)   # the accumulation window still holds points swept during the turn
    runner.phase(job, "approaching")
    a, why = runner.drive_forward(job, float(cfg["max_m"]), float(cfg["timeout_s"]), {"blocked", "distance"},
                                  float(cfg["vx"]), stop_ahead_m=stop_m)
    runner.core.set_cmd(0.0, 0.0, 0.0, job.tool)
    job.result["final_range_m"] = round(runner.sectors().get("ahead", 99.0), 2)
    job.result["stopped_because"] = "reached" if why == "blocked" else why
    if a == "interrupted" and str(job.result.get("interrupted_by", "")).startswith("gate_min_range") \
            and job.result["final_range_m"] <= stop_m + float(cfg.get("gate_stop_tolerance_m", 0.25)):
        # the L1 gate stopped us right at the target (sensor lag): that is a successful approach, not an interruption
        job.result["note"] = f"stopped_by_gate:{job.result.pop('interrupted_by')}"
        job.result["stopped_because"], a = "reached", None
    if a is None and why == "stalled":
        job.status, job.result["reason"] = "failed", "stalled: 機体が前進指令に反応しない（側方や斜め前の障害物に機体側の回避が反応している可能性）"
    elif a is None:
        job.status = "ok"
        runner.snapshot(job, 0, hires=True, settle_s=0.3)
    job.result["free_space"] = runner.core.free_space()

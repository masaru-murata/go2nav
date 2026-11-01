"""zoom: face a bearing, close in up to 2 m, take a hires image, turn back (robot side)."""
from __future__ import annotations

import math
import time

from .approach import BEARINGS
from .registry import SkillSpec

SPEC = SkillSpec(
    name="zoom", kind="observe",
    description="指定方位へ向き、最大2m寄って高解像度で1枚撮り、元の向きに戻る。表示が読めない・顔が小さいときだけ使う。",
    params={"bearing": {"type": "string", "enum": BEARINGS}}, required=["bearing"], needs_motion=False)


def run(runner, job) -> None:
    core = runner.core
    cfg = runner.cfg["zoom"]
    deg = runner.bearing_deg(job.args.get("bearing", "ahead"))
    if deg is None:
        job.status, job.result = "failed", {"reason": f"bad_bearing:{job.args.get('bearing')}"}
        return
    if not core.allow_motion:
        frame = core.camera.fresh_frame(500)
        if not frame:
            job.status, job.result = "failed", {"reason": "no_frame"}
            return
        job.result["images"] = [runner.img(frame, 0, hires=True, kind="zoom")]
        job.result["note"] = "motion_disabled: hires crop of the current forward view"
        job.status = "ok"
        return
    if core.gate.latch is not None:
        job.status, job.result = "rejected_by_gate", {"reason": f"latched:{core.gate.latch}"}
        return
    if core.pose != "standing" and not runner.ensure_standing(job):
        return
    y0 = runner.yaw()
    runner.phase(job, "turning")
    if abs(deg) > 1.0:
        if runner.turn_to(job, y0 + math.radians(deg), runner.cfg["turn"]):
            return
        runner.wait_fresh_lidar(job)   # the accumulation window still holds points swept during the turn
    runner.phase(job, "closing_in")
    a, why = runner.drive_forward(job, float(cfg["max_m"]), float(cfg["timeout_s"]), {"blocked", "distance"},
                                  float(cfg["vx"]), stop_ahead_m=float(cfg["stop_ahead_m"]))
    core.set_cmd(0.0, 0.0, 0.0, job.tool)
    if a:
        return
    time.sleep(0.4)
    frame = core.camera.fresh_frame(400)
    job.result["images"] = [runner.img(frame, deg, hires=True, kind="zoom")] if frame else []
    runner.phase(job, "return_heading")
    a = runner.turn_to(job, y0, runner.cfg["turn"])
    job.result["free_space"] = core.free_space()
    job.result["moved_m"] = job.result.get("traveled_m", 0.0)
    if a is None:
        job.status = "ok"

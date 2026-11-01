"""look_around: 5 images at -90/-45/0/45/90 deg, then return to the original heading (robot side)."""
from __future__ import annotations

import math
import time

from .registry import SkillSpec

SPEC = SkillSpec(
    name="look_around", kind="observe",
    description="その場で -90,-45,0,45,90 度の5方向を撮影して返す（正面は高解像度の切り出し付き）。分岐点や、探し物・人を探すときに使う。",
    params={}, required=[], needs_motion=False)


def run(runner, job) -> None:
    core = runner.core
    cfg = runner.cfg["look_around"]
    bearings = list(cfg.get("bearings_deg", [-90, -45, 0, 45, 90]))
    images: list[dict] = []
    if not core.allow_motion:
        # read-only mode: a single forward image (cannot turn)
        frame = core.camera.fresh_frame(500)
        if not frame:
            job.status, job.result = "failed", {"reason": "no_frame"}
            return
        job.result["images"] = [runner.img(frame, 0, hires=True)]
        job.result["note"] = "motion_disabled: forward image only"
        job.result["free_space"] = core.free_space()
        job.status = "ok"
        return
    if core.gate.latch is not None:
        job.status, job.result = "rejected_by_gate", {"reason": f"latched:{core.gate.latch}"}
        return
    if core.pose != "standing" and not runner.ensure_standing(job):
        return
    y0 = runner.yaw()
    for b in sorted(bearings):
        runner.phase(job, f"turn_{b}")
        if runner.turn_to(job, y0 + math.radians(b), runner.cfg["turn"]):
            job.result["images"] = images
            return
        time.sleep(float(cfg.get("settle_s", 0.5)))
        frame = core.camera.fresh_frame(400)
        if frame:
            images.append(runner.img(frame, b, hires=(b == 0), kind="look"))
    runner.phase(job, "return")
    a = runner.turn_to(job, y0, runner.cfg["turn"])
    job.result["images"] = images
    job.result["free_space"] = core.free_space()
    if a is None:
        job.status = "ok"

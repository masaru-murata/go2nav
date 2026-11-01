"""observe: one forward image without moving (robot side, allowed in read-only mode)."""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="observe", kind="observe", description="前方画像1枚（768px）と周囲距離を返す。行動は消費しない。",
    params={}, required=[], needs_motion=False)


def run(runner, job) -> None:
    runner.phase(job, "capture")
    frame = runner.core.camera.fresh_frame(int(runner.cfg["observe"].get("max_frame_age_ms", 300)))
    if not frame:
        job.status, job.result = "failed", {"reason": "no_frame"}
        return
    job.result["images"] = [runner.img(frame, 0, hires=bool(job.args.get("hires", False)))]
    job.result["free_space"] = runner.core.free_space()
    job.status = "ok"

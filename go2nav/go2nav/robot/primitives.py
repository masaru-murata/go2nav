"""Job runner and motion library of the robot process (design 6.1 / 6.2).

The executors themselves live in `go2nav/skills/<name>.py` (`run(runner, job)`); this module runs one job
at a time in a worker thread and offers the shared helpers (closed-loop turn, forward drive with
opening/blocked detection, image capture). Each job loops at ~20 Hz, writes the desired velocity through
`core.set_cmd()`, and ends on its own termination event, a cancel request, or a core interrupt (gate refusal /
latch). Results follow design 6.5/6.6: status ∈ {ok, failed, cancelled, rejected_by_gate, timeout, interrupted}.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..common import mono, now, wrap_rad
from .camera import resize_long_edge, center_crop
from .core import RobotCore
from .. import skills

log = logging.getLogger("go2nav.primitives")

TICK = 0.05


@dataclass
class Job:
    id: str
    tool: str
    args: dict
    status: str = "queued"          # queued | running | ok | failed | cancelled | rejected_by_gate | timeout | interrupted
    result: dict = field(default_factory=dict)
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    phase: str = ""
    done: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        status = self.status if (self.done or self.status in ("queued", "running")) else "running"
        return {"job_id": self.id, "tool": self.tool, "args": self.args, "status": status, "phase": self.phase,
                "t_start": self.t_start, "t_end": self.t_end, **self.result}


class JobRunner:
    def __init__(self, core: RobotCore, cfg: dict):
        self.core = core
        self.cfg = cfg          # primitives.yaml
        self.jobs: dict[str, Job] = {}
        self.current: Optional[Job] = None
        self._lock = threading.Lock()

    # -- API ---------------------------------------------------------------------------
    def submit(self, tool: str, args: dict) -> Job:
        spec = skills.get(tool)
        if spec is None or spec.side != "robot" or spec.run is None:
            raise ValueError(f"unknown tool {tool}")
        with self._lock:
            if self.current is not None and not self.current.done:
                raise RuntimeError("busy")
            job = Job(id=f"j-{uuid.uuid4().hex[:8]}", tool=tool, args=dict(args or {}))
            self.jobs[job.id] = job
            self.current = job
            if len(self.jobs) > 200:
                for k in list(self.jobs)[:-100]:
                    self.jobs.pop(k, None)
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"job-{tool}").start()
        return job

    def cancel(self) -> Optional[str]:
        with self._lock:
            j = self.current
        if j is not None and not j.done:
            j.cancel.set()
            return j.id
        return None

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def busy(self) -> bool:
        j = self.current
        return j is not None and not j.done

    # -- execution ---------------------------------------------------------------------
    def _run(self, job: Job) -> None:
        core = self.core
        job.status = "running"
        job.t_start = now()
        core.job_status = {"id": job.id, "tool": job.tool, "phase": "start"}
        core.take_interrupt()   # discard stale interrupts from before this job
        spec = skills.get(job.tool)
        try:
            if spec.needs_motion and not core.allow_motion:
                job.status, job.result = "rejected_by_gate", {"reason": "motion_disabled"}
            elif spec.needs_motion and core.gate.latch is not None:
                job.status, job.result = "rejected_by_gate", {"reason": f"latched:{core.gate.latch}"}
            elif spec.kind == "action" and core.pose != "standing" and not self.ensure_standing(job):
                pass   # ensure_standing filled status/result
            else:
                spec.run(self, job)
        except Exception as e:
            log.exception("job %s failed: %s", job.tool, e)
            job.status, job.result = "failed", {"reason": f"exception:{e}"}
        finally:
            core.clear_cmd()
            job.t_end = now()
            core.job_status = {"id": None, "tool": None, "phase": ""}
            core.last_stop_reason = job.result.get("stopped_because") or job.result.get("interrupted_by") or job.status
            core._event("job_end", job_id=job.id, tool=job.tool, status=job.status, result={k: v for k, v in job.result.items() if k != "images"})
            job.done = True

    # -- helpers -----------------------------------------------------------------------
    def phase(self, job: Job, p: str) -> None:
        job.phase = p
        self.core.job_status = {"id": job.id, "tool": job.tool, "phase": p}

    def check_abort(self, job: Job) -> Optional[str]:
        """Return a terminal status if the job must stop now."""
        if job.cancel.is_set():
            job.status, job.result["reason"] = "cancelled", "cancel_requested"
            return "cancelled"
        r = self.core.take_interrupt()
        if r:
            if r.startswith("gate_"):
                job.status = "interrupted"
                job.result["interrupted_by"] = r
            else:
                job.status = "interrupted"
                job.result["interrupted_by"] = r
            return "interrupted"
        return None

    def sectors(self) -> dict:
        s = self.core.sectors
        return {} if s is None else s.dist

    def yaw(self) -> float:
        return self.core.odom.yaw

    def travel(self) -> float:
        return self.core.odom.traveled_total

    def ensure_standing(self, job: Job) -> bool:
        """Recover the walking stance before a motion skill when a pose skill left the robot sitting/lying."""
        from ..skills.pose import recover_standing
        self.phase(job, "recover_stand")
        ok, why = recover_standing(self, job)
        if not ok:
            job.status, job.result = "failed", {"reason": f"pose_recovery_failed:{why}"}
        return ok

    # -- turn ----------------------------------------------------------------------------
    def turn_to(self, job: Job, target_yaw: float, cfg: dict) -> Optional[str]:
        """Closed-loop in-place turn to an absolute yaw (rad). Returns terminal status or None on success."""
        kp, vmax, vmin = float(cfg.get("kp", 1.5)), float(cfg.get("max_vyaw", 0.8)), float(cfg.get("min_vyaw", 0.3))
        tol = math.radians(float(cfg.get("tol_deg", 3.0)))
        timeout = float(cfg.get("timeout_s", 15.0))
        # stall handling (2026-10-30: the firmware ignored |vyaw| <= 0.3 rad/s near an obstacle and the turn timed out):
        # if the yaw rate stays ~0 while we command, first kick at max_vyaw, then accept the heading when the error is small.
        stall_s = float(cfg.get("stall_s", 1.0))
        stall_rate = float(cfg.get("stall_rate", 0.05))
        accept = math.radians(float(cfg.get("stall_accept_deg", 10.0)))
        t0 = mono()
        last_yaw, last_t = self.yaw(), t0
        stalled_since = None
        kicks = 0
        while True:
            a = self.check_abort(job)
            if a:
                return a
            y = self.yaw()
            t = mono()
            err = wrap_rad(target_yaw - y)
            if abs(err) <= tol:
                self.core.set_cmd(0.0, 0.0, 0.0, job.tool)
                time.sleep(float(cfg.get("settle_s", 0.3)))
                return None
            if t - last_t >= 0.25:
                rate = abs(wrap_rad(y - last_yaw)) / (t - last_t)
                last_yaw, last_t = y, t
                if rate < stall_rate:
                    stalled_since = stalled_since or t
                else:
                    stalled_since = None
            stalled = stalled_since is not None and t - stalled_since >= stall_s
            if stalled and abs(err) <= accept:
                job.result["stall_accepted_deg"] = round(math.degrees(err), 1)
                self.core.set_cmd(0.0, 0.0, 0.0, job.tool)
                time.sleep(float(cfg.get("settle_s", 0.3)))
                return None
            if t - t0 > timeout:
                job.status, job.result["reason"] = "timeout", "turn_timeout"
                return "timeout"
            w = max(vmin, min(vmax, kp * abs(err)))
            if stalled:          # kick: full speed for one window, then re-evaluate
                w = vmax
                kicks += 1
                stalled_since = None
                job.result["stall_kicks"] = kicks
            self.core.set_cmd(0.0, 0.0, w * (1 if err > 0 else -1), job.tool)
            time.sleep(TICK)

    # -- forward drive ---------------------------------------------------------------------
    def drive_forward(self, job: Job, max_m: float, timeout_s: float, stop_on: set[str], vx_cruise: float,
                      stop_ahead_m: Optional[float] = None, hold_heading: bool = True) -> tuple[Optional[str], str]:
        """Drive forward until an event. Returns (abort_status, stopped_because)."""
        cfg = self.cfg["go_forward_until"]
        ocfg = cfg["opening"]
        slow_vx, slow_below = float(cfg["slow_vx"]), float(cfg["slow_below_ahead_m"])
        blocked_m = float(cfg["blocked_ahead_m"]) if stop_ahead_m is None else float(stop_ahead_m)
        tol = math.radians(float(cfg.get("heading_hold_tol_deg", 4.0)))
        kp = float(cfg.get("heading_kp", 1.2))
        yaw0 = self.yaw()
        d0 = self.travel()
        t0 = mono()
        side_state = {"left": "unknown", "right": "unknown"}
        side_count = {"left": 0, "right": 0}
        last_sect_t = None
        # stall handling (2026-10-30: the firmware ignored vx <= 0.2 m/s with an object 0.9 m off the front-right corner):
        # no progress while commanding -> kick at kick_vx; still no progress -> end with stopped_because=stalled
        st_cfg = self.cfg.get("stall", {})
        stall_s, stall_speed = float(st_cfg.get("forward_stall_s", 1.5)), float(st_cfg.get("forward_stall_speed", 0.03))
        kick_vx, give_up_s = float(st_cfg.get("kick_vx", 0.35)), float(st_cfg.get("forward_give_up_s", 4.0))
        last_d, last_t, stalled_since, kicking = d0, t0, None, False
        while True:
            a = self.check_abort(job)
            if a:
                return a, job.result.get("interrupted_by", "cancelled")
            tn = mono()
            if tn - last_t >= 0.25:
                speed = (self.travel() - last_d) / (tn - last_t)
                last_d, last_t = self.travel(), tn
                if speed < stall_speed and tn - t0 > 1.0:
                    stalled_since = stalled_since or tn
                else:
                    stalled_since, kicking = None, False
            if stalled_since is not None:
                if tn - stalled_since >= give_up_s:
                    job.result["stalled_s"] = round(tn - stalled_since, 1)
                    return None, "stalled"
                if tn - stalled_since >= stall_s and not kicking:
                    kicking = True
                    job.result["stall_kicks"] = int(job.result.get("stall_kicks", 0)) + 1
            s = self.core.sectors
            if s is None:
                self.core.set_cmd(0.0, 0.0, 0.0, job.tool)
                time.sleep(TICK)
                if mono() - t0 > 2.0:
                    job.status, job.result["reason"] = "failed", "no_lidar"
                    return "failed", "no_lidar"
                continue
            traveled = self.travel() - d0
            job.result["traveled_m"] = round(traveled, 2)
            ahead = s.dist.get("ahead", 99.0)
            if ahead < blocked_m:
                return None, "blocked"
            if traveled >= max_m:
                return None, "distance"
            if mono() - t0 > timeout_s:
                job.status, job.result["reason"] = "timeout", "forward_timeout"
                return "timeout", "timeout"
            # opening detection on new lidar frames only: a side is "walled" while its median range is short,
            # and "open" once the median range (i.e. at least half of the sector's rays) exceeds open_min_m.
            if "opening" in stop_on and s.t != last_sect_t:
                last_sect_t = s.t
                for side in ("left", "right"):
                    med = s.median.get(side, s.dist.get(side, 99.0))
                    if med < float(ocfg["side_wall_max_m"]):
                        side_state[side] = "wall"
                        side_count[side] = 0
                    elif med > float(ocfg["open_min_m"]):
                        if side_state[side] == "wall" or ocfg.get("fire_without_prior_wall", False):
                            side_count[side] += 1
                            if side_count[side] >= int(ocfg["confirm_frames"]) and traveled >= float(ocfg["min_travel_m"]):
                                # the opening fires at its leading edge: advance a little so the robot stands in the branch's mouth
                                adv = float(ocfg.get("center_advance_m", 0.0))
                                if adv > 0:
                                    job.result["opening_edge_at_m"] = round(traveled, 2)
                                    a2, why2 = self.advance(job, adv, blocked_m, slow_vx)
                                    job.result["traveled_m"] = round(self.travel() - d0, 2)
                                    if a2:
                                        return a2, job.result.get("interrupted_by", why2)
                                return None, f"opening_{side}"
                    else:
                        side_count[side] = 0
            vx = slow_vx if ahead < slow_below else vx_cruise
            if stop_ahead_m is not None and ahead < blocked_m + float(cfg.get("creep_zone_m", 0.5)):
                vx = min(vx, float(cfg.get("creep_vx", 0.15)))     # approach/zoom: creep for the last stretch (overshoot ~17 cm at 0.3 m/s)
            if kicking:
                vx = max(vx, kick_vx)
            w = 0.0
            if hold_heading:
                err = wrap_rad(yaw0 - self.yaw())
                if abs(err) > tol:
                    w = max(-0.5, min(0.5, kp * err))
            self.core.set_cmd(vx, 0.0, w, job.tool)
            time.sleep(TICK)

    def advance(self, job: Job, dist_m: float, blocked_m: float, vx: float, timeout_s: float = 10.0) -> tuple[Optional[str], str]:
        """Short straight advance (no opening detection), stops early when blocked."""
        d0 = self.travel()
        t0 = mono()
        last_d, last_t = d0, t0
        while self.travel() - d0 < dist_m:
            a = self.check_abort(job)
            if a:
                return a, "interrupted"
            if mono() - last_t >= 1.0:
                if self.travel() - last_d < 0.02 and mono() - t0 > 1.5:
                    return None, "stalled"
                last_d, last_t = self.travel(), mono()
            s = self.core.sectors
            if s is not None and s.dist.get("ahead", 99.0) < blocked_m:
                return None, "blocked"
            if mono() - t0 > timeout_s:
                return None, "timeout"
            self.core.set_cmd(vx, 0.0, 0.0, job.tool)
            time.sleep(TICK)
        return None, "distance"

    # -- bearings / images --------------------------------------------------------------------
    def bearing_deg(self, bearing) -> Optional[float]:
        if isinstance(bearing, str):
            v = self.cfg["bearing_vocab"].get(bearing)
            if v is None:
                v = self.cfg["turn"]["bearings"].get(bearing)
            return None if v is None else float(v)
        try:
            return float(bearing)
        except (TypeError, ValueError):
            return None

    def img(self, jpeg: bytes, bearing_deg: float, hires: bool = False, kind: str = "view") -> dict:
        long_edge = int(self.cfg.get("observe", {}).get("image_long_edge_px", 768))
        small = resize_long_edge(jpeg, long_edge)
        rec = {"id": self.core.store_image(small), "bearing_deg": bearing_deg, "kind": kind, "t": now(),
               "free_space": self.core.free_space()}
        if hires:
            crop = center_crop(jpeg, float(self.cfg["look_around"].get("hires_center_crop", 0.5)))
            rec["hires_id"] = self.core.store_image(crop)
        return rec

    def snapshot(self, job: Job, bearing_deg: float = 0.0, hires: bool = False, kind: str = "view",
                 max_age_ms: int = 500, settle_s: float = 0.0) -> None:
        """Capture one image into job.result['images'] (appends) after an optional settle."""
        if settle_s > 0:
            time.sleep(settle_s)
        frame = self.core.camera.fresh_frame(max_age_ms)
        if frame:
            job.result.setdefault("images", []).append(self.img(frame, bearing_deg, hires=hires, kind=kind))

    def wait_fresh_lidar(self, job: Job) -> Optional[str]:
        """Wait one LiDAR accumulation window so range readings reflect the current heading (after an in-place turn)."""
        return self.sleep_abortable(job, float(self.core.cfg["lidar"].get("accumulate_s", 0.4)) + 0.1)

    def sleep_abortable(self, job: Job, seconds: float) -> Optional[str]:
        """Sleep in small slices while watching for cancel/latch (used by non-velocity skills such as pose)."""
        t0 = mono()
        while mono() - t0 < seconds:
            a = self.check_abort(job)
            if a:
                return a
            time.sleep(TICK)
        return None

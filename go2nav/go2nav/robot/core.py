"""RobotCore: owns the adapter, gate, remote monitor, LiDAR vocabulary, odometry, 20 Hz control loop,
state log, images and the active task directory. Primitives (jobs) drive it through `set_cmd()`.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from ..common import mono, now, iso, JsonlWriter, wrap_rad, disk_free_fraction
from .adapter import RobotAdapter
from .camera import CameraService
from .gate import Gate, Cmd, GateDecision
from .lidar import LidarProcessor, Sectors, SECTORS
from .remote import RemoteMonitor

log = logging.getLogger("go2nav.core")


class Odometry:
    """Integrates planar path length and tracks heading from sportmodestate (position + imu yaw)."""

    def __init__(self):
        self.x = self.y = self.yaw = 0.0
        self.traveled_total = 0.0
        self._last_xy: Optional[tuple[float, float]] = None
        self.t: Optional[float] = None

    def update(self, x: float, y: float, yaw: float, t: float) -> None:
        if self._last_xy is not None:
            d = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if d < 1.0:  # ignore jumps (odometry resets)
                self.traveled_total += d
        self._last_xy = (x, y)
        self.x, self.y, self.yaw, self.t = x, y, yaw, t


class RobotCore:
    def __init__(self, adapter: RobotAdapter, cfg: dict, allow_motion: bool = False):
        """cfg: dict with keys robot, gate, remote, lidar, primitives (loaded YAMLs)."""
        self.adapter = adapter
        self.cfg = cfg
        self.allow_motion = bool(allow_motion)
        self.gate = Gate(cfg["gate"], allow_motion=self.allow_motion)
        self.remote = RemoteMonitor(cfg["remote"])
        self.lidar = LidarProcessor(cfg["lidar"])
        self.camera = CameraService(adapter, cfg["robot"].get("camera", {}))
        self.odom = Odometry()
        self.sectors: Optional[Sectors] = None
        self._desired: Optional[Cmd] = None
        self._lock = threading.Lock()
        self.moving = False
        self._last_nonzero_t = -1e9
        self.last_decision: Optional[GateDecision] = None
        self.last_sent: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_stop_reason: Optional[str] = None
        self.motion_status = "stopped"
        self.interrupt: Optional[str] = None       # reason that should cancel the running job (gate/latch)
        self.events: list[dict] = []
        self.images: "collections.OrderedDict[str, tuple[bytes, float]]" = collections.OrderedDict()
        self.task_id: Optional[str] = None
        self.task_dir: Optional[Path] = None
        self._robot_log: Optional[JsonlWriter] = None
        self._run = False
        self._threads: list[threading.Thread] = []
        self.started_at = now()
        self.job_status: dict = {}   # updated by JobRunner for the state log
        self._wl_t = -1.0
        self.oa_state: Optional[bool] = None
        self.pose = "standing"       # standing | sitting | lying  (set by the pose skill; motion skills recover standing first)

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        self.adapter.start()
        self.classic_walk_code: Optional[int] = None
        if self.allow_motion and self.cfg["robot"].get("classic_walk_on_start", False):
            # 2026-10-30: the default (learned) gait ignored |vyaw|<=0.3 rad/s and vx<=0.2 m/s near obstacles; classic walk follows small commands
            self.classic_walk_code = self.adapter.set_classic_walk(True)
            log.info("classic walk on start -> code=%s", self.classic_walk_code)
        if self.allow_motion and self.cfg["robot"].get("obstacle_avoid_on_start", True):
            code = self.adapter.set_obstacle_avoid(True)
            self.oa_state = self.adapter.obstacle_avoid()
            log.info("obstacle avoid switch -> code=%s state=%s", code, self.oa_state)
        self.camera.start()
        self._run = True
        for name, fn, hz in (("control", self._control_loop, float(self.cfg["gate"].get("rate_hz", 20))),
                             ("lidar", self._lidar_loop, float(self.cfg["lidar"].get("rate_hz", 10))),
                             ("statelog", self._statelog_loop, float(self.cfg["robot"].get("state_log_hz", 10)))):
            t = threading.Thread(target=self._periodic, args=(fn, hz), daemon=True, name=name)
            t.start()
            self._threads.append(t)
        log.info("RobotCore started allow_motion=%s adapter=%s", self.allow_motion, self.adapter.name)

    def stop(self) -> None:
        self._run = False
        for t in self._threads:
            t.join(timeout=1.0)
        if self.moving:
            try:
                self.adapter.stop_move()
            except Exception:
                pass
        self.camera.stop()
        self.adapter.stop()
        self.end_task()

    def _periodic(self, fn, hz: float) -> None:
        period = 1.0 / hz
        while self._run:
            t0 = mono()
            try:
                fn()
            except Exception as e:  # pragma: no cover
                log.exception("loop error in %s: %s", fn.__name__, e)
            dt = mono() - t0
            if dt < period:
                time.sleep(period - dt)

    # -- commands from primitives ------------------------------------------------------
    def set_cmd(self, vx: float, vy: float, vyaw: float, source: str) -> None:
        with self._lock:
            self._desired = Cmd(vx, vy, vyaw, mono(), source)

    def clear_cmd(self) -> None:
        with self._lock:
            self._desired = None

    def take_interrupt(self) -> Optional[str]:
        with self._lock:
            r, self.interrupt = self.interrupt, None
            return r

    def peek_interrupt(self) -> Optional[str]:
        return self.interrupt

    def set_pose(self, pose: str) -> None:
        if pose != self.pose:
            self._event("pose", pose=pose)
        self.pose = pose

    # -- latch / estop -----------------------------------------------------------------
    def latch(self, reason: str, detail: str = "") -> bool:
        t = mono()
        new = self.gate.set_latch(reason, t)
        if new:
            try:
                self.adapter.stop_move()
            except Exception as e:  # pragma: no cover
                log.error("stop_move failed: %s", e)
            self.moving = False
            self.last_sent = (0.0, 0.0, 0.0)   # nothing is sent after a latch; keep the state log truthful
            with self._lock:
                self.interrupt = reason
                self._desired = None
            self.last_stop_reason = reason
            self.motion_status = "stopped"
            self._event("latch", reason=reason, detail=detail)
            log.warning("LATCH %s %s", reason, detail)
        return new

    def release(self) -> tuple[bool, str]:
        if self.gate.latch is None:
            return True, "not_latched"
        if not self.allow_motion:
            return False, "motion_disabled"
        ok, why = self.remote.release_ok()
        if not ok:
            return False, why
        self.remote.clear()
        self.gate.release()
        self._event("release")
        return True, "released"

    # -- loops -------------------------------------------------------------------------
    def _control_loop(self) -> None:
        t = mono()
        low = self.adapter.low_state()
        if low is not None:
            self.remote.update_low(low.wireless_remote, low.t)
        wl = self.adapter.wireless()
        if wl is not None and wl.t > self._wl_t:
            self._wl_t = wl.t
            self.remote.update_wireless(wl.lx, wl.ly, wl.rx, wl.ry, wl.keys, wl.t)
        r = self.remote.poll(t)
        if r and self.gate.latch is None:
            self.latch(r, detail=str(self.remote.snapshot()))

        sport = self.adapter.sport_state()
        if sport is not None:
            self.odom.update(sport.x, sport.y, sport.yaw, sport.t)
        state_t = None if sport is None else sport.t
        if self.moving and (state_t is None or (t - state_t) > float(self.cfg["gate"].get("state_freshness_ms", 300)) / 1000.0 * 3):
            # state has been gone for a long time while we were moving -> hard stop and latch
            self.latch("state_lost")

        with self._lock:
            desired = self._desired
        front = sport.range_obstacle[0] if sport is not None else None
        d = self.gate.decide(desired, t, state_t, self.sectors, front)
        self.last_decision = d
        if desired is not None and not desired.is_zero() and not d.allowed and d.latched is None:
            # the gate refused a non-zero command: tell the job (it decides to end with interrupted_by=gate_*)
            if d.reason in ("min_range_ahead", "min_range_builtin", "min_range_behind", "min_range_left", "min_range_right",
                            "state_stale", "lidar_stale", "cmd_stale", "motion_disabled"):
                with self._lock:
                    if self.interrupt is None:
                        self.interrupt = f"gate_{d.reason}"
        self._send(d, t)

    def _send(self, d: GateDecision, t: float) -> None:
        c = d.cmd
        if not c.is_zero():
            self.adapter.move(c.vx, c.vy, c.vyaw)
            self.last_sent = (c.vx, c.vy, c.vyaw)
            self.moving = True
            self._last_nonzero_t = t
            self.motion_status = "moving"
        elif self.moving:
            # zero-velocity hold instead of StopMove (keeps the gait state; prior team finding)
            self.adapter.move(0.0, 0.0, 0.0)
            self.last_sent = (0.0, 0.0, 0.0)
            if t - self._last_nonzero_t >= float(self.cfg["gate"].get("zero_hold_s", 0.3)):
                self.moving = False
                self.motion_status = "stopped"
        else:
            self.last_sent = (0.0, 0.0, 0.0)
            self.motion_status = "stopped"

    def _lidar_loop(self) -> None:
        c = self.adapter.lidar_cloud()
        if c is None:
            return
        if self.sectors is not None and self.sectors.t == c.t:
            return
        self.sectors = self.lidar.push(c.xyz, c.t, c.frame_id)

    def _statelog_loop(self) -> None:
        w = self._robot_log
        if w is None:
            return
        sp = self.adapter.sport_state()
        d = self.last_decision
        w.write({
            "t": now(), "t_mono": mono(),
            "odom": {"x": round(self.odom.x, 3), "y": round(self.odom.y, 3), "yaw_deg": round(math.degrees(self.odom.yaw), 1), "traveled_m": round(self.odom.traveled_total, 3)},
            "vel": None if sp is None else [round(sp.vx, 3), round(sp.vy, 3), round(sp.vyaw, 3)],
            "mode": None if sp is None else [sp.mode, sp.gait],
            "free_space_m": None if self.sectors is None else {k: round(v, 2) for k, v in self.sectors.dist.items()},
            "lidar": None if self.sectors is None else {"frames": getattr(self.sectors, "frames", 1), "counts": self.sectors.counts,
                                                       "scan_plane": getattr(self.lidar, "last_plane", None)},
            "cmd_sent": list(self.last_sent),
            "gate": None if d is None else {"allowed": d.allowed, "reason": d.reason},
            "latch": self.gate.latch, "pose": self.pose,
            "job": self.job_status.get("id"), "job_tool": self.job_status.get("tool"), "job_phase": self.job_status.get("phase"),
        })

    def _event(self, kind: str, **kw) -> None:
        e = {"t": now(), "t_mono": mono(), "event": kind, **kw}
        self.events.append(e)
        if self._robot_log is not None:
            self._robot_log.write(e)

    # -- images ------------------------------------------------------------------------
    def store_image(self, jpeg: bytes) -> str:
        iid = uuid.uuid4().hex[:12]
        self.images[iid] = (jpeg, now())
        while len(self.images) > 80:
            self.images.popitem(last=False)
        return iid

    # -- task directory ----------------------------------------------------------------
    def begin_task(self, task_id: str, data_dir: Path) -> Path:
        self.end_task()
        self.task_id = task_id
        self.task_dir = Path(data_dir) / task_id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._robot_log = JsonlWriter(self.task_dir / "robot.jsonl")
        self._event("task_begin", task_id=task_id, allow_motion=self.allow_motion, adapter=self.adapter.name)
        self.camera.start_recording(self.task_dir / "cam")
        return self.task_dir

    def end_task(self) -> None:
        if self.task_id is None:
            return
        self.camera.stop_recording()
        self._event("task_end", task_id=self.task_id)
        w, self._robot_log = self._robot_log, None
        if w:
            w.close()
        self.task_id, self.task_dir = None, None

    # -- snapshots ---------------------------------------------------------------------
    def free_space(self) -> dict:
        if self.sectors is None:
            return {s: "unknown" for s in SECTORS}
        return self.sectors.vocab(self.cfg["lidar"])

    def state(self) -> dict:
        t = mono()
        sp = self.adapter.sport_state()
        low = self.adapter.low_state()
        img_age = self.camera.age_ms()
        return {
            "t": iso(),
            "age_ms": {
                "image": img_age,
                "lidar": None if self.sectors is None else int((t - self.sectors.t) * 1000),
                "odom": None if sp is None else int((t - sp.t) * 1000),
                "lowstate": None if low is None else int((t - low.t) * 1000),
            },
            "free_space": self.free_space(),
            "free_space_m": None if self.sectors is None else {k: round(v, 2) for k, v in self.sectors.dist.items()},
            "lidar": None if self.sectors is None else {"frames": getattr(self.sectors, "frames", 1), "counts": self.sectors.counts,
                                                       "scan_plane": getattr(self.lidar, "last_plane", None)},
            "motion": {"status": self.motion_status, "last_stop_reason": self.last_stop_reason, "pose": self.pose,
                       "job": self.job_status.get("id"), "job_tool": self.job_status.get("tool")},
            "odom": {"x": round(self.odom.x, 3), "y": round(self.odom.y, 3), "yaw_deg": round(math.degrees(self.odom.yaw), 1),
                     "traveled_total_m": round(self.odom.traveled_total, 2)},
            "safety": {"estop": self.gate.latch is not None, "latch": self.gate.latch, "allow_motion": self.allow_motion,
                       "last_gate_action": None if self.last_decision is None else self.last_decision.reason,
                       "obstacle_avoid": self.oa_state},
            "battery": None if low is None else {"soc": low.soc, "v": round(low.power_v, 1)},
            "task_id": self.task_id,
        }

    def health(self) -> dict:
        st = self.state()
        data_dir = Path(self.cfg["robot"].get("data_dir", "~/go2nav/data")).expanduser()
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            free = disk_free_fraction(data_dir)
        except Exception:
            free = None
        rates = {k: round(v.hz(), 1) for k, v in getattr(self.adapter, "rates", {}).items()} if getattr(self.adapter, "rates", None) else {}
        sp = self.adapter.sport_state()
        return {
            "ok": self.gate.latch is None and st["age_ms"]["odom"] is not None and st["age_ms"]["odom"] < 1000,
            "adapter": self.adapter.name, "allow_motion": self.allow_motion, "latch": self.gate.latch,
            "age_ms": st["age_ms"], "rates_hz": rates, "battery": st["battery"],
            "tilt_deg": None if sp is None else {"roll": round(math.degrees(sp.roll), 1), "pitch": round(math.degrees(sp.pitch), 1)},
            "disk_free_fraction": None if free is None else round(free, 3),
            "camera": {"frames_ok": self.camera.frames_ok, "failures": self.camera.failures, "recording": self.camera.recording()},
            "task_id": self.task_id, "uptime_s": round(now() - self.started_at, 1),
            "obstacle_avoid": self.oa_state,
            "classic_walk_code": self.classic_walk_code,
            "mode_gait": None if sp is None else [sp.mode, sp.gait],
        }

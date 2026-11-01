"""L1 deterministic gate (design 9, L1). Pure logic, no I/O, so it is unit-testable.

Inputs each tick: desired command (with timestamp), freshness of state/lidar, sector distances,
remote latch, allow_motion. Output: the command actually allowed (possibly zero) and the reason.
Latches are sticky; release only via `release()` which the CLI calls (never the model).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .lidar import Sectors


@dataclass
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    t: float = 0.0          # monotonic time the command was issued by the primitive
    source: str = ""        # primitive name

    def is_zero(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.vyaw == 0.0


@dataclass
class GateDecision:
    cmd: Cmd
    allowed: bool
    reason: str            # "ok" or why the command was zeroed / modified
    latched: Optional[str] = None
    modified: bool = False


class Gate:
    LATCH_REASONS = ("estop_remote", "operator_takeover", "remote_lost", "cli", "state_lost")

    def __init__(self, cfg: dict, allow_motion: bool = False):
        self.cfg = cfg
        self.allow_motion = bool(allow_motion)
        self.latch: Optional[str] = None
        self.latch_t: Optional[float] = None
        self.last_reason = "idle"

    # -- latch management -------------------------------------------------------------
    def set_latch(self, reason: str, t: float) -> bool:
        if self.latch is None:
            self.latch, self.latch_t = reason, t
            return True
        return False

    def release(self) -> None:
        self.latch, self.latch_t = None, None

    # -- per-tick decision ------------------------------------------------------------
    def decide(self, desired: Optional[Cmd], now: float, state_t: Optional[float], sectors: Optional[Sectors],
               range_obstacle_front: Optional[float] = None) -> GateDecision:
        zero = Cmd(0.0, 0.0, 0.0, now, desired.source if desired else "")
        if self.latch:
            return self._d(zero, False, f"latched:{self.latch}", self.latch)
        if not self.allow_motion:
            return self._d(zero, False, "motion_disabled")
        if desired is None or desired.is_zero():
            return self._d(zero, True, "ok")
        if (now - desired.t) * 1000.0 > float(self.cfg.get("cmd_freshness_ms", 200)):
            return self._d(zero, False, "cmd_stale")
        if state_t is None or (now - state_t) * 1000.0 > float(self.cfg.get("state_freshness_ms", 300)):
            return self._d(zero, False, "state_stale")

        vx, vy, vyaw = desired.vx, desired.vy, desired.vyaw
        modified = False
        mx, my, mw = float(self.cfg.get("max_vx", 0.5)), float(self.cfg.get("max_vy", 0.3)), float(self.cfg.get("max_vyaw", 0.8))
        c_vx, c_vy, c_vw = _clamp(vx, mx), _clamp(vy, my), _clamp(vyaw, mw)
        if (c_vx, c_vy, c_vw) != (vx, vy, vyaw):
            modified = True
            vx, vy, vyaw = c_vx, c_vy, c_vw
        min_w = float(self.cfg.get("min_vyaw", 0.25))
        if vyaw != 0.0 and abs(vyaw) < min_w:
            vyaw = min_w if vyaw > 0 else -min_w
            modified = True

        if vx > 0.0:
            lidar_ms = float(self.cfg.get("lidar_freshness_ms", 500))
            if sectors is None or (now - sectors.t) * 1000.0 > lidar_ms:
                return self._d(zero, False, "lidar_stale")
            min_ahead = float(self.cfg.get("min_range_ahead_m", 0.6))
            if sectors.dist.get("ahead", 99.0) < min_ahead:
                return self._d(zero, False, "min_range_ahead")
            if self.cfg.get("use_builtin_range_obstacle", True) and range_obstacle_front is not None \
                    and 0.0 < range_obstacle_front < min_ahead:
                return self._d(zero, False, "min_range_builtin")
        if vx < 0.0 and sectors is not None:
            if sectors.dist.get("behind", 99.0) < float(self.cfg.get("min_range_ahead_m", 0.6)):
                return self._d(zero, False, "min_range_behind")
        side_min = float(self.cfg.get("min_range_side_m", 0.0))
        if side_min > 0 and vyaw != 0.0 and sectors is not None:
            side = "left" if vyaw > 0 else "right"
            if sectors.dist.get(side, 99.0) < side_min:
                return self._d(zero, False, f"min_range_{side}")
        out = Cmd(vx, vy, vyaw, desired.t, desired.source)
        return self._d(out, True, "ok" if not modified else "clamped", None, modified)

    def _d(self, cmd, allowed, reason, latched=None, modified=False) -> GateDecision:
        self.last_reason = reason
        return GateDecision(cmd=cmd, allowed=allowed, reason=reason, latched=latched, modified=modified)


def _clamp(v: float, m: float) -> float:
    return max(-m, min(m, v))

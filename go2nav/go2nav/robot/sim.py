"""Kinematic simulator with a synthetic building (for offline tests of primitives/server/brain).

World: axis-aligned rectangular corridors described as free-space rectangles. The LiDAR is simulated
by ray casting in the body frame (returned already in the body frame; SimAdapter's "lidar frame" == body frame,
so tests should run LidarProcessor with an identity mount). The camera returns a synthetic JPEG with the pose drawn.
Remote-controller bytes can be injected via `set_remote(keys=..., lx=...)`.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from typing import Optional

import numpy as np

from ..common import mono
from .adapter import RobotAdapter, SportSample, LowSample, WirelessSample, CloudSample

# Default map: main corridor along +x (width 2 m), a side corridor opening to the left at x=6..8,
# dead end at x=12. Rectangles are (xmin, xmax, ymin, ymax) of FREE space.
DEFAULT_MAP = [
    (-2.0, 12.0, -1.0, 1.0),      # main corridor
    (6.0, 8.0, 1.0, 10.0),        # side corridor to the left (opening_left when passing x≈6-8)
]


class SimAdapter(RobotAdapter):
    name = "sim"

    def __init__(self, cfg: Optional[dict] = None, allow_motion: bool = True, world=None, dt: float = 0.02, realtime: bool = True,
                 time_scale: float = 1.0):
        """time_scale > 1 makes the simulated robot move faster than real time (tests)."""
        self.cfg = cfg or {}
        self.time_scale = float(time_scale)
        self.allow_motion = allow_motion
        self.world = world or DEFAULT_MAP
        self.dt = dt
        self.realtime = realtime
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = self.vy = self.vyaw = 0.0
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_t = -1e9
        self.cmd_hold_s = 0.5         # firmware-like deadman: stop if no Move for this long
        self._remote = bytearray(40)
        self._remote[0] = 0x0A        # fake header
        self._lock = threading.Lock()
        self._run = False
        self._thread: Optional[threading.Thread] = None
        self.moves: list[tuple] = []  # log of Move() calls (for tests)
        self.stop_moves = 0
        self.actions: list[tuple[float, str]] = []   # log of sport_action() calls (for tests)
        self.action_codes: dict[str, int] = {}       # test hook: name -> return code (default 0)
        self.oa = True
        self.camera_fail = False
        self.frozen_sensors = False    # when True, samples stop updating (freshness tests)
        self.yaw_deadband = 0.0        # |vyaw| below this is ignored (firmware deadband seen on the robot near obstacles)
        self.vx_deadband = 0.0         # |vx| below this is ignored
        self._sport_t = mono()
        self._low_t = mono()
        self._cloud_t = mono()
        self.rates = {}

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sim")
        self._thread.start()

    def stop(self) -> None:
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._run:
            self.step(self.dt)
            if self.realtime:
                time.sleep(self.dt / self.time_scale)

    def step(self, dt: float) -> None:
        with self._lock:
            if mono() - self._cmd_t > self.cmd_hold_s:
                self._cmd = (0.0, 0.0, 0.0)
            vx, vy, vyaw = self._cmd
            if abs(vyaw) < self.yaw_deadband:
                vyaw = 0.0
            if abs(vx) < self.vx_deadband:
                vx = 0.0
            self.vx, self.vy, self.vyaw = vx, vy, vyaw
            nx = self.x + (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * dt
            ny = self.y + (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * dt
            if self._free(nx, ny):
                self.x, self.y = nx, ny
            else:
                self.vx = self.vy = 0.0
            self.yaw = (self.yaw + vyaw * dt + math.pi) % (2 * math.pi) - math.pi
            if not self.frozen_sensors:
                t = mono()
                self._sport_t = self._low_t = self._cloud_t = t

    # -- world -------------------------------------------------------------------------
    def _free(self, x: float, y: float) -> bool:
        return any(a <= x <= b and c <= y <= d for a, b, c, d in self.world)

    def raycast(self, bearing_body_rad: float, max_r: float = 12.0, step: float = 0.05) -> float:
        a = self.yaw + bearing_body_rad
        r = 0.0
        while r < max_r:
            r += step
            if not self._free(self.x + r * math.cos(a), self.y + r * math.sin(a)):
                return r
        return max_r

    # -- samples -----------------------------------------------------------------------
    def sport_state(self):
        with self._lock:
            front = self.raycast(0.0)
            return SportSample(t=self._sport_t, x=self.x, y=self.y, yaw=self.yaw, vx=self.vx, vy=self.vy, vyaw=self.vyaw,
                               range_obstacle=(front, self.raycast(math.pi / 2), self.raycast(-math.pi / 2), self.raycast(math.pi)))

    def low_state(self):
        with self._lock:
            return LowSample(t=self._low_t, wireless_remote=bytes(self._remote), soc=77, power_v=28.5)

    def wireless(self):
        return None

    def lidar_cloud(self):
        with self._lock:
            t = self._cloud_t
        pts = []
        for i in range(360):
            b = math.radians(i - 180)
            r = self.raycast(b)
            if r < 12.0:
                for z in (0.0, 0.2):
                    pts.append((r * math.cos(b), r * math.sin(b), z))
        xyz = np.array(pts, dtype=np.float32).reshape(-1, 3)
        return CloudSample(t=t, xyz=xyz, frame_id="sim_body")

    def get_image(self) -> Optional[bytes]:
        if self.camera_fail:
            return None
        import cv2
        img = np.full((720, 1280, 3), 200, dtype=np.uint8)
        txt = f"sim x={self.x:.2f} y={self.y:.2f} yaw={math.degrees(self.yaw):.0f} ahead={self.raycast(0.0):.1f}m"
        cv2.putText(img, txt, (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return bytes(buf) if ok else None

    # -- commands ----------------------------------------------------------------------
    def move(self, vx, vy, vyaw) -> int:
        if not self.allow_motion:
            return -1
        with self._lock:
            self._cmd = (float(vx), float(vy), float(vyaw))
            self._cmd_t = mono()
        self.moves.append((mono(), float(vx), float(vy), float(vyaw)))
        return 0

    def stop_move(self) -> int:
        if not self.allow_motion:
            return -1
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)
        self.stop_moves += 1
        return 0

    def set_obstacle_avoid(self, on: bool) -> int:
        self.oa = bool(on)
        return 0

    def set_classic_walk(self, on: bool) -> int:
        self.classic_walk = bool(on)
        return 0

    def sport_action(self, name: str) -> int:
        if not self.allow_motion:
            return -1
        self.actions.append((mono(), name))
        return int(self.action_codes.get(name, 0))

    def obstacle_avoid(self):
        return self.oa

    # -- test hooks --------------------------------------------------------------------
    def set_remote(self, keys: int = 0, lx: float = 0.0, ly: float = 0.0, rx: float = 0.0, ry: float = 0.0) -> None:
        with self._lock:
            struct.pack_into("<H", self._remote, 2, keys)
            struct.pack_into("<f", self._remote, 4, lx)
            struct.pack_into("<f", self._remote, 8, rx)
            struct.pack_into("<f", self._remote, 12, ry)
            struct.pack_into("<f", self._remote, 20, ly)

    def teleport(self, x: float, y: float, yaw_deg: float = 0.0) -> None:
        with self._lock:
            self.x, self.y, self.yaw = x, y, math.radians(yaw_deg)

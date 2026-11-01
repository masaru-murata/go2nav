"""Robot adapter: the only place that talks to the Go2 (unitree_sdk2py / DDS).

Two implementations:
  - UnitreeAdapter : real robot (imports unitree_sdk2py lazily so the rest of the package works without it)
  - SimAdapter     : kinematic simulation with a synthetic corridor (go2nav/robot/sim.py) for offline tests

Every sample carries `t` (monotonic receive time) so freshness can be checked by the gate.
Commands are never sent unless `allow_motion` was enabled explicitly.
"""
from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..common import mono

log = logging.getLogger("go2nav.adapter")


@dataclass
class SportSample:
    t: float                      # monotonic receive time
    x: float
    y: float
    yaw: float                    # rad, from imu rpy[2]
    vx: float
    vy: float
    vyaw: float
    range_obstacle: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # firmware front/left/right/back (order to verify)
    mode: int = 0
    gait: int = 0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class LowSample:
    t: float
    wireless_remote: bytes        # 40 bytes
    soc: int = -1                 # battery percent
    power_v: float = 0.0
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class WirelessSample:
    t: float
    lx: float
    ly: float
    rx: float
    ry: float
    keys: int


@dataclass
class CloudSample:
    t: float
    xyz: np.ndarray               # (N, 3) float32 in the LiDAR frame
    frame_id: str = ""


class RobotAdapter:
    """Interface. All getters return the latest sample or None."""

    name = "base"

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def sport_state(self) -> Optional[SportSample]: ...
    def low_state(self) -> Optional[LowSample]: ...
    def wireless(self) -> Optional[WirelessSample]: ...
    def lidar_cloud(self) -> Optional[CloudSample]: ...
    def get_image(self) -> Optional[bytes]:
        """Fresh JPEG from the front camera (blocking RPC)."""
        ...
    def move(self, vx: float, vy: float, vyaw: float) -> int: ...
    def stop_move(self) -> int: ...
    def set_obstacle_avoid(self, on: bool) -> int: ...
    def obstacle_avoid(self) -> Optional[bool]: ...
    def sport_action(self, name: str) -> int:
        """Call a parameterless SportClient action by method name (StandDown, StandUp, BalanceStand, Sit, RiseSit, Hello, Stretch...)."""
        ...
    def set_classic_walk(self, on: bool) -> int:
        """SportClient.ClassicWalk(flag): classic (periodic trot) gait vs the default learned gait."""
        ...


# --------------------------------------------------------------------------------------
# PointCloud2 decoding (shared by UnitreeAdapter and tests)
# --------------------------------------------------------------------------------------
_PF_DTYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def decode_pointcloud2(fields, point_step: int, data, is_bigendian: bool = False) -> np.ndarray:
    """Return (N,3) float32 xyz from a PointCloud2-like message (fields with name/offset/datatype)."""
    off = {}
    dt = {}
    for f in fields:
        off[f.name] = int(f.offset)
        dt[f.name] = _PF_DTYPE.get(int(f.datatype), "f4")
    if not all(k in off for k in ("x", "y", "z")):
        raise ValueError(f"pointcloud has no xyz fields: {list(off)}")
    buf = bytes(data) if not isinstance(data, (bytes, bytearray, memoryview)) else data
    n = len(buf) // int(point_step)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    endian = ">" if is_bigendian else "<"
    arr = np.frombuffer(buf, dtype=np.uint8, count=n * int(point_step)).reshape(n, int(point_step))
    cols = []
    for k in ("x", "y", "z"):
        d = np.dtype(endian + dt[k])
        o = off[k]
        cols.append(arr[:, o:o + d.itemsize].copy().view(d).reshape(n).astype(np.float32))
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------------------
# Real robot
# --------------------------------------------------------------------------------------
class UnitreeAdapter(RobotAdapter):
    name = "unitree"

    def __init__(self, cfg: dict, allow_motion: bool = False):
        self.cfg = cfg
        self.allow_motion = allow_motion
        self._lock = threading.Lock()
        self._sport: Optional[SportSample] = None
        self._low: Optional[LowSample] = None
        self._wl: Optional[WirelessSample] = None
        self._cloud: Optional[CloudSample] = None
        self._subs = []
        self._sport_client = None
        self._action_client = None    # second SportClient with a long timeout: gesture RPCs block until the motion ends (~5 s)
        self._video_client = None
        self._oa_client = None
        self._video_lock = threading.Lock()
        self.rates = {"sport": _Rate(), "low": _Rate(), "wireless": _Rate(), "lidar": _Rate()}

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_, WirelessController_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from unitree_sdk2py.go2.video.video_client import VideoClient

        iface = self.cfg.get("network_interface") or None
        ChannelFactoryInitialize(0, iface)
        t = self.cfg["topics"]

        def sub(topic, typ, handler):
            s = ChannelSubscriber(topic, typ)
            s.Init(handler, 10)
            self._subs.append(s)

        sub(t["sport_state"], SportModeState_, self._on_sport)
        sub(t["low_state"], LowState_, self._on_low)
        sub(t["wireless"], WirelessController_, self._on_wireless)
        sub(t["lidar"], PointCloud2_, self._on_cloud)

        self._video_client = VideoClient()
        self._video_client.SetTimeout(3.0)
        self._video_client.Init()

        if self.allow_motion:
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            self._sport_client = SportClient()
            self._sport_client.SetTimeout(3.0)
            self._sport_client.Init()
            self._action_client = SportClient()
            self._action_client.SetTimeout(float(self.cfg.get("sport_action_timeout_s", 20.0)))
            self._action_client.Init()
            try:
                from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
                self._oa_client = ObstaclesAvoidClient()
                self._oa_client.SetTimeout(3.0)
                self._oa_client.Init()
            except Exception as e:  # pragma: no cover - depends on firmware
                log.warning("ObstaclesAvoidClient unavailable: %s", e)
                self._oa_client = None
        log.info("UnitreeAdapter started (iface=%s, allow_motion=%s)", iface, self.allow_motion)

    def stop(self) -> None:
        for s in self._subs:
            try:
                s.Close()
            except Exception:
                pass
        self._subs = []

    # -- DDS handlers ----------------------------------------------------------------
    def _on_sport(self, msg) -> None:
        try:
            s = SportSample(
                t=mono(),
                x=float(msg.position[0]), y=float(msg.position[1]),
                yaw=float(msg.imu_state.rpy[2]),
                vx=float(msg.velocity[0]), vy=float(msg.velocity[1]), vyaw=float(msg.yaw_speed),
                range_obstacle=tuple(float(v) for v in msg.range_obstacle),
                mode=int(msg.mode), gait=int(msg.gait_type),
                pitch=float(msg.imu_state.rpy[1]), roll=float(msg.imu_state.rpy[0]),
            )
        except Exception as e:  # pragma: no cover
            log.debug("sport parse error: %s", e)
            return
        with self._lock:
            self._sport = s
        self.rates["sport"].tick()

    def _on_low(self, msg) -> None:
        try:
            s = LowSample(
                t=mono(),
                wireless_remote=bytes(msg.wireless_remote),
                soc=int(msg.bms_state.soc),
                power_v=float(msg.power_v),
                rpy=tuple(float(v) for v in msg.imu_state.rpy),
            )
        except Exception as e:  # pragma: no cover
            log.debug("low parse error: %s", e)
            return
        with self._lock:
            self._low = s
        self.rates["low"].tick()

    def _on_wireless(self, msg) -> None:
        s = WirelessSample(t=mono(), lx=float(msg.lx), ly=float(msg.ly), rx=float(msg.rx), ry=float(msg.ry), keys=int(msg.keys))
        with self._lock:
            self._wl = s
        self.rates["wireless"].tick()

    def _on_cloud(self, msg) -> None:
        try:
            xyz = decode_pointcloud2(msg.fields, msg.point_step, msg.data, bool(msg.is_bigendian))
            fid = str(getattr(msg.header, "frame_id", ""))
        except Exception as e:  # pragma: no cover
            log.debug("cloud parse error: %s", e)
            return
        with self._lock:
            self._cloud = CloudSample(t=mono(), xyz=xyz, frame_id=fid)
        self.rates["lidar"].tick()

    # -- getters ---------------------------------------------------------------------
    def sport_state(self):
        with self._lock:
            return self._sport

    def low_state(self):
        with self._lock:
            return self._low

    def wireless(self):
        with self._lock:
            return self._wl

    def lidar_cloud(self):
        with self._lock:
            return self._cloud

    def get_image(self) -> Optional[bytes]:
        if self._video_client is None:
            return None
        with self._video_lock:
            code, data = self._video_client.GetImageSample()
        if code != 0 or not data:
            log.warning("GetImageSample failed code=%s", code)
            return None
        return bytes(data)

    # -- commands --------------------------------------------------------------------
    def move(self, vx: float, vy: float, vyaw: float) -> int:
        if not self.allow_motion or self._sport_client is None:
            return -1
        return int(self._sport_client.Move(float(vx), float(vy), float(vyaw)))

    def stop_move(self) -> int:
        if not self.allow_motion or self._sport_client is None:
            return -1
        return int(self._sport_client.StopMove())

    SPORT_ACTIONS = ("StandDown", "StandUp", "BalanceStand", "Sit", "RiseSit", "Hello", "Stretch", "RecoveryStand")

    def set_classic_walk(self, on: bool) -> int:
        if not self.allow_motion or self._action_client is None:
            return -1
        try:
            code = int(self._action_client.ClassicWalk(bool(on)))
        except Exception as e:  # pragma: no cover
            log.warning("ClassicWalk(%s) failed: %s", on, e)
            return -2
        log.info("ClassicWalk(%s) -> code=%s", on, code)
        return code

    def sport_action(self, name: str) -> int:
        """Blocking: the firmware answers when the motion has finished (Hello ~4.7 s measured 2026-10-30)."""
        if not self.allow_motion or self._action_client is None:
            return -1
        if name not in self.SPORT_ACTIONS:
            return -3
        try:
            code = int(getattr(self._action_client, name)())
        except Exception as e:  # pragma: no cover
            log.warning("sport action %s failed: %s", name, e)
            return -2
        log.info("sport action %s -> code=%s", name, code)
        return code

    def set_obstacle_avoid(self, on: bool) -> int:
        if not self.allow_motion or self._oa_client is None:
            return -1
        try:
            return int(self._oa_client.SwitchSet(bool(on)))
        except Exception as e:  # pragma: no cover
            log.warning("obstacle avoid switch failed: %s", e)
            return -2

    def obstacle_avoid(self) -> Optional[bool]:
        if self._oa_client is None:
            return None
        try:
            code, on = self._oa_client.SwitchGet()
            return bool(on) if code == 0 else None
        except Exception:  # pragma: no cover
            return None


class _Rate:
    """Cheap rolling rate estimator (Hz over the last ~2 s)."""

    def __init__(self):
        self._ts: list[float] = []
        self._lock = threading.Lock()

    def tick(self) -> None:
        t = mono()
        with self._lock:
            self._ts.append(t)
            cut = t - 2.0
            while self._ts and self._ts[0] < cut:
                self._ts.pop(0)

    def hz(self) -> float:
        with self._lock:
            if len(self._ts) < 2:
                return 0.0
            span = self._ts[-1] - self._ts[0]
            return (len(self._ts) - 1) / span if span > 0 else 0.0

"""Remote-controller decoding and monitoring (design 9.1).

The `/wirelesscontroller` topic streams only while the operator touches the remote, so continuous
monitoring reads `lowstate.wireless_remote` (40 bytes, Unitree xRockerBtnDataStruct):
  head[0:2], keys[2:4] (LE16), lx[4:8], rx[8:12], ry[12:16], L2[16:20], ly[20:24]  (float32 LE)
Bit map (measured, docs/inventory_20261026.md): R1=0 L1=1 START=2 SELECT=3 R2=4 L2=5 F1=6 F3=7 A=8 B=9 X=10 Y=11 UP=12 RIGHT=13 DOWN=14 LEFT=15
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from ..common import mono


@dataclass
class RemoteState:
    t: float = 0.0                  # time of the sample this was decoded from
    keys: int = 0
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    buttons: list[str] = field(default_factory=list)

    def stick_max(self) -> float:
        return max(abs(self.lx), abs(self.ly), abs(self.rx), abs(self.ry))


def decode_wireless_remote(raw: bytes, cfg: dict) -> RemoteState:
    bits = cfg["button_bits"]
    offs = cfg.get("stick_offsets", {"lx": 4, "rx": 8, "ry": 12, "ly": 20})
    raw = bytes(raw)
    keys = struct.unpack_from("<H", raw, 2)[0] if len(raw) >= 4 else 0

    def f(o):
        if len(raw) < o + 4:
            return 0.0
        v = struct.unpack_from("<f", raw, o)[0]
        if v != v or abs(v) > 2.0:  # NaN / garbage guard
            return 0.0
        return float(v)

    st = RemoteState(t=mono(), keys=keys, lx=f(offs["lx"]), ly=f(offs["ly"]), rx=f(offs["rx"]), ry=f(offs["ry"]))
    st.buttons = [name for name, b in bits.items() if keys & (1 << int(b))]
    return st


class RemoteMonitor:
    """Turns remote samples into latch events: estop_remote / operator_takeover / remote_lost.

    Feed it with `update_low(raw_bytes, t)` (continuous) and optionally `update_wireless(...)` (event-driven).
    Call `poll(now)` from the 20 Hz loop; it returns a latch reason or None. Latches are sticky until `clear()`.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bits = cfg["button_bits"]
        self.estop_mask = 0
        for name in cfg.get("estop_buttons", ["SELECT", "L1"]):
            self.estop_mask |= 1 << int(self.bits[name])
        self.deadzone = float(cfg.get("stick_deadzone", 0.15))
        self.hold = int(cfg.get("stick_hold_samples", 2))
        self.timeout_s = float(cfg.get("lowstate_timeout_ms", 500)) / 1000.0
        self.state = RemoteState()
        self.last_low_t: Optional[float] = None
        self._stick_count = 0
        self.latch: Optional[str] = None
        self.latch_t: Optional[float] = None
        self.events: list[dict] = []

    # -- inputs -----------------------------------------------------------------------
    def update_low(self, raw: bytes, t: float) -> None:
        self.last_low_t = t
        st = decode_wireless_remote(raw, self.cfg)
        st.t = t
        self._ingest(st)

    def update_wireless(self, lx: float, ly: float, rx: float, ry: float, keys: int, t: float) -> None:
        st = RemoteState(t=t, keys=keys, lx=lx, ly=ly, rx=rx, ry=ry)
        st.buttons = [n for n, b in self.bits.items() if keys & (1 << int(b))]
        self._ingest(st)

    def _ingest(self, st: RemoteState) -> None:
        self.state = st
        if (st.keys & self.estop_mask) == self.estop_mask and self.estop_mask:
            self._set("estop_remote", st.t)
        if st.stick_max() > self.deadzone:
            self._stick_count += 1
            if self._stick_count >= self.hold:
                self._set("operator_takeover", st.t)
        else:
            self._stick_count = 0

    # -- polling ----------------------------------------------------------------------
    def poll(self, now: Optional[float] = None) -> Optional[str]:
        now = mono() if now is None else now
        if self.last_low_t is not None and now - self.last_low_t > self.timeout_s:
            self._set("remote_lost", now)
        return self.latch

    def _set(self, reason: str, t: float) -> None:
        if self.latch is None:
            self.latch = reason
            self.latch_t = t
            self.events.append({"t": t, "event": reason, "keys": self.state.keys, "buttons": list(self.state.buttons),
                                "sticks": [self.state.lx, self.state.ly, self.state.rx, self.state.ry]})

    def estop_held(self) -> bool:
        return self.estop_mask != 0 and (self.state.keys & self.estop_mask) == self.estop_mask

    def sticks_neutral(self) -> bool:
        return self.state.stick_max() <= self.deadzone

    def release_ok(self, now: Optional[float] = None) -> tuple[bool, str]:
        """Conditions to release a latch (design 9.1): sticks neutral, E-stop buttons released, lowstate fresh."""
        now = mono() if now is None else now
        if not self.sticks_neutral():
            return False, "sticks_not_neutral"
        if self.estop_held():
            return False, "estop_buttons_held"
        if self.last_low_t is not None and now - self.last_low_t > self.timeout_s:
            return False, "lowstate_stale"
        return True, "ok"

    def clear(self) -> None:
        self.latch = None
        self.latch_t = None
        self._stick_count = 0

    def snapshot(self) -> dict:
        age = None if self.last_low_t is None else round((mono() - self.last_low_t) * 1000)
        return {
            "age_ms": age,
            "keys": self.state.keys,
            "buttons": list(self.state.buttons),
            "sticks": {"lx": round(self.state.lx, 3), "ly": round(self.state.ly, 3), "rx": round(self.state.rx, 3), "ry": round(self.state.ry, 3)},
            "latch": self.latch,
            "estop_buttons": self.cfg.get("estop_buttons"),
        }

"""LiDAR point cloud -> 8-sector minimum planar distance -> fixed vocabulary (design 5.2).

Sectors (bearing measured from the body x axis, positive to the left):
  ahead 0, ahead_left +45, left +90, behind_left +135, behind 180, behind_right -135, right -90, ahead_right -45
`ahead` is special: it is the robot's path band (|y| <= ahead_half_width_m, x > 0) instead of a 45 deg fan, so that
corridor walls do not cap the forward distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SECTORS = ["ahead", "ahead_left", "left", "behind_left", "behind", "behind_right", "right", "ahead_right"]
SECTOR_DEG = {"ahead": 0.0, "ahead_left": 45.0, "left": 90.0, "behind_left": 135.0, "behind": 180.0,
              "behind_right": -135.0, "right": -90.0, "ahead_right": -45.0}
FORWARD_SECTORS = {"ahead", "ahead_left", "ahead_right"}


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


@dataclass
class Sectors:
    t: float
    dist: dict[str, float]                 # planar min distance per sector (m), capped at r_max (robust k-th smallest)
    median: dict[str, float] = field(default_factory=dict)   # median planar distance per sector (opening detection)
    counts: dict[str, int] = field(default_factory=dict)
    frame_id: str = ""
    frames: int = 1

    def vocab(self, cfg: dict) -> dict[str, str]:
        """A sector is 'unknown' when it has too few points to say anything, or when no dense bin was found with only a
        handful of points (the L1 does not see behind the robot well; 'open' must not be claimed from a few stray returns)."""
        min_pts = int(cfg.get("min_points_per_sector", 3))
        known = int(cfg.get("open_requires_points", 20))
        out = {}
        for s, d in self.dist.items():
            n = self.counts.get(s, 0)
            if n < min_pts or (d >= float(cfg.get("r_max_m", 12.0)) and n < known):
                out[s] = "unknown"
            else:
                out[s] = distance_to_vocab(s, d, cfg)
        return out


def distance_to_vocab(sector: str, d: float, cfg: dict) -> str:
    blocked = float(cfg.get("blocked_below_m", 1.0))
    near = float(cfg.get("near_below_m", 2.0))
    cap = float(cfg.get("open_cap_m", 8.0))
    if d < blocked:
        kind = "blocked" if sector in FORWARD_SECTORS else "wall"
        return f"{kind}_{_fmt(d)}m"
    if d < near:
        return f"near_{_fmt(d)}m"
    n = int(min(cap, math.floor(d)))
    return f"open_{n}m"


def _fmt(d: float) -> str:
    """0.83 -> '0p8' (one decimal, 'p' for the point)."""
    v = max(0.0, round(d, 1))
    return f"{v:.1f}".replace(".", "p")


class LidarProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        r, p, y = cfg.get("mount_rpy", [0.0, 0.0, 0.0])
        self.R = rpy_to_matrix(float(r), float(p), float(y))
        self.tvec = np.array(cfg.get("mount_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.z_min = float(cfg.get("z_min_m", -0.2))
        self.z_max = float(cfg.get("z_max_m", 0.6))
        self.r_min = float(cfg.get("r_min_m", 0.15))
        bx = cfg.get("body_box", None)   # [x_min, x_max, y_half]: points inside are the robot itself
        self.body_box = None if not bx else (float(bx[0]), float(bx[1]), float(bx[2]))
        self.r_max = float(cfg.get("r_max_m", 12.0))
        self.half = float(cfg.get("sector_half_width_deg", 22.5))
        self.ahead_half_width = float(cfg.get("ahead_half_width_m", 0.4))
        self.min_pts = int(cfg.get("min_points_per_sector", 3))
        self.kth = int(cfg.get("kth_smallest", 3))
        self.bin_m = float(cfg.get("bin_m", 0.2))
        self.min_per_bin = int(cfg.get("min_points_per_bin", 1))

    def to_body(self, xyz_lidar: np.ndarray) -> np.ndarray:
        if xyz_lidar.size == 0:
            return xyz_lidar.reshape(0, 3)
        return xyz_lidar @ self.R.T + self.tvec

    # -- scan-plane artifact ------------------------------------------------------------------------------------
    # The Go2's L1 returns a dense disc of spurious points lying on the sensor's own scan plane (tilted ~15 deg in the
    # levelled cloud frame) out to ~1 m in every direction (measured 2026-10-29: 93% of near points within 5 cm of one
    # plane). Real obstacles produce points off that plane, so we fit the plane online and drop its inliers near the robot.
    def remove_scan_plane(self, xyz: np.ndarray) -> tuple[np.ndarray, Optional[dict]]:
        cfg = self.cfg.get("scan_plane") or {}
        if not cfg.get("enabled", False) or xyz.size == 0:
            return xyz, None
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        z = xyz[:, 2]
        fit_r = (r >= float(cfg.get("fit_r_min_m", 0.35))) & (r <= float(cfg.get("fit_r_max_m", 0.9)))
        fit = fit_r & (z >= 0.0) & (z <= 1.0)
        if self.body_box is not None:
            x0, x1, yh = self.body_box
            fit &= ~((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (np.abs(xyz[:, 1]) <= yh))
        coef = None
        info: dict = {"fit_n": int(fit.sum())}
        if fit.sum() >= int(cfg.get("fit_min_points", 150)):
            A = np.c_[xyz[fit, 0], xyz[fit, 1], np.ones(int(fit.sum()), dtype=np.float32)]
            c, *_ = np.linalg.lstsq(A, z[fit], rcond=None)
            res = z[fit] - A @ c
            tol = float(cfg.get("inlier_m", 0.06))
            inl = float(np.mean(np.abs(res) < tol))
            tilt = math.degrees(math.acos(1.0 / math.sqrt(1.0 + c[0] ** 2 + c[1] ** 2)))
            info.update({"a": round(float(c[0]), 3), "b": round(float(c[1]), 3), "c": round(float(c[2]), 3), "inlier_frac": round(inl, 2), "tilt_deg": round(tilt, 1)})
            if inl >= float(cfg.get("min_inlier_frac", 0.7)) and tilt <= float(cfg.get("max_tilt_deg", 35.0)):
                coef = c
                self._plane_cal = (float(c[0]), float(c[1]), float(c[2]))
        if coef is None:
            cal = getattr(self, "_plane_cal", None) or cfg.get("calibrated")
            if cal:
                coef = np.array(cal, dtype=np.float32)
                info["source"] = "calibrated"
            else:
                return xyz, info
        else:
            info["source"] = "fit"
        d = np.abs(z - (coef[0] * xyz[:, 0] + coef[1] * xyz[:, 1] + coef[2]))
        # the plane's angular error grows with range: widen the removal slab with r
        tol = float(cfg.get("inlier_m", 0.06)) + float(cfg.get("inlier_per_m", 0.0)) * r
        drop = (d < tol) & (r < float(cfg.get("remove_r_max_m", 1.2)))
        info["removed"] = int(drop.sum())
        return xyz[~drop], info

    def sectors_from_body(self, xyz: np.ndarray, t: float, frame_id: str = "") -> Sectors:
        xyz, plane = self.remove_scan_plane(xyz)
        self.last_plane = plane
        dist = {s: self.r_max for s in SECTORS}
        median = {s: self.r_max for s in SECTORS}
        counts = {s: 0 for s in SECTORS}
        if xyz.size:
            z = xyz[:, 2]
            m = (z >= self.z_min) & (z <= self.z_max) & np.isfinite(xyz).all(axis=1)
            p = xyz[m]
            if p.size:
                r = np.hypot(p[:, 0], p[:, 1])
                m2 = (r >= self.r_min) & (r <= self.r_max)
                if self.body_box is not None:
                    x0, x1, yh = self.body_box
                    m2 &= ~((p[:, 0] >= x0) & (p[:, 0] <= x1) & (np.abs(p[:, 1]) <= yh))
                p, r = p[m2], r[m2]
                if p.size:
                    bearing = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
                    for s in SECTORS:
                        if s == "ahead" and self.ahead_half_width > 0:
                            # "ahead" is the robot's own path: a band of +-ahead_half_width around the x axis, not an angular fan
                            # (in a 2 m corridor a 45 deg fan would always report the side walls at ~2.6 m)
                            sel = (p[:, 0] > 0) & (np.abs(p[:, 1]) <= self.ahead_half_width)
                        else:
                            c = SECTOR_DEG[s]
                            d = (bearing - c + 180.0) % 360.0 - 180.0
                            sel = np.abs(d) <= self.half
                        n = int(sel.sum())
                        counts[s] = n
                        if n >= self.min_pts:
                            rs = r[sel]
                            dist[s] = self._robust_min(rs)
                            median[s] = float(np.median(rs))
        return Sectors(t=t, dist=dist, median=median, counts=counts, frame_id=frame_id)

    def _robust_min(self, rs: np.ndarray) -> float:
        """Minimum distance that is supported by enough points: the first `bin_m`-wide radial bin holding at least
        `min_points_per_bin` points (ghost/multipath points from the head shell are sparse). Falls back to k-th smallest."""
        if self.min_per_bin <= 1:
            k = min(self.kth, len(rs)) - 1
            return float(np.partition(rs, k)[k])
        nb = int(np.ceil(self.r_max / self.bin_m)) + 1
        h = np.bincount(np.minimum((rs / self.bin_m).astype(np.int64), nb - 1), minlength=nb)
        idx = np.nonzero(h >= self.min_per_bin)[0]
        if idx.size == 0:
            return self.r_max
        lo = idx[0] * self.bin_m
        inbin = rs[(rs >= lo) & (rs < lo + self.bin_m)]
        return float(np.median(inbin)) if inbin.size else float(lo + self.bin_m / 2)

    def process(self, xyz_lidar: np.ndarray, t: float, frame_id: str = "") -> Sectors:
        return self.sectors_from_body(self.to_body(xyz_lidar), t, frame_id)

    # -- multi-frame accumulation (the L1 is a non-repetitive scanner: one 15 Hz frame covers only part of the FOV) ----
    def push(self, xyz_lidar: np.ndarray, t: float, frame_id: str = "") -> Sectors:
        """Add a frame and return sectors computed over the last `accumulate_s` seconds of frames (no motion compensation;
        at 0.4 m/s and 0.4 s this smears by <= 0.16 m, acceptable for 1 m thresholds)."""
        if not hasattr(self, "_frames"):
            self._frames = []
        self._frames.append((t, self.to_body(xyz_lidar)))
        win = float(self.cfg.get("accumulate_s", 0.4))
        self._frames = [(tt, p) for (tt, p) in self._frames if t - tt <= win]
        pts = np.concatenate([p for (_, p) in self._frames]) if len(self._frames) > 1 else self._frames[-1][1]
        s = self.sectors_from_body(pts, t, frame_id)
        s.frames = len(self._frames)
        return s

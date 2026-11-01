import math
import struct

import numpy as np

from go2nav.robot.lidar import LidarProcessor, distance_to_vocab, rpy_to_matrix
from go2nav.robot.remote import decode_wireless_remote, RemoteMonitor
from go2nav.robot.adapter import decode_pointcloud2

LCFG = {"mount_rpy": [0, 0, 0], "mount_xyz": [0, 0, 0], "z_min_m": -0.2, "z_max_m": 0.6, "r_min_m": 0.15, "r_max_m": 12.0,
        "sector_half_width_deg": 22.5, "blocked_below_m": 1.0, "near_below_m": 2.0, "open_cap_m": 8.0, "min_points_per_sector": 3, "kth_smallest": 3}
RCFG = {"button_bits": {"R1": 0, "L1": 1, "START": 2, "SELECT": 3, "R2": 4, "L2": 5, "F1": 6, "F3": 7, "A": 8, "B": 9, "X": 10, "Y": 11, "UP": 12, "RIGHT": 13, "DOWN": 14, "LEFT": 15},
        "estop_buttons": ["SELECT", "L1"], "stick_deadzone": 0.15, "stick_hold_samples": 2, "lowstate_timeout_ms": 500,
        "stick_offsets": {"lx": 4, "rx": 8, "ry": 12, "ly": 20}}


def ring(r_by_deg, z=0.0):
    pts = []
    for deg in range(-180, 180):
        r = r_by_deg(deg)
        a = math.radians(deg)
        for _ in range(3):
            pts.append((r * math.cos(a), r * math.sin(a), z))
    return np.array(pts, dtype=np.float32)


def test_sectors_and_vocab():
    lp = LidarProcessor(LCFG)
    xyz = ring(lambda d: 0.8 if abs(d) < 10 else (0.5 if abs(d - 90) < 10 else 6.0))
    s = lp.process(xyz, 1.0)
    assert s.dist["ahead"] < 1.0 and s.dist["left"] < 1.0 and s.dist["behind"] > 5.0
    v = s.vocab(LCFG)
    assert v["ahead"].startswith("blocked_0p8") and v["left"].startswith("wall_0p5") and v["behind"] == "open_6m"
    assert distance_to_vocab("ahead", 1.5, LCFG) == "near_1p5m"
    assert distance_to_vocab("right", 9.7, LCFG) == "open_8m"


def test_height_band_filters_floor():
    lp = LidarProcessor(LCFG)
    floor = ring(lambda d: 0.5, z=-0.3)     # floor hits below the band must not count as obstacles
    s = lp.process(floor, 1.0)
    assert all(d == 12.0 for d in s.dist.values())
    assert all(v == "unknown" for v in s.vocab(LCFG).values())   # no valid points -> unknown, not open


def test_body_box_excludes_own_body():
    lp = LidarProcessor({**LCFG, "body_box": [-0.85, 0.15, 0.30]})
    pts = np.array([[-0.5, 0.1, 0.3], [-0.5, 0.1, 0.3], [-0.5, 0.1, 0.3], [3.0, 0.0, 0.3]] * 3, dtype=np.float32)
    s = lp.process(pts, 1.0)
    assert s.dist["behind"] == 12.0 and abs(s.dist["ahead"] - 3.0) < 1e-6


def test_mount_transform_pitch():
    cfg = {**LCFG, "mount_rpy": [0, math.pi, 0]}   # upside-down mount: lidar +x maps to body -x
    lp = LidarProcessor(cfg)
    p = lp.to_body(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert abs(p[0, 0] + 1.0) < 1e-5


def test_robust_min_ignores_sparse_ghosts():
    lp = LidarProcessor({**LCFG, "bin_m": 0.2, "min_points_per_bin": 6})
    wall = ring(lambda d: 3.0)                                  # dense wall at 3 m
    ghosts = np.array([[0.5, 0.0, 0.3], [0.9, 0.1, 0.3], [1.3, -0.1, 0.3]], dtype=np.float32)   # 3 sparse ghost points ahead
    s = lp.process(np.concatenate([wall, ghosts]), 1.0)
    assert abs(s.dist["ahead"] - 3.0) < 0.15
    dense = np.repeat(np.array([[0.7, 0.05, 0.3]], dtype=np.float32), 8, axis=0)   # a real obstacle: 8 points in one bin
    s = lp.process(np.concatenate([wall, dense]), 1.0)
    assert abs(s.dist["ahead"] - 0.7) < 0.15


def test_push_accumulates_frames():
    lp = LidarProcessor({**LCFG, "accumulate_s": 0.4})
    left = ring(lambda d: 2.0 if 60 < d < 120 else 50.0)[lambda a: a[np.hypot(a[:, 0], a[:, 1]) < 12]] if False else None
    pts_left = np.array([[0.0, 2.0, 0.3]] * 5, dtype=np.float32)
    pts_ahead = np.array([[4.0, 0.0, 0.3]] * 5, dtype=np.float32)
    s1 = lp.push(pts_left, 1.0)
    assert s1.vocab(LCFG)["ahead"] == "unknown"
    s2 = lp.push(pts_ahead, 1.1)
    assert s2.frames == 2 and abs(s2.dist["left"] - 2.0) < 1e-5 and abs(s2.dist["ahead"] - 4.0) < 1e-5
    s3 = lp.push(pts_ahead, 2.0)        # the left frame has expired
    assert s3.frames == 1 and s3.vocab(LCFG)["left"] == "unknown"


def test_scan_plane_artifact_removed_but_wall_kept():
    cfg = {**LCFG, "body_box": [-0.85, 0.15, 0.35], "scan_plane": {"enabled": True, "fit_r_min_m": 0.35, "fit_r_max_m": 0.9, "fit_min_points": 150,
                                                                   "inlier_m": 0.06, "min_inlier_frac": 0.7, "max_tilt_deg": 35, "remove_r_max_m": 1.2}}
    lp = LidarProcessor(cfg)
    rng = np.random.default_rng(0)
    n = 3000
    ang = rng.uniform(-np.pi, np.pi, n); rad = rng.uniform(0.3, 1.0, n)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    disc = np.stack([x, y, 0.12 * x - 0.25 * y + 0.39 + rng.normal(0, 0.015, n)], 1).astype(np.float32)   # artifact disc
    wall = np.array([[3.0, yy, zz] for yy in np.linspace(-1, 1, 30) for zz in np.linspace(0.15, 0.7, 6)], dtype=np.float32)
    s = lp.process(np.concatenate([disc, wall]), 1.0)
    assert lp.last_plane and lp.last_plane["source"] == "fit" and lp.last_plane["removed"] > 2500
    assert abs(s.dist["ahead"] - 3.0) < 0.15
    # a real box 0.5 m ahead spanning several heights survives the plane removal
    box = np.array([[0.5, yy, zz] for yy in np.linspace(-0.2, 0.2, 8) for zz in np.linspace(0.15, 0.7, 8)], dtype=np.float32)
    s = lp.process(np.concatenate([disc, wall, box]), 2.0)
    assert abs(s.dist["ahead"] - 0.5) < 0.15


def test_decode_pointcloud2():
    class F:
        def __init__(self, n, o, dt):
            self.name, self.offset, self.datatype = n, o, dt
    fields = [F("x", 0, 7), F("y", 4, 7), F("z", 8, 7), F("intensity", 12, 7)]
    data = struct.pack("<ffff", 1.0, 2.0, 3.0, 9.0) + struct.pack("<ffff", -1.0, 0.5, 0.0, 1.0)
    xyz = decode_pointcloud2(fields, 16, data)
    assert xyz.shape == (2, 3) and xyz[0, 1] == 2.0 and xyz[1, 0] == -1.0


def remote_bytes(keys=0, lx=0.0, rx=0.0, ry=0.0, ly=0.0):
    b = bytearray(40)
    struct.pack_into("<H", b, 2, keys)
    struct.pack_into("<f", b, 4, lx); struct.pack_into("<f", b, 8, rx); struct.pack_into("<f", b, 12, ry); struct.pack_into("<f", b, 20, ly)
    return bytes(b)


def test_decode_wireless_remote():
    st = decode_wireless_remote(remote_bytes(keys=(1 << 3) | (1 << 8), ly=0.42), RCFG)
    assert set(st.buttons) == {"SELECT", "A"} and abs(st.ly - 0.42) < 1e-6


def test_remote_monitor_estop_takeover_lost():
    m = RemoteMonitor(RCFG)
    m.update_low(remote_bytes(keys=1 << 3), 1.0)        # SELECT only: no estop
    assert m.poll(1.0) is None
    m.update_low(remote_bytes(keys=(1 << 3) | (1 << 1)), 1.01)
    assert m.poll(1.01) == "estop_remote"
    assert m.release_ok(1.01)[0] is False                # buttons still held
    m.update_low(remote_bytes(), 1.02)
    assert m.release_ok(1.02)[0] is True
    m.clear()
    m.update_low(remote_bytes(lx=0.3), 1.03)            # one sample: debounce
    assert m.poll(1.03) is None
    m.update_low(remote_bytes(lx=0.3), 1.04)
    assert m.poll(1.04) == "operator_takeover"
    m.clear()
    m.update_low(remote_bytes(), 2.0)
    assert m.poll(2.4) is None and m.poll(2.6) == "remote_lost"

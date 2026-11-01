import threading
import time

from conftest import run_job


def test_readonly_rejects_actions(data_dir):
    from go2nav.robot.server import build_core, create_app
    from fastapi.testclient import TestClient
    core = build_core(allow_motion=False, sim=True, sim_time_scale=4.0)
    core.start()
    try:
        with TestClient(create_app(core)) as c:
            time.sleep(0.3)
            j = run_job(c, "go_forward_until", {"stop_on": ["blocked"], "max_m": 2})
            assert j["status"] == "rejected_by_gate" and j["reason"] == "motion_disabled"
            assert core.adapter.moves == []
            j = run_job(c, "look_around", {})
            assert j["status"] == "ok" and len(j["images"]) == 1 and "hires_id" in j["images"][0]
            assert c.get("/health").json()["allow_motion"] is False
    finally:
        core.stop()


def test_forward_opening_turn_lookaround(sim_app):
    c, core = sim_app
    j = run_job(c, "go_forward_until", {"stop_on": ["opening", "blocked"], "max_m": 15})
    assert j["status"] == "ok" and j["stopped_because"] == "opening_left" and 6.0 <= j["traveled_m"] <= 8.0
    assert j["images"] and "free_space" in j
    j = run_job(c, "turn", {"bearing": "left_90"})
    assert j["status"] == "ok" and 85 <= j["turned_deg"] <= 95
    j = run_job(c, "go_forward_until", {"stop_on": ["blocked"], "max_m": 3})
    assert j["status"] == "ok" and j["stopped_because"] == "distance" and 2.8 <= j["traveled_m"] <= 3.4
    j = run_job(c, "look_around", {})
    assert j["status"] == "ok" and [i["bearing_deg"] for i in j["images"]] == [-90, -45, 0, 45, 90]
    assert "hires_id" in j["images"][2]
    assert c.get(f"/image/{j['images'][2]['hires_id']}").status_code == 200
    assert c.post("/primitive", json={"tool": "nope", "args": {}}).status_code == 400


def test_estop_button_stops_within_half_second(sim_app):
    c, core = sim_app
    t_press = [None]

    def press():
        time.sleep(0.8)
        t_press[0] = time.monotonic()
        core.adapter.set_remote(keys=(1 << 3) | (1 << 1))
    threading.Thread(target=press).start()
    j = run_job(c, "go_forward_until", {"stop_on": ["blocked"], "max_m": 15})
    assert j["status"] == "interrupted" and j["interrupted_by"] == "estop_remote"
    assert core.gate.latch == "estop_remote"
    assert core.gate.latch_t - t_press[0] < 0.5
    assert core.adapter.stop_moves >= 1
    assert not any(m[0] > core.gate.latch_t and (m[1] or m[3]) for m in core.adapter.moves)
    assert run_job(c, "turn", {"bearing": "left_45"})["status"] == "rejected_by_gate"
    assert c.post("/estop/release").json()["released"] is False      # buttons still held
    core.adapter.set_remote(keys=0)
    time.sleep(0.2)
    assert c.post("/estop/release").json()["released"] is True
    assert run_job(c, "turn", {"bearing": "right_45"})["status"] == "ok"


def test_stick_takeover_and_cli_estop(sim_app):
    c, core = sim_app
    threading.Thread(target=lambda: (time.sleep(0.6), core.adapter.set_remote(ly=0.5))).start()
    j = run_job(c, "go_forward_until", {"stop_on": ["blocked"], "max_m": 15})
    assert j["interrupted_by"] == "operator_takeover"
    core.adapter.set_remote()
    time.sleep(0.2)
    assert c.post("/estop/release").json()["released"] is True
    r = c.post("/estop", json={"reason": "cli"}).json()
    assert r["latched"] == "cli"
    assert c.get("/state").json()["safety"]["estop"] is True
    assert c.post("/estop/release").json()["released"] is True


def test_sensor_freeze_stops_motion(sim_app):
    c, core = sim_app
    threading.Thread(target=lambda: (time.sleep(0.6), setattr(core.adapter, "frozen_sensors", True))).start()
    j = run_job(c, "go_forward_until", {"stop_on": ["blocked"], "max_m": 15})
    assert j["status"] == "interrupted" and j["interrupted_by"].startswith("gate_")
    time.sleep(1.2)
    assert core.gate.latch == "state_lost" or core.moving is False
    core.adapter.frozen_sensors = False


def test_task_recording_and_state_log(sim_app):
    c, core = sim_app
    r = c.post("/task/start", json={"task_id": "t-test"}).json()
    assert r["recording"]
    run_job(c, "turn", {"bearing": "left_45"})
    c.post("/task/stop")
    from pathlib import Path
    d = Path(r["task_dir"])
    assert (d / "robot.jsonl").exists() and (d / "cam" / "cam_index.jsonl").exists()
    lines = (d / "robot.jsonl").read_text().splitlines()
    assert any('"event": "task_begin"' in l for l in lines) and any('"job_tool": "turn"' in l for l in lines)
    time.sleep(0.5)
    assert list((d / "cam").glob("cam_*.mp4"))


def test_approach_stops_close_to_target(sim_app):
    c, core = sim_app
    core.adapter.teleport(9.0, 0.0, 0.0)          # dead end at x=12
    j = run_job(c, "approach", {"bearing": "ahead", "stop_m": 0.5})
    assert j["status"] == "ok" and j["stopped_because"] == "reached"
    assert 0.25 <= j["final_range_m"] <= 0.62, j  # creep keeps the overshoot small (sim runs 4x: sensor lag is 4x too); gate floor 0.45


def test_pose_skill_and_auto_recovery(sim_app):
    c, core = sim_app
    j = run_job(c, "pose", {"action": "hello"})
    assert j["status"] == "ok" and j["pose"] == "standing" and j["calls"] == [{"call": "Hello", "code": 0}] and len(j["images"]) == 1
    j = run_job(c, "pose", {"action": "sit"})
    assert j["status"] == "ok" and j["pose"] == "sitting" and core.pose == "sitting"
    assert c.get("/state").json()["motion"]["pose"] == "sitting"
    # a motion skill while sitting recovers first (RiseSit), then moves
    j = run_job(c, "go_forward_until", {"stop_on": ["distance"], "max_m": 0.5})
    assert j["status"] == "ok" and j["stopped_because"] == "distance" and [r["call"] for r in j["recovery"]] == ["RiseSit"]
    assert core.pose == "standing"
    # explicit stand_up when already standing is a no-op
    j = run_job(c, "pose", {"action": "stand_up"})
    assert j["status"] == "ok" and j.get("note") == "already_standing"
    # firmware refusal is reported, pose unchanged
    core.adapter.action_codes["StandDown"] = 3104
    j = run_job(c, "pose", {"action": "stand_down"})
    assert j["status"] == "failed" and j["reason"] == "StandDown_code_3104" and core.pose == "standing"
    assert run_job(c, "pose", {"action": "backflip"})["status"] == "failed"
    # E-stop during a pose interrupts it
    core.adapter.action_codes.pop("StandDown")
    threading.Thread(target=lambda: (time.sleep(0.5), core.adapter.set_remote(keys=(1 << 3) | (1 << 1)))).start()
    j = run_job(c, "pose", {"action": "stand_down"})
    assert j["status"] == "interrupted" and j["interrupted_by"] == "estop_remote"
    core.adapter.set_remote(keys=0)
    time.sleep(0.2)
    assert c.post("/estop/release").json()["released"] is True
    assert run_job(c, "pose", {"action": "stand_up"})["status"] == "ok" and core.pose == "standing"


def test_pose_rejected_without_motion(data_dir):
    from go2nav.robot.server import build_core, create_app
    from fastapi.testclient import TestClient
    core = build_core(allow_motion=False, sim=True, sim_time_scale=4.0)
    core.start()
    try:
        with TestClient(create_app(core)) as c:
            time.sleep(0.3)
            j = run_job(c, "pose", {"action": "hello"})
            assert j["status"] == "rejected_by_gate" and j["reason"] == "motion_disabled" and core.adapter.actions == []
    finally:
        core.stop()


def test_turn_stall_kick_and_accept(sim_app):
    """Firmware deadband: |vyaw| < 0.6 ignored. The turn must kick at max_vyaw and finish within tolerance or the accept window."""
    c, core = sim_app
    core.adapter.yaw_deadband = 0.6     # > min_vyaw (0.5): the fine-approach phase stalls, the kick (0.8) moves
    y0 = core.odom.yaw
    j = run_job(c, "turn", {"bearing": "left_45"})
    assert j["status"] == "ok", j
    assert 35 <= j["turned_deg"] <= 55
    assert j.get("stall_kicks", 0) >= 1 or j.get("stall_accepted_deg") is not None
    core.adapter.yaw_deadband = 0.0


def test_forward_stall_kick_and_give_up(sim_app):
    c, core = sim_app
    core.adapter.vx_deadband = 0.32      # cruise 0.4 moves, slow 0.3 does not, kick 0.35 does
    core.adapter.teleport(9.0, 0.0, 0.0)  # dead end at x=12: slow zone starts at ahead < 1.8 m
    j = run_job(c, "go_forward_until", {"stop_on": ["blocked", "distance"], "max_m": 5})
    assert j["status"] == "ok" and j["stopped_because"] == "blocked" and j.get("stall_kicks", 0) >= 1, j
    core.adapter.vx_deadband = 0.6       # nothing moves: give up quickly with a clear reason
    core.adapter.teleport(2.0, 0.0, 0.0)
    time.sleep(0.6)                       # let the 0.4 s LiDAR accumulation window forget the dead end
    t0 = time.monotonic()
    j = run_job(c, "go_forward_until", {"stop_on": ["distance"], "max_m": 3})
    assert j["status"] == "failed" and j["stopped_because"] == "stalled" and "stalled" in j["reason"] and j["traveled_m"] < 0.1
    assert time.monotonic() - t0 < 15
    j = run_job(c, "approach", {"bearing": "ahead", "stop_m": 0.8})
    assert j["status"] == "failed" and j["stopped_because"] == "stalled"
    core.adapter.vx_deadband = 0.0

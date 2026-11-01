from go2nav.robot.gate import Gate, Cmd
from go2nav.robot.lidar import Sectors, SECTORS

CFG = {"max_vx": 0.5, "max_vy": 0.3, "max_vyaw": 0.8, "min_vyaw": 0.25, "cmd_freshness_ms": 200, "state_freshness_ms": 300,
       "lidar_freshness_ms": 500, "min_range_ahead_m": 0.6, "min_range_side_m": 0.0, "use_builtin_range_obstacle": True}


def sect(t, ahead=5.0, **kw):
    d = {s: 5.0 for s in SECTORS}
    d["ahead"] = ahead
    d.update(kw)
    return Sectors(t=t, dist=d)


def test_motion_disabled_by_default():
    g = Gate(CFG, allow_motion=False)
    d = g.decide(Cmd(0.3, 0, 0, 10.0, "x"), 10.0, 10.0, sect(10.0))
    assert not d.allowed and d.reason == "motion_disabled" and d.cmd.is_zero()


def test_clamp_and_min_yaw():
    g = Gate(CFG, allow_motion=True)
    d = g.decide(Cmd(2.0, 0, 0.1, 10.0, "x"), 10.0, 10.0, sect(10.0))
    assert d.allowed and d.cmd.vx == 0.5 and d.cmd.vyaw == 0.25 and d.modified


def test_stale_command_and_state():
    g = Gate(CFG, allow_motion=True)
    assert g.decide(Cmd(0.3, 0, 0, 9.0, "x"), 10.0, 10.0, sect(10.0)).reason == "cmd_stale"
    assert g.decide(Cmd(0.3, 0, 0, 10.0, "x"), 10.0, 9.0, sect(10.0)).reason == "state_stale"
    assert g.decide(Cmd(0.3, 0, 0, 10.0, "x"), 10.0, 10.0, sect(9.0)).reason == "lidar_stale"


def test_min_range_blocks_forward_only():
    g = Gate(CFG, allow_motion=True)
    d = g.decide(Cmd(0.3, 0, 0, 10.0, "x"), 10.0, 10.0, sect(10.0, ahead=0.5))
    assert d.reason == "min_range_ahead" and d.cmd.is_zero()
    d = g.decide(Cmd(0.0, 0, 0.5, 10.0, "x"), 10.0, 10.0, sect(10.0, ahead=0.5))
    assert d.allowed and d.cmd.vyaw == 0.5
    d = g.decide(Cmd(0.3, 0, 0, 10.0, "x"), 10.0, 10.0, sect(10.0), range_obstacle_front=0.4)
    assert d.reason == "min_range_builtin"


def test_latch_is_sticky_until_release():
    g = Gate(CFG, allow_motion=True)
    assert g.set_latch("estop_remote", 10.0)
    assert not g.set_latch("cli", 11.0)
    d = g.decide(Cmd(0.3, 0, 0, 12.0, "x"), 12.0, 12.0, sect(12.0))
    assert d.latched == "estop_remote" and d.cmd.is_zero()
    g.release()
    assert g.decide(Cmd(0.3, 0, 0, 12.0, "x"), 12.0, 12.0, sect(12.0)).allowed

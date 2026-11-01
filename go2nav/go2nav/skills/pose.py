"""pose: body actions of the Go2 sport mode (sit, lie down, stand up, hello, stretch) — robot side.

Each action is one SportClient RPC that returns immediately while the motion plays for a few seconds, so
the skill waits `settle_s` (abortable: E-stop / stick / cancel still interrupt) and then takes one image.
`sit` and `stand_down` leave the robot in a non-walking stance; `core.pose` tracks it and the job runner
recovers the walking stance automatically before the next motion skill (see `recover_standing`).
"""
from __future__ import annotations

from .registry import SkillSpec

# action name -> (adapter call sequence, resulting core.pose)
ACTIONS = {
    "stand_down": (["StandDown"], "lying"),      # 伏せ（しゃがむ）
    "sit":        (["Sit"], "sitting"),          # お座り
    "stand_up":   (["recover"], "standing"),     # 立ち上がって歩ける姿勢に戻る
    "hello":      (["Hello"], "standing"),       # 前足を上げて挨拶
    "stretch":    (["Stretch"], "standing"),     # 伸び
}

SPEC = SkillSpec(
    name="pose", kind="effect",
    description="その場で体の動作をする。stand_down=伏せる（しゃがむ）、sit=お座り、stand_up=立ち上がって歩ける姿勢に戻る、hello=前足を上げて挨拶する、stretch=伸びをする。"
                "伏せ・お座りの後に移動するときは自動で立ち上がる。動作後に前方画像を1枚返す。",
    params={"action": {"type": "string", "enum": list(ACTIONS)}}, required=["action"], evidence=True)


def _cfg(runner) -> dict:
    return runner.cfg.get("pose", {})


def _settle(runner, name: str) -> float:
    return float(_cfg(runner).get("settle_s", {}).get(name, 3.0))


def recover_standing(runner, job) -> tuple[bool, str]:
    """Bring the robot back to the balance-stand (walking) stance from sitting/lying. Returns (ok, why)."""
    core = runner.core
    seq = {"sitting": ["RiseSit"], "lying": ["StandUp", "BalanceStand"]}.get(core.pose, [])
    for name in seq:
        code = core.adapter.sport_action(name)
        job.result.setdefault("recovery", []).append({"call": name, "code": code})
        if code != 0:
            return False, f"{name}_code_{code}"
        a = runner.sleep_abortable(job, _settle(runner, name))
        if a:
            return False, a
    core.set_pose("standing")
    return True, "ok"


def run(runner, job) -> None:
    core = runner.core
    name = str(job.args.get("action", ""))
    if name not in ACTIONS:
        job.status, job.result = "failed", {"reason": f"bad_action:{name}"}
        return
    if core.moving:   # let the zero-velocity hold finish before a body action
        a = runner.sleep_abortable(job, float(core.cfg["gate"].get("zero_hold_s", 0.3)) + 0.2)
        if a:
            return
    calls, final_pose = ACTIONS[name]
    runner.phase(job, name)
    for call in calls:
        if call == "recover":
            if core.pose == "standing":
                job.result["note"] = "already_standing"
                continue
            ok, why = recover_standing(runner, job)
            if not ok:
                job.status, job.result["reason"] = ("interrupted" if why in ("cancelled", "interrupted") else "failed"), why
                return
            continue
        if core.pose != "standing" and call in ("Hello", "Stretch", "Sit", "StandDown"):
            ok, why = recover_standing(runner, job)        # gestures need the standing stance first
            if not ok:
                job.status, job.result["reason"] = "failed", why
                return
        code = core.adapter.sport_action(call)
        job.result.setdefault("calls", []).append({"call": call, "code": code})
        if code != 0:
            job.status, job.result["reason"] = "failed", f"{call}_code_{code}"
            return
        core.set_pose(final_pose if call in ("Sit", "StandDown") else "standing")
        a = runner.sleep_abortable(job, _settle(runner, call))
        if a:
            return
    job.result["pose"] = core.pose
    job.status = "ok"
    runner.snapshot(job, 0, hires=False, kind="pose", max_age_ms=800)
    job.result["free_space"] = core.free_space()

"""report: declare the task finished (control kind, brain side).

status=done is accepted only with recent evidence: a `verify` (and then something must be within
report_max_ahead_range_m ahead, LiDAR cross-check) or a successful effect skill (pose / notify) within
`report_evidence_turns` turns.
"""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="report", kind="control", side="brain", alone=True, needs_motion=False,
    description="終了宣言（単独で返す）。status=done は、直前 2 ターン以内に verify で目標を確認した（かつ目標が前方 2m 以内）か、"
                "指示された動作（pose / notify）を実行済みのときだけ受理される。summary に何を見つけ何をしたかを書く。",
    params={"status": {"type": "string", "enum": ["done", "failed", "need_help"]}, "summary": {"type": "string"}}, required=["status", "summary"])


def check(ex, status: str) -> tuple[bool, str]:
    if status != "done":
        return True, "ok"
    n = int(ex.budget.get("report_evidence_turns", 2))
    recent_effect = ex.last_effect_turn is not None and ex.turn - ex.last_effect_turn <= n
    recent_verify = ex.last_verify_turn is not None and ex.turn - ex.last_verify_turn <= n
    if ex.budget.get("report_requires_verify", True) and not (recent_verify or recent_effect):
        return False, "verify_required_before_report"
    if recent_verify and not recent_effect:
        ws = ex.last_ws or {}
        ahead = (ws.get("free_space_m") or {}).get("ahead")
        mx = float(ex.budget.get("report_max_ahead_range_m", 2.0))
        if ahead is not None and ahead > mx:
            return False, f"nothing_within_{mx}m_ahead (ahead={ahead}m); approach first"
    return True, "ok"


def run_brain(ex, call, turn: int, images: list) -> dict:
    status = call.args.get("status", "done")
    ok, why = check(ex, status)
    if ok:
        ex.finish(status, {"summary": call.args.get("summary", "")})
        return {"tool": "report", "status": "ok"}
    ex.progress(f"  report rejected: {why}")
    return {"tool": "report", "status": "rejected", "reason": why}

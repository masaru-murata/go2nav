"""backtrack: walk back along the remembered junction graph (brain side; expands into turn/go_forward_until)."""
from __future__ import annotations

from ..common import now
from .registry import SkillSpec

SPEC = SkillSpec(
    name="backtrack", kind="action", side="brain", alone=True,
    description="記憶した分岐点を steps 個分、逆順に辿って戻る（単独で返す）。到達後に見回し画像を返す。",
    params={"steps": {"type": "integer", "minimum": 1, "maximum": 3}}, required=["steps"])


def run_brain(ex, call, turn: int, images: list) -> dict:
    steps = int(call.args.get("steps", 1) or 1)
    plan, target = ex.memory.backtrack_plan(steps)
    if not plan:
        return {"tool": "backtrack", "status": "failed", "reason": "no_edges_to_backtrack"}
    legs = []
    for tool, args in plan:
        res = ex.robot.run(tool, args)
        legs.append(ex.exec_record(tool, res))
        if tool in ("go_forward_until", "approach"):
            ex.distance_m += float(res.get("traveled_m", 0.0) or 0.0)
        if res.get("status") != "ok":
            return {"tool": "backtrack", "status": "interrupted", "interrupted_by": res.get("interrupted_by") or res.get("status"),
                    "legs": legs, "node_id": None}
    ex.memory.arrive_back(target)
    la = ex.robot.run("look_around", {})
    ex.absorb_result("look_around", la, turn, images)
    return {"tool": "backtrack", "status": "ok", "node_id": target, "legs": legs, "images": la.get("images", []),
            "free_space": la.get("free_space", {}), "t": now()}

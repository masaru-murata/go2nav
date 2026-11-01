"""verify: a hires forward image with the model's own confirmation question (brain side; evidence for report)."""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="verify", kind="observe", side="brain", evidence=True,
    description="近接画像（高解像度）に対する確認質問。目標を見つけたと思ったら approach の後に使い、report の前に必須。質問文を question に書く。",
    params={"question": {"type": "string"}}, required=["question"])


def run_brain(ex, call, turn: int, images: list) -> dict:
    res = ex.robot.run("observe", {"hires": True})
    res["question"] = call.args.get("question", "")
    ex.absorb_result("verify", res, turn, images)
    return res

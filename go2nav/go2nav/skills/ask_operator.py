"""ask_operator: ask the human operator and wait for `go2nav answer` (control kind, brain side)."""
from __future__ import annotations

from .registry import SkillSpec

SPEC = SkillSpec(
    name="ask_operator", kind="control", side="brain", alone=True, needs_motion=False,
    description="操作者に質問して回答を待つ（単独で返す）。",
    params={"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}, required=["question", "options"])


def run_brain(ex, call, turn: int, images: list) -> dict:
    q = call.args.get("question", "")
    opts = call.args.get("options", [])
    ex.progress(f"ASK_OPERATOR: {q} options={opts}  -> answer with: go2nav answer \"...\"")
    ans = ex.rec.wait_answer(ex.answer_timeout_s)
    if ans is None:
        ex.finish("no_operator_answer", {"question": q})
        return {"tool": "ask_operator", "status": "timeout", "reason": "no_answer"}
    return {"tool": "ask_operator", "status": "ok", "answer": ans}

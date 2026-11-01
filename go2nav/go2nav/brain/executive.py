"""Executive (design chapters 6-9, 12): runs the turn loop between the model and the robot process.

One turn = one model call. The model returns a rationale (mandatory), optional remember notes, and either
a batch of up to 3 actions/effects (+ one observation at the end) or a single control tool. Actions are executed
sequentially; an interruption discards the rest. Everything is written to the task directory.

Which tools exist, how they batch and who executes them (robot process or brain) comes from the skill
registry (go2nav/skills). The system prompt may contain `{{SKILLS}}`, replaced by the generated catalog.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from ..common import now, iso, wrap_deg
from .memory import Memory
from .guard import GuardTripped
from .model_client import ModelClient, ModelTurn, image_part, text_part, is_transient
from .recorder import TaskRecorder, new_task_id
from .robot_client import RobotClient
from .tools import Call, TurnPlan, plan_turn, tool_declarations
from .. import skills

log = logging.getLogger("go2nav.executive")


class Executive:
    def __init__(self, robot: RobotClient, model: ModelClient, cfg: dict, data_dir: Path, instructions: str,
                 instruction: str, memo_paths: Optional[list[str]] = None, task_id: Optional[str] = None,
                 progress: Optional[Callable[[str], None]] = None, answer_timeout_s: float = 600.0, http=None):
        self.robot = robot
        self.model = model
        self.cfg = cfg                       # keys: budget, model, primitives, notify (optional)
        self.http = http                     # httpx.Client-compatible object for brain-side effects (tests inject a mock)
        self.budget = cfg["budget"]
        self.data_dir = Path(data_dir)
        self.instructions = instructions.replace("{{SKILLS}}", skills.prompt_catalog(cfg))
        self.instruction = instruction
        self.memo_paths = list(memo_paths or [])
        self.task_id = task_id or new_task_id(self.data_dir)
        self.progress = progress or (lambda s: None)
        self.answer_timeout_s = answer_timeout_s
        self.tools = tool_declarations(cfg)
        self.memory = Memory()
        self.history: list[dict] = []
        self._image_msgs: list[tuple[int, dict]] = []     # (turn, message dict with images) for later stubbing
        self.turn = 0
        self.cost = 0.0
        self.t_start = now()
        self.distance_m = 0.0
        self._last_odom: Optional[dict] = None
        self._last_batches: list[str] = []
        self._invalid_count = 0
        self.last_verify_turn: Optional[int] = None
        self.last_effect_turn: Optional[int] = None
        self.finished: Optional[dict] = None
        self.rec: Optional[TaskRecorder] = None
        self.last_ws: Optional[dict] = None

    # ------------------------------------------------------------------------------------------
    def run(self) -> dict:
        meta = {"model": getattr(self.model, "model", self.model.name), "reasoning_effort": self.cfg["model"].get("reasoning_effort"),
                "skills": [t["name"] for t in self.tools],
                "budget": {k: self.budget.get(k) for k in ("turns_max", "elapsed_max_s", "distance_max_m", "cost_max_usd")}}
        self.rec = TaskRecorder(self.data_dir, self.task_id, self.instruction, self.memo_paths, meta)
        self.progress(f"task {self.task_id} start: {self.instruction}")
        try:
            self.robot.task_start(self.task_id)
            self._initial_message()
            while self.finished is None:
                self._one_turn()
        except KeyboardInterrupt:
            self._finish("aborted", {"reason": "keyboard_interrupt"})
        except Exception as e:
            log.exception("executive error: %s", e)
            self._finish("error", {"reason": f"{type(e).__name__}: {e}"})
        finally:
            try:
                self.robot.task_stop()
            except Exception:
                pass
            self.rec.save_memory(self.memory.to_dict())
            self.rec.close()
            try:
                from .export_html import export_task
                export_task(self.rec.dir)
            except Exception as e:  # pragma: no cover
                log.warning("html export failed: %s", e)
        return self.finished or {}

    # ------------------------------------------------------------------------------------------
    def _initial_message(self) -> None:
        ws = self._world_state(turn=0)
        self.rec.save_world_state(0, ws)
        content = [text_part(f"指示: {self.instruction}")]
        for p in self.memo_paths:
            rel = self.rec.save_memo(p)
            content.append(text_part(f"メモの写真 ({Path(p).name})"))
            content.append(image_part(Path(p).read_bytes(), self.cfg["model"].get("image_detail", "high")))
        frame = self.robot.frame(hires=False)
        if frame:
            rel = self.rec.save_image(0, "start_ahead", frame)
            content.append(text_part("出発時の前方画像 (bearing=0)"))
            content.append(image_part(frame, self.cfg["model"].get("image_detail", "high")))
        content.append(text_part("初期 WorldState:\n" + yaml.safe_dump(ws, allow_unicode=True, sort_keys=False)))
        content.append(text_part("最初の応答では rationale と remember(kind=instruction)（目標と、見つけたらする動作）を書き、look_around で周囲を見てください。"))
        msg = {"role": "user", "content": content}
        self.history.append(msg)
        self._image_msgs.append((0, msg))
        self.memory.add_node(0, now(), [], ws["free_space"], "start", 0.0, where_am_i="出発点")

    # ------------------------------------------------------------------------------------------
    def _one_turn(self) -> None:
        self.turn += 1
        turn = self.turn
        t_turn = now()
        over = self._budget_exceeded()
        if over:
            self._handle_budget(over)
            if self.finished is not None:
                return

        mt = self._call_model()
        if mt is None:
            return
        self.history.extend(mt.output_items)
        self.rec.save_model_raw(turn, {"calls": [{"name": c.name, "args": c.args, "call_id": c.call_id} for c in mt.calls],
                                       "text": mt.text, "reasoning_summary": mt.reasoning_summary, "usage": mt.usage,
                                       "output_items": _strip_items(mt.output_items)})
        self.cost += float(mt.usage.get("cost_usd", 0.0) or 0.0)
        plan = plan_turn(mt.calls, int(self.budget.get("max_actions_per_batch", 3)), float(self.cfg["primitives"]["go_forward_until"]["max_m"]), self.cfg)

        record: dict[str, Any] = {"task_id": self.task_id, "turn_id": turn, "t": t_turn, "t_iso": iso(t_turn),
                                  "model": {"rationale": plan.rationale, "remember": [c.args for c in plan.remembers],
                                            "tool_calls": [{"tool": c.name, "args": c.args} for c in mt.calls],
                                            "text": mt.text, "reasoning_summary": mt.reasoning_summary, "usage": mt.usage},
                                  "validation": {"errors": plan.errors, "truncated": plan.truncated},
                                  "inputs": {"world_state_ref": None, "images": []},
                                  "execution": None}

        outputs: list[dict] = []
        images: list[tuple[str, bytes, str]] = []   # (label, jpeg, relpath)

        if not plan.ok or not mt.calls:
            self._invalid_count += 1
            errs = plan.errors or ["no_tool_calls"]
            for c in mt.calls:
                outputs.append(_fc_output(c.call_id, {"status": "rejected", "reason": "turn_rejected:" + ",".join(errs)}))
            msg = {"role": "user", "content": [text_part("応答は受理されませんでした: " + ", ".join(errs) +
                                                          "。毎回 rationale ツールを含め、行動（最大 3 つ）か観測か制御ツールのいずれかを必ず付け、制御ツールは単独で返してください。")]}
            self.history.extend(outputs)
            self.history.append(msg)
            record["execution"] = {"rejected": errs}
            self._write_turn(record, None)
            self.progress(f"turn {turn}: rejected {errs}")
            if self._invalid_count >= 3:
                self._finish("model_protocol_error", {"reason": errs})
            return
        self._invalid_count = 0

        # rationale / remember
        r = plan.rationale or {}
        self.memory.set_where(str(r.get("where_am_i", "")))
        self.memory.set_chosen(str(r.get("chosen", "")))
        for c in mt.calls:
            if c.name == "rationale":
                outputs.append(_fc_output(c.call_id, {"status": "ok"}))
        for c in plan.remembers:
            self.memory.add_note(turn, str(c.args.get("kind", "evidence")), str(c.args.get("note", "")))
            outputs.append(_fc_output(c.call_id, {"status": "ok"}))
        for c in plan.dropped:
            outputs.append(_fc_output(c.call_id, {"status": "rejected", "reason": "not_executed:" + ",".join(plan.truncated or ["control_tool_present"])}))
        self.progress(f"turn {turn}: {r.get('chosen','')} (conf {r.get('confidence','?')}) cost so far ${self.cost:.2f}")

        execution: dict[str, Any] = {"executed": [], "skipped": [], "interrupted_by": None}
        ws_before = self.last_ws or self._world_state(turn)

        if plan.control is not None:
            c = plan.control
            res = self._run_skill(c, turn, images)
            outputs.append(_fc_output(c.call_id, _result_for_model(c.name, res)))
            execution["executed"].append(self.exec_record(c.name, res))
        else:
            # L2 rule checks
            rule = self._l2_check(plan, ws_before)
            if rule:
                for c in plan.actions + ([plan.observation] if plan.observation else []):
                    outputs.append(_fc_output(c.call_id, {"status": "rejected", "reason": rule}))
                execution["interrupted_by"] = rule
                self.progress(f"turn {turn}: L2 rejected ({rule})")
            else:
                self._execute_batch(plan, outputs, images, execution, turn)

        ws = self._world_state(turn)
        record["inputs"]["world_state_ref"] = self.rec.save_world_state(turn, ws)
        record["inputs"]["images"] = [rel for (_, _, rel) in images]
        record["execution"] = execution
        # next input: function outputs + a user message with the images and the world state
        content: list[dict] = []
        for label, jpeg, rel in images:
            content.append(text_part(f"画像 {rel}: {label}"))
            content.append(image_part(jpeg, self.cfg["model"].get("image_detail", "high")))
        content.append(text_part(f"WorldState (turn {turn}):\n" + yaml.safe_dump(ws, allow_unicode=True, sort_keys=False)))
        if execution.get("interrupted_by"):
            content.append(text_part(f"バッチは中断されました: {execution['interrupted_by']}"))
        msg = {"role": "user", "content": content}
        self.history.extend(outputs)
        self.history.append(msg)
        if images:
            self._image_msgs.append((turn, msg))
        self._stub_old_images(turn)
        self._write_turn(record, ws)

    # ------------------------------------------------------------------------------------------
    def _call_model(self) -> Optional[ModelTurn]:
        """One model call, at most one retry and only for transient errors. A guard trip ends the task at once."""
        last = "model_unreachable"
        for attempt in (1, 2):
            try:
                return self.model.respond(self.instructions, self.history, self.tools)
            except GuardTripped as e:
                log.error("API guard tripped: %s", e)
                self.progress(f"API GUARD tripped: {e}")
                self._finish("api_guard", {"reason": e.reason, "detail": e.detail, "guard": getattr(self.model.guard, "summary", lambda: {})()})
                return None
            except Exception as e:
                last = f"{type(e).__name__}: {e}"[:300]
                log.warning("model call failed (attempt %d): %s", attempt, last)
                self.progress(f"model call failed: {last}")
                if not is_transient(e):
                    break
                time.sleep(float(self.budget.get("model_retry_wait_s", 5.0)))
        self._finish("model_error", {"reason": last})
        return None

    # ------------------------------------------------------------------------------------------
    def _execute_batch(self, plan: TurnPlan, outputs: list[dict], images: list, execution: dict, turn: int) -> None:
        interrupted = None
        for c in plan.actions:
            if interrupted:
                execution["skipped"].append({"tool": c.name, "reason": "interrupted"})
                outputs.append(_fc_output(c.call_id, {"status": "skipped", "reason": f"interrupted:{interrupted}"}))
                continue
            res = self._run_skill(c, turn, images)
            execution["executed"].append(self.exec_record(c.name, res))
            outputs.append(_fc_output(c.call_id, _result_for_model(c.name, res)))
            self.progress(f"  {c.name} {c.args} -> {res.get('status')} {res.get('stopped_because') or res.get('interrupted_by') or res.get('reason') or ''}")
            if res.get("status") != "ok":
                interrupted = res.get("interrupted_by") or res.get("reason") or res.get("status")
        if plan.observation is not None:
            c = plan.observation
            if interrupted:
                execution["skipped"].append({"tool": c.name, "reason": "interrupted"})
                outputs.append(_fc_output(c.call_id, {"status": "skipped", "reason": f"interrupted:{interrupted}"}))
            else:
                res = self._run_skill(c, turn, images)
                execution["executed"].append(self.exec_record(c.name, res))
                outputs.append(_fc_output(c.call_id, _result_for_model(c.name, res)))
                self.progress(f"  {c.name} -> {res.get('status')} ({len(res.get('images', []))} images)")
        execution["interrupted_by"] = interrupted

    def _run_skill(self, c: Call, turn: int, images: list) -> dict:
        """Execute one skill call: brain-side skills run here, robot-side ones through the robot process."""
        spec = skills.get(c.name)
        if spec.side == "brain":
            res = spec.run_brain(self, c, turn, images)
        else:
            res = self.robot.run(c.name, c.args)
            self.absorb_result(c.name, res, turn, images)
        if res.get("status") == "ok":
            if spec.evidence and spec.kind == "observe":
                self.last_verify_turn = turn
            elif spec.evidence:
                self.last_effect_turn = turn
        return res

    def absorb_result(self, tool: str, res: dict, turn: int, images: list) -> None:
        """Download images, update odometry/memory from a primitive result."""
        saved = []
        for im in res.get("images", []) or []:
            for key, suffix in (("id", ""), ("hires_id", "_hires")):
                iid = im.get(key)
                if not iid:
                    continue
                jpeg = self.robot.image(iid)
                if not jpeg:
                    continue
                b = im.get("bearing_deg", 0)
                label = f"{tool}_{int(b):+d}{suffix}"
                rel = self.rec.save_image(turn, label, jpeg)
                fs = im.get("free_space") or {}
                desc = f"{tool} bearing={b:+}" + (" 高解像度切り出し" if suffix else "") + (f" free_space={json.dumps(fs, ensure_ascii=False)}" if fs and not suffix else "")
                if tool == "verify":
                    desc += f" 確認質問: {res.get('question','')}"
                if tool == "pose":
                    desc += f" 動作後の姿勢: {res.get('pose','')}"
                images.append((desc, jpeg, rel))
                saved.append(rel)
        res["saved_images"] = saved
        spec = skills.get(tool)
        if res.get("turned_deg") is not None and res.get("status") in ("ok", "interrupted", "timeout"):
            self.memory.note_turn(float(res.get("turned_deg", 0.0) or 0.0))
        if res.get("traveled_m") is not None:
            self.distance_m += float(res.get("traveled_m", 0.0) or 0.0)
        if spec is not None and spec.makes_node and res.get("status") == "ok":
            self.memory.add_node(turn, now(), saved, res.get("free_space", {}), str(res.get("stopped_because", "")),
                                 float(res.get("traveled_m", 0.0) or 0.0))

    def exec_record(self, tool: str, res: dict) -> dict:
        return _exec_record(tool, res)

    def finish(self, status: str, result: Optional[dict]) -> None:
        self._finish(status, result)

    def _l2_check(self, plan: TurnPlan, ws: dict) -> Optional[str]:
        if plan.actions and plan.actions[0].name == "go_forward_until":
            if str(ws.get("free_space", {}).get("ahead", "")).startswith("blocked"):
                return "rule_forward_into_blocked"
        key = json.dumps([[c.name, c.args] for c in plan.actions], sort_keys=True, ensure_ascii=False)
        self._last_batches.append(key)
        lim = int(self.budget.get("same_batch_repeat_limit", 3))
        if plan.actions and len(self._last_batches) >= lim and all(k == key for k in self._last_batches[-lim:]):
            return "rule_same_batch_repeated"
        return None

    # ------------------------------------------------------------------------------------------
    def _budget_exceeded(self) -> Optional[str]:
        if self.turn > int(self.budget.get("turns_max", 40)):
            return "turns"
        if now() - self.t_start > float(self.budget.get("elapsed_max_s", 1200)):
            return "time"
        if self.distance_m > float(self.budget.get("distance_max_m", 300)):
            return "distance"
        if self.cost > float(self.budget.get("cost_max_usd", 10.0)):
            return "cost"
        return None

    def _handle_budget(self, what: str) -> None:
        self.progress(f"BUDGET exceeded ({what}). Answer 'continue' to extend, anything else to stop: go2nav answer continue")
        ans = self.rec.wait_answer(min(self.answer_timeout_s, 300.0))
        if ans and ans.strip().lower() in ("continue", "y", "yes", "続行"):
            if what == "turns":
                self.budget["turns_max"] = int(self.budget["turns_max"]) + 10
            elif what == "time":
                self.budget["elapsed_max_s"] = float(self.budget["elapsed_max_s"]) + 300
            elif what == "distance":
                self.budget["distance_max_m"] = float(self.budget["distance_max_m"]) + 100
            elif what == "cost":
                self.budget["cost_max_usd"] = float(self.budget["cost_max_usd"]) + 5
            self.history.append({"role": "user", "content": [text_part(f"予算（{what}）を超えましたが、操作者が続行を許可しました。")]})
        else:
            self._finish("budget_exceeded", {"budget": what})

    # ------------------------------------------------------------------------------------------
    def _world_state(self, turn: int) -> dict:
        st = self.robot.state()
        od = st.get("odom") or {}
        traveled_since = 0.0
        heading_change = 0.0
        if self._last_odom:
            traveled_since = float(od.get("traveled_total_m", 0.0)) - float(self._last_odom.get("traveled_total_m", 0.0))
            heading_change = wrap_deg(float(od.get("yaw_deg", 0.0)) - float(self._last_odom.get("yaw_deg", 0.0)))
        self._last_odom = od
        ws = {
            "t": st.get("t"),
            "age_ms": st.get("age_ms"),
            "free_space": st.get("free_space"),
            "free_space_m": st.get("free_space_m"),
            "motion": {"status": (st.get("motion") or {}).get("status"), "last_stop_reason": (st.get("motion") or {}).get("last_stop_reason"),
                       "pose": (st.get("motion") or {}).get("pose"),
                       "traveled_since_last_decision_m": round(traveled_since, 1), "heading_change_deg": round(heading_change, 0)},
            "memory": self.memory.summary(),
            "budget": {"turns_used": turn, "turns_max": self.budget.get("turns_max"), "elapsed_s": int(now() - self.t_start),
                       "elapsed_max_s": self.budget.get("elapsed_max_s"), "distance_m": round(self.distance_m, 1),
                       "distance_max_m": self.budget.get("distance_max_m"), "cost_usd": round(self.cost, 2), "cost_max_usd": self.budget.get("cost_max_usd")},
            "safety": st.get("safety"),
            "battery": st.get("battery"),
            "notebook_tail": self.memory.notebook_tail(),
        }
        self.last_ws = ws
        return ws

    def _stub_old_images(self, turn: int) -> None:
        keep = int(self.cfg["model"].get("keep_images_for_turns", 2))
        for t, msg in self._image_msgs:
            if turn - t > keep:
                new_content = []
                for part in msg.get("content", []):
                    if part.get("type") == "input_image":
                        new_content.append(text_part("(画像は履歴から省略。必要なら observe / look_around で再取得)"))
                    else:
                        new_content.append(part)
                msg["content"] = new_content
        self._image_msgs = [(t, m) for (t, m) in self._image_msgs if turn - t <= keep]

    def _write_turn(self, record: dict, ws: Optional[dict]) -> None:
        record["cost_total_usd"] = round(self.cost, 4)
        record["elapsed_s"] = round(now() - self.t_start, 1)
        record["memory"] = self.memory.summary()
        self.rec.write_turn(record)
        self.rec.task.update({"turns": self.turn, "elapsed_s": round(now() - self.t_start, 1), "distance_m": round(self.distance_m, 1), "cost_usd": round(self.cost, 3)})
        self.rec.write_task()

    def _finish(self, status: str, result: Optional[dict]) -> None:
        if self.finished is not None:
            return
        self.finished = {"task_id": self.task_id, "status": status, "result": result, "turns": self.turn,
                         "elapsed_s": round(now() - self.t_start, 1), "distance_m": round(self.distance_m, 1), "cost_usd": round(self.cost, 3),
                         "api_calls": getattr(getattr(self.model, "guard", None), "calls", None)}
        if self.rec:
            self.rec.finish(status, result, self.turn, now() - self.t_start, self.distance_m, self.cost)
        self.progress(f"task {self.task_id} finished: {status} {result} turns={self.turn} cost=${self.cost:.2f}")


# ----------------------------------------------------------------------------------------------
def _fc_output(call_id: str, obj: dict) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(obj, ensure_ascii=False)}


def _result_for_model(tool: str, res: dict) -> dict:
    keep = ("status", "stopped_because", "traveled_m", "turned_deg", "interrupted_by", "reason", "final_range_m", "truncated",
            "free_space", "new_node_id", "node_id", "note", "question", "moved_m", "answer", "pose", "http_status", "delivered", "calls", "recovery")
    out = {k: res[k] for k in keep if k in res and res[k] is not None}
    out["tool"] = tool
    if res.get("saved_images"):
        out["images"] = res["saved_images"]
    if tool == "go_forward_until" and res.get("status") == "ok":
        out["new_node_id"] = out.get("new_node_id")
    return out


def _exec_record(tool: str, res: dict) -> dict:
    keep = ("status", "stopped_because", "traveled_m", "turned_deg", "interrupted_by", "reason", "final_range_m", "truncated", "t_start", "t_end",
            "node_id", "legs", "saved_images", "answer", "pose", "http_status", "delivered", "calls", "recovery", "latency_s")
    return {"tool": tool, **{k: res[k] for k in keep if k in res}}


def _strip_items(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        d = dict(it)
        if d.get("type") == "reasoning":
            d.pop("encrypted_content", None)
        out.append(d)
    return out

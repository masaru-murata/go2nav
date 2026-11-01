import json
import os
import time
from pathlib import Path

from go2nav.brain.tools import Call, plan_turn, tool_declarations
from go2nav.brain.memory import Memory


def C(name, **args):
    return Call(name=name, args=args, call_id=f"c_{name}_{len(args)}")


RAT = dict(where_am_i="廊下", goal_belief="右", options=[], chosen="x", why="y", confidence=0.5)


NOTIFY_CFG = {"notify": {"targets": {"ops": {"url": "http://hook.test/go2", "headers": {"X-Token": "${GO2NAV_TEST_TOKEN}"}}}}}


def test_tool_declarations_cover_contract():
    names = {t["name"] for t in tool_declarations()}
    assert names == {"go_forward_until", "turn", "approach", "backtrack", "pose", "observe", "look_around", "zoom", "verify",
                     "rationale", "remember", "ask_operator", "stop", "report"}          # notify needs a configured target
    decl = {t["name"]: t for t in tool_declarations(NOTIFY_CFG)}
    assert decl["notify"]["parameters"]["properties"]["target"]["enum"] == ["ops"]
    assert decl["pose"]["parameters"]["properties"]["action"]["enum"] == ["stand_down", "sit", "stand_up", "hello", "stretch"]
    from go2nav import skills
    cat = skills.prompt_catalog(NOTIFY_CFG)
    assert "pose(action∈{stand_down" in cat and "notify(target∈{ops}" in cat


def test_plan_effects_batch_with_actions_and_notify_only_when_offered():
    p = plan_turn([C("rationale", **RAT), C("approach", bearing="ahead", stop_m=0.8), C("pose", action="stand_down"), C("verify", question="椅子か")])
    assert p.ok and [a.name for a in p.actions] == ["approach", "pose"] and p.observation.name == "verify"
    p = plan_turn([C("rationale", **RAT), C("notify", target="ops", message="m", include_route=True)], cfg={"notify": {}})
    assert "unknown_tool:notify" in p.errors
    p = plan_turn([C("rationale", **RAT), C("notify", target="ops", message="m", include_route=True)], cfg=NOTIFY_CFG)
    assert p.ok and p.actions[0].name == "notify"


def test_plan_requires_rationale_and_limits():
    p = plan_turn([C("go_forward_until", stop_on=["blocked"], max_m=5)])
    assert "rationale_missing" in p.errors
    p = plan_turn([C("rationale", **RAT), C("turn", bearing="left_90"), C("turn", bearing="left_45"), C("turn", bearing="right_45"),
                   C("go_forward_until", stop_on=["opening"], max_m=40), C("look_around")])
    assert p.ok and len(p.actions) == 3 and len(p.dropped) == 1 and "max_actions:3" in p.truncated
    assert p.observation.name == "look_around"
    p = plan_turn([C("rationale", **RAT), C("go_forward_until", stop_on=["opening"], max_m=40)])
    assert p.actions[0].args["max_m"] == 15.0 and "max_m_capped" in p.truncated
    p = plan_turn([C("rationale", **RAT), C("remember", note="n", kind="evidence")])
    assert p.errors == ["no_executable_call"]


def test_plan_control_wins_and_backtrack_alone():
    p = plan_turn([C("rationale", **RAT), C("turn", bearing="around"), C("report", status="done", summary="s")])
    assert p.control.name == "report" and p.actions == [] and len(p.dropped) == 1
    p = plan_turn([C("rationale", **RAT), C("backtrack", steps=1), C("turn", bearing="left_90")])
    assert [a.name for a in p.actions] == ["backtrack"] and "backtrack_alone" in p.truncated


def test_memory_backtrack_plan():
    m = Memory()
    m.add_node(0, 0.0, [], {}, "start", 0.0)
    m.note_turn(90.0)
    m.add_node(1, 1.0, ["img/a.jpg"], {"ahead": "open_8m"}, "opening_left", 11.2)
    m.note_turn(-45.0)
    m.add_node(2, 2.0, [], {}, "blocked", 3.0)
    plan, target = m.backtrack_plan(2)
    assert target == "n0"
    assert plan[0] == ("turn", {"bearing": "around"}) and plan[1][1]["max_m"] == 3.0 and plan[2] == ("turn", {"bearing": -45.0})
    assert plan[3] == ("turn", {"bearing": "around"}) and plan[4][1]["max_m"] == 11.2 and plan[5] == ("turn", {"bearing": 90.0})
    m.arrive_back("n0")
    assert m.summary()["current_node"] == "n0" and m.visits["n0"] == 2
    assert m.backtrack_plan(1) == ([], None)


def test_executive_end_to_end(sim_app, data_dir):
    from go2nav.brain.robot_client import RobotClient
    from go2nav.brain.model_client import ScriptedModel
    from go2nav.brain.executive import Executive
    from go2nav.common import load_yaml, home
    c, core = sim_app
    R = lambda chosen: ("rationale", {**RAT, "chosen": chosen})
    script = [
        [R("look"), ("remember", {"note": "目標は305", "kind": "instruction"}), ("look_around", {})],
        [("go_forward_until", {"stop_on": ["blocked"], "max_m": 2})],                       # rejected: no rationale
        [R("report early"), ("report", {"status": "done", "summary": "着いた"})],              # rejected: verify first
        [R("fwd"), ("go_forward_until", {"stop_on": ["distance"], "max_m": 2}), ("observe", {})],   # -> n1 at x=9
        [R("t"), ("turn", {"bearing": "left_45"})],
        [R("t"), ("turn", {"bearing": "left_45"})],
        [R("t"), ("turn", {"bearing": "left_45"})],                                           # 3rd identical batch -> L2 reject
        [R("t"), ("turn", {"bearing": "right_90"})],                                          # face +x again
        [R("back"), ("backtrack", {"steps": 1})],                                             # -> n0 at x=7, facing -x
        [R("verify"), ("verify", {"question": "荷物か"})],
        [R("done"), ("report", {"status": "done", "summary": "荷物の前"})],                     # ahead open -> rejected
        [R("appr"), ("turn", {"bearing": "around"}), ("approach", {"bearing": "ahead", "stop_m": 0.8}), ("verify", {"question": "荷物か"})],
        [R("done"), ("report", {"status": "done", "summary": "荷物の前"})],
    ]
    cfg = {"budget": load_yaml("budget"), "model": load_yaml("model"), "primitives": load_yaml("primitives"), "notify": {}}
    model = ScriptedModel(script)
    core.adapter.teleport(7.0, 0.0, 0.0)   # 5 m of corridor ahead (dead end at x=12)
    ex = Executive(RobotClient(client=c), model, cfg, data_dir, (home() / "prompts/system_v1.md").read_text(), "305号室の荷物", task_id="t-ut")
    res = ex.run()
    assert res["status"] == "done" and res["turns"] == 13
    d = data_dir / "t-ut"
    turns = [json.loads(l) for l in (d / "turns.jsonl").read_text().splitlines()]
    assert len(turns) == 13
    assert turns[1]["execution"] == {"rejected": ["rationale_missing"]}
    assert turns[2]["execution"]["executed"][0]["reason"] == "verify_required_before_report"
    assert turns[3]["execution"]["executed"][0]["stopped_because"] == "distance"
    assert turns[6]["execution"]["interrupted_by"] == "rule_same_batch_repeated"
    assert turns[8]["execution"]["executed"][0]["tool"] == "backtrack" and turns[8]["execution"]["executed"][0]["node_id"] == "n0"
    assert "nothing_within" in turns[10]["execution"]["executed"][0]["reason"]
    assert turns[11]["execution"]["executed"][1]["stopped_because"] == "reached"
    assert turns[12]["execution"]["executed"][0]["status"] == "ok"
    assert (d / "memory.json").exists() and (d / "timeline.html").exists() and (d / "task.json").exists()
    mem = json.loads((d / "memory.json").read_text())
    assert len(mem["nodes"]) >= 2 and mem["notes"][0]["kind"] == "instruction" and mem["current"] == "n0"
    # the first input is the user message with the instruction and the initial world state
    first = ex.history[0]
    texts = " ".join(p.get("text", "") for p in first["content"] if p.get("type") == "input_text")
    assert first["role"] == "user" and "305号室の荷物" in texts and "WorldState" in texts
    assert "{{SKILLS}}" not in ex.instructions and "pose(action" in ex.instructions and "notify(target" not in ex.instructions
    # images older than keep_images_for_turns were replaced by text stubs (in place), recent ones are still images
    assert any("画像は履歴から省略" in p.get("text", "") for p in first["content"])
    recent = [p for m in ex.history[-12:] if m.get("role") == "user" for p in m.get("content", []) if p.get("type") == "input_image"]
    assert recent
    # function_call_output items reference the call ids
    outs = [m for m in ex.history if m.get("type") == "function_call_output"]
    assert outs and all(m["call_id"] for m in outs)
    html = (d / "timeline.html").read_text()
    assert "Turn" in html and "cam_index" not in html and "荷物の前" in html


def test_executive_ask_operator_and_stop(sim_app, data_dir):
    from go2nav.brain.robot_client import RobotClient
    from go2nav.brain.model_client import ScriptedModel
    from go2nav.brain.executive import Executive
    from go2nav.common import load_yaml, home
    import threading
    c, core = sim_app
    script = [
        [("rationale", RAT), ("ask_operator", {"question": "右ですか", "options": ["右", "左"]})],
        [("rationale", RAT), ("stop", {"reason": "テスト終了"})],
    ]
    cfg = {"budget": load_yaml("budget"), "model": load_yaml("model"), "primitives": load_yaml("primitives")}
    model = ScriptedModel(script)
    ex = Executive(RobotClient(client=c), model, cfg, data_dir, (home() / "prompts/system_v1.md").read_text(), "q", task_id="t-ask", answer_timeout_s=10)
    def answer():
        time.sleep(1.5)
        (data_dir / "t-ask" / "answer.txt").write_text("右です")
    threading.Thread(target=answer).start()
    res = ex.run()
    assert res["status"] == "stopped" and res["turns"] == 2
    outs = [json.loads(m["output"]) for m in ex.history if m.get("type") == "function_call_output"]
    assert any(o.get("answer") == "右です" for o in outs)
    assert (data_dir / "current_task").read_text() == "t-ask"


def test_executive_effect_skills_pose_notify_report(sim_app, data_dir, monkeypatch):
    """'Go to the chair and lie down, then notify the route': approach + pose in one batch, notify with route, report accepted
    on the strength of the effect (no verify needed), and the robot stands up automatically before the next motion."""
    import httpx
    from go2nav.brain.robot_client import RobotClient
    from go2nav.brain.model_client import ScriptedModel
    from go2nav.brain.executive import Executive
    from go2nav.common import load_yaml, home
    c, core = sim_app
    monkeypatch.setenv("GO2NAV_TEST_TOKEN", "sekrit")
    posted = []

    def handler(req: httpx.Request):
        posted.append((str(req.url), dict(req.headers), json.loads(req.content)))
        return httpx.Response(200, json={"ok": True})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    R = lambda chosen: ("rationale", {**RAT, "chosen": chosen})
    script = [
        [R("done early"), ("report", {"status": "done", "summary": "x"})],                                      # no evidence -> rejected
        [R("fwd"), ("go_forward_until", {"stop_on": ["distance"], "max_m": 1.5}), ("look_around", {})],           # node n1
        [R("lie"), ("approach", {"bearing": "ahead", "stop_m": 0.8}), ("pose", {"action": "stand_down"}), ("observe", {})],
        [R("notify"), ("notify", {"target": "ops", "message": "椅子の前で伏せました", "include_route": True})],
        [R("move while lying"), ("turn", {"bearing": "left_90"})],                                               # auto stand-up first
        [R("done"), ("report", {"status": "done", "summary": "伏せて通知した"})],
    ]
    cfg = {"budget": load_yaml("budget"), "model": load_yaml("model"), "primitives": load_yaml("primitives"), **NOTIFY_CFG}
    core.adapter.teleport(9.0, 0.0, 0.0)   # dead end at x=12
    ex = Executive(RobotClient(client=c), ScriptedModel(script), cfg, data_dir, (home() / "prompts/system_v1.md").read_text(),
                   "椅子のところへ行って伏せて、順路を通知して", task_id="t-fx", http=http)
    res = ex.run()
    assert res["status"] == "done" and res["turns"] == 6
    turns = [json.loads(l) for l in (data_dir / "t-fx" / "turns.jsonl").read_text().splitlines()]
    assert turns[0]["execution"]["executed"][0]["reason"] == "verify_required_before_report"
    ex3 = turns[2]["execution"]["executed"]
    assert ex3[0]["tool"] == "approach" and ex3[0]["stopped_because"] == "reached"
    assert ex3[1]["tool"] == "pose" and ex3[1]["status"] == "ok" and ex3[1]["pose"] == "lying" and ex3[1]["saved_images"]
    assert [a[1] for a in core.adapter.actions][:1] == ["StandDown"]
    assert turns[3]["execution"]["executed"][0]["http_status"] == 200
    url, headers, payload = posted[0]
    assert url == "http://hook.test/go2" and headers["x-token"] == "sekrit"
    assert payload["message"] == "椅子の前で伏せました" and payload["task_id"] == "t-fx" and len(payload["route"]) >= 1
    assert "出発点" in payload["route_text"] and "1." in payload["route_text"]
    # turn while lying: recovery (StandUp, BalanceStand) happened before the turn, pose back to standing
    ex5 = turns[4]["execution"]["executed"][0]
    assert ex5["tool"] == "turn" and ex5["status"] == "ok" and [r["call"] for r in ex5["recovery"]] == ["StandUp", "BalanceStand"]
    assert core.pose == "standing" and [a[1] for a in core.adapter.actions] == ["StandDown", "StandUp", "BalanceStand"]
    assert turns[5]["execution"]["executed"][0]["status"] == "ok"
    ws = turns[4]["inputs"]["world_state_ref"]
    assert "pose: standing" in (data_dir / "t-fx" / ws).read_text()
    # the skills offered are recorded with the task
    task = json.loads((data_dir / "t-fx" / "task.json").read_text())
    assert "notify" in task["skills"] and "pose" in task["skills"]

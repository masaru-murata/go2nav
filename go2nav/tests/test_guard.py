import json

import pytest

from go2nav.brain.guard import CallGuard, GuardTripped, estimate_input, IMAGE_TOKENS_EST
from go2nav.brain.model_client import image_part, text_part

PRICE = {"input": 10.0, "cached_input": 1.0, "output": 50.0}


def mk(tmp_path, **over):
    clock = {"t": 1000.0}
    cfg = {"ledger": str(tmp_path / "ledger.jsonl"), "min_interval_s": 0.0, "max_calls_per_minute": 100, **over}
    g = CallGuard(cfg, PRICE, task_id="t1", model="m", clock=lambda: clock["t"], today=lambda: "2026-10-30")
    return g, clock


def test_estimate_counts_images_and_text():
    hist = [{"role": "user", "content": [text_part("a" * 250), image_part(b"x" * 10, "high")]},
            {"type": "function_call_output", "call_id": "c", "output": "b" * 500}]
    tokens, images = estimate_input(hist, "sys" * 100, [{"name": "t"}])
    assert images == 1 and tokens >= IMAGE_TOKENS_EST + 300


def test_rate_and_count_limits(tmp_path):
    g, clock = mk(tmp_path, min_interval_s=2.0, max_calls_per_minute=3, calls_max_per_task=5)
    g.before_call([]); g.after_call({"input_tokens": 100, "cost_usd": 0.01})
    with pytest.raises(GuardTripped) as e:
        g.before_call([])                       # 0 s later
    assert e.value.reason == "min_interval" and g.tripped == "min_interval"
    with pytest.raises(GuardTripped):           # stays tripped
        clock["t"] += 10; g.before_call([])
    g, clock = mk(tmp_path, min_interval_s=0.0, max_calls_per_minute=3)
    for _ in range(3):
        clock["t"] += 5; g.before_call([]); g.after_call(None, ok=True)
    clock["t"] += 5
    with pytest.raises(GuardTripped) as e:
        g.before_call([])
    assert e.value.reason == "max_calls_per_minute"
    g, clock = mk(tmp_path, calls_max_per_task=2)
    for _ in range(2):
        clock["t"] += 70; g.before_call([]); g.after_call(None)
    clock["t"] += 70
    with pytest.raises(GuardTripped) as e:
        g.before_call([])
    assert e.value.reason == "calls_max_per_task"


def test_cost_limits_use_usage_or_estimate_and_ledger_is_shared(tmp_path):
    g, clock = mk(tmp_path, cost_hard_max_usd=1.0, daily_cost_max_usd=100.0)
    g.before_call([]); g.after_call({"input_tokens": 1000, "cost_usd": 0.6})
    clock["t"] += 70
    g.before_call([]); g.after_call(None, ok=True)            # no usage -> conservative estimate (> 0)
    assert g.cost_usd > 0.6
    clock["t"] += 70
    g.before_call([]); g.after_call({"input_tokens": 1000, "cost_usd": 0.3})
    clock["t"] += 70
    with pytest.raises(GuardTripped) as e:
        g.before_call([])
    assert e.value.reason == "cost_hard_max_usd"
    # ledger: another guard (another process / task) sees today's spend
    rows = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["task_id"] == "t1" and rows[1]["cost_source"] == "estimate"
    g2, _ = mk(tmp_path, daily_cost_max_usd=1.0)
    assert g2.daily_spend() == pytest.approx(g.cost_usd, abs=1e-6)
    with pytest.raises(GuardTripped) as e:
        g2.before_call([])
    assert e.value.reason == "daily_cost_max_usd"
    # a failed call records no cost but is logged
    g3, _ = mk(tmp_path, daily_cost_max_usd=100.0)
    g3.before_call([]); g3.after_call(None, ok=False, error="APITimeoutError")
    assert g3.cost_usd == 0.0 and json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])["ok"] is False


def test_request_size_limits(tmp_path):
    g, _ = mk(tmp_path, max_images_in_input=2)
    hist = [{"role": "user", "content": [image_part(b"x", "high")] * 3}]
    with pytest.raises(GuardTripped) as e:
        g.before_call(hist)
    assert e.value.reason == "max_images_in_input"
    g, _ = mk(tmp_path, max_input_tokens_est=1000)
    with pytest.raises(GuardTripped) as e:
        g.before_call([{"role": "user", "content": [text_part("x" * 10000)]}])
    assert e.value.reason == "max_input_tokens_est"


def test_executive_stops_on_guard_trip(sim_app, data_dir):
    """A model that answers instantly (bug loop) trips min_interval on the 2nd call; the task ends with api_guard, no retry."""
    from go2nav.brain.robot_client import RobotClient
    from go2nav.brain.model_client import ScriptedModel
    from go2nav.brain.executive import Executive
    from go2nav.common import load_yaml, home
    c, core = sim_app
    rat = ("rationale", {"where_am_i": "a", "goal_belief": "b", "options": [], "chosen": "c", "why": "d", "confidence": 0.5})
    script = [[rat, ("observe", {})]] * 10
    guard = CallGuard({"ledger": str(data_dir / "ledger.jsonl"), "min_interval_s": 30.0}, PRICE, task_id="t-g")
    model = ScriptedModel(script, guard=guard)
    cfg = {"budget": load_yaml("budget"), "model": load_yaml("model"), "primitives": load_yaml("primitives"), "notify": {}}
    ex = Executive(RobotClient(client=c), model, cfg, data_dir, (home() / "prompts/system_v1.md").read_text(), "g", task_id="t-g")
    res = ex.run()
    assert res["status"] == "api_guard" and res["result"]["reason"] == "min_interval" and res["api_calls"] == 1
    assert model.i == 1   # no retry after the trip
    assert json.loads((data_dir / "t-g" / "task.json").read_text())["status"] == "api_guard"

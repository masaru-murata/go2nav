"""Validation of one model turn (design 6.6, 7.1 rules), driven by the skill registry (go2nav/skills)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import skills


def tool_declarations(cfg: Optional[dict] = None) -> list[dict]:
    return skills.tool_declarations(cfg)


@dataclass
class Call:
    name: str
    args: dict
    call_id: str


@dataclass
class TurnPlan:
    rationale: Optional[dict] = None
    remembers: list[Call] = field(default_factory=list)
    actions: list[Call] = field(default_factory=list)       # action + effect skills, executed in order
    observation: Optional[Call] = None
    control: Optional[Call] = None
    errors: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    dropped: list[Call] = field(default_factory=list)   # calls not executed because of the rules (reported back as rejected)

    @property
    def ok(self) -> bool:
        return not self.errors

    def all_calls(self) -> list[Call]:
        out = list(self.remembers + self.actions + self.dropped)
        if self.observation:
            out.append(self.observation)
        if self.control:
            out.append(self.control)
        return out


def plan_turn(calls: list[Call], max_actions: int = 3, max_forward_m: float = 15.0, cfg: Optional[dict] = None) -> TurnPlan:
    """Classify and validate the calls of one model response according to design 6.6 / 7.1."""
    p = TurnPlan()
    offered = {s.name for s in skills.offered(cfg)} if cfg is not None else set(skills.registry())
    rationale_calls = [c for c in calls if c.name == "rationale"]
    if not rationale_calls:
        p.errors.append("rationale_missing")
    else:
        p.rationale = rationale_calls[-1].args
        if len(rationale_calls) > 1:
            p.errors.append("multiple_rationale")
        for k in ("where_am_i", "goal_belief", "chosen", "why"):
            if not str(p.rationale.get(k, "")).strip():
                p.errors.append(f"rationale_field_empty:{k}")
    for c in calls:
        spec = skills.get(c.name)
        if spec is None or c.name not in offered:
            p.errors.append(f"unknown_tool:{c.name}")
            continue
        if c.name == "rationale":
            continue
        if spec.kind == "record":
            p.remembers.append(c)
        elif spec.kind == "control":
            if p.control is not None:
                p.errors.append("multiple_control_tools")
            p.control = c
        elif spec.kind == "observe":
            if p.observation is not None:
                p.errors.append("multiple_observation_tools")
            p.observation = c
        else:   # action / effect
            p.actions.append(c)
    if not p.actions and p.observation is None and p.control is None:
        p.errors.append("no_executable_call")      # rationale/remember alone would waste a turn
    # control tools win over actions (actions are not executed)
    if p.control is not None and (p.actions or p.observation):
        p.dropped.extend(p.actions)
        if p.observation:
            p.dropped.append(p.observation)
        p.actions, p.observation = [], None
    # "alone" skills (backtrack) must be the only action
    lonely = [c for c in p.actions if skills.get(c.name).alone]
    if lonely and len(p.actions) > 1:
        keep = lonely[:1]
        p.dropped.extend([c for c in p.actions if c not in keep])
        p.actions = keep
        p.truncated.append(f"{keep[0].name}_alone")
    if len(p.actions) > max_actions:
        p.dropped.extend(p.actions[max_actions:])
        p.actions = p.actions[:max_actions]
        p.truncated.append(f"max_actions:{max_actions}")
    # forward distance capped (also summed within the batch)
    total_fwd = 0.0
    for c in p.actions:
        if c.name == "go_forward_until":
            m = float(c.args.get("max_m", max_forward_m) or max_forward_m)
            if m > max_forward_m:
                c.args["max_m"] = max_forward_m
                p.truncated.append("max_m_capped")
            if total_fwd + float(c.args.get("max_m", max_forward_m)) > max_forward_m:
                c.args["max_m"] = max(0.5, max_forward_m - total_fwd)
                p.truncated.append("batch_forward_capped")
            total_fwd += float(c.args["max_m"])
            if not c.args.get("stop_on"):
                c.args["stop_on"] = ["opening", "blocked", "distance"]
    # evidence before report(done) is checked by the report skill (needs history)
    return p


def parse_function_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}

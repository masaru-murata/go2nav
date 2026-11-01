"""Skill registry: every robot capability the model can call is declared once, as one module in this package.

A skill module defines `SPEC` (a SkillSpec) and an executor:
  - robot-side skills:  `run(runner, job)`   executed by the robot process (JobRunner), may move the robot
  - brain-side skills:  `run_brain(ex, call, turn, images) -> dict`   executed by the brain (Executive)

The registry generates the OpenAI tool declarations, the batching rules (plan_turn) and the robot's
job table from the specs, so adding a capability = adding one file here (and listing it in SKILL_MODULES).

Kinds:
  action  : moves the robot (batchable, up to max_actions per turn; motion gate applies)
  effect  : does something in the world without navigation (pose, notify); batchable with actions
  observe : returns images; at most one per turn, executed last
  record  : rationale / remember (no execution)
  control : ask_operator / stop / report; must be the only executable call of the turn
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

KINDS = ("action", "effect", "observe", "record", "control")

# order = order shown to the model
SKILL_MODULES = [
    "go_forward_until", "turn", "approach", "backtrack",
    "pose", "notify",
    "observe", "look_around", "zoom", "verify",
    "rationale", "remember",
    "ask_operator", "stop", "report",
]


@dataclass
class SkillSpec:
    name: str
    kind: str
    description: str
    params: dict = field(default_factory=dict)          # JSON schema properties (or built by `declare`)
    required: list[str] = field(default_factory=list)
    side: str = "robot"                                  # robot | brain
    alone: bool = False                                  # must be the only action of the batch
    needs_motion: bool = True                            # robot-side: requires --allow-motion and no latch
    makes_node: bool = False                             # a successful run creates a memory node (junction)
    evidence: bool = False                               # a successful run counts as evidence for report(done)
    declare: Optional[Callable[[dict], tuple[dict, list[str]]]] = None   # cfg -> (params, required) when config-dependent
    available: Optional[Callable[[dict], bool]] = None   # cfg -> bool (skill offered to the model only if True)
    run: Optional[Callable] = None                       # robot-side executor  run(runner, job)
    run_brain: Optional[Callable] = None                 # brain-side executor  run_brain(ex, call, turn, images) -> dict

    def __post_init__(self):
        assert self.kind in KINDS, self.kind
        assert self.side in ("robot", "brain"), self.side

    def schema(self, cfg: Optional[dict]) -> tuple[dict, list[str]]:
        if self.declare is not None:
            return self.declare(cfg or {})
        return self.params, self.required

    def to_tool(self, cfg: Optional[dict] = None) -> dict:
        props, req = self.schema(cfg)
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False}}


_REGISTRY: dict[str, SkillSpec] = {}


def _load() -> dict[str, SkillSpec]:
    if not _REGISTRY:
        for m in SKILL_MODULES:
            mod = importlib.import_module(f"{__package__}.{m}")
            spec: SkillSpec = mod.SPEC
            spec.run = getattr(mod, "run", None)
            spec.run_brain = getattr(mod, "run_brain", None)
            assert spec.name == m, (spec.name, m)
            _REGISTRY[spec.name] = spec
    return _REGISTRY


def registry() -> dict[str, SkillSpec]:
    return _load()


def get(name: str) -> Optional[SkillSpec]:
    return _load().get(name)


def names(kind: Optional[str] = None, side: Optional[str] = None) -> list[str]:
    return [s.name for s in _load().values() if (kind is None or s.kind == kind) and (side is None or s.side == side)]


def offered(cfg: Optional[dict] = None) -> list[SkillSpec]:
    """Skills offered to the model for this configuration."""
    return [s for s in _load().values() if s.available is None or s.available(cfg or {})]


def tool_declarations(cfg: Optional[dict] = None) -> list[dict]:
    """OpenAI Responses API function tools, generated from the registry."""
    return [s.to_tool(cfg) for s in offered(cfg)]


def prompt_catalog(cfg: Optional[dict] = None) -> str:
    """Human/model readable list of the offered skills, grouped by kind (inserted into the system prompt)."""
    labels = {"action": "行動（移動）", "effect": "作用（移動以外の動作）", "observe": "観測", "record": "記録", "control": "制御"}
    lines = []
    for kind in KINDS:
        ss = [s for s in offered(cfg) if s.kind == kind]
        if not ss:
            continue
        lines.append(f"- {labels[kind]}: " + " / ".join(s.name for s in ss))
        for s in ss:
            props, _ = s.schema(cfg)
            args = ", ".join(_arg_hint(k, v) for k, v in props.items())
            lines.append(f"  - {s.name}({args}): {s.description}")
    return "\n".join(lines)


def _arg_hint(k: str, v: dict) -> str:
    if "enum" in v:
        return f"{k}∈{{{','.join(map(str, v['enum']))}}}"
    if v.get("type") == "array" and "enum" in (v.get("items") or {}):
        return f"{k}⊆{{{','.join(map(str, v['items']['enum']))}}}"
    return f"{k}:{v.get('type', 'any')}"

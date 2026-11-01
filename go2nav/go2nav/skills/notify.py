"""notify: post a message (optionally with the remembered route) to a configured webhook (brain side).

Targets come from config/notify.yaml; the model only sees target names, never URLs. The URL may reference
environment variables (e.g. ${GO2NAV_WEBHOOK_URL} from config/secrets.env).
"""
from __future__ import annotations

import os
import time

from ..common import now
from .registry import SkillSpec


def _targets(cfg: dict) -> dict:
    return dict(((cfg or {}).get("notify") or {}).get("targets") or {})


def _declare(cfg: dict) -> tuple[dict, list[str]]:
    names = sorted(_targets(cfg)) or ["default"]
    return ({"target": {"type": "string", "enum": names},
             "message": {"type": "string"},
             "include_route": {"type": "boolean"}},
            ["target", "message", "include_route"])


SPEC = SkillSpec(
    name="notify", kind="effect", side="brain", evidence=True, needs_motion=False,
    description="設定済みの通知先（webhook）にメッセージを送る。include_route=true で出発点からここまでの順路（旋回と距離の列）と画像の一覧を添える。"
                "通知後は画像は返らない。",
    declare=_declare, available=lambda cfg: bool(_targets(cfg)))


def run_brain(ex, call, turn: int, images: list) -> dict:
    import httpx
    targets = _targets(ex.cfg)
    name = str(call.args.get("target", "default"))
    t = targets.get(name)
    if t is None:
        return {"tool": "notify", "status": "failed", "reason": f"unknown_target:{name}"}
    url = os.path.expandvars(str(t.get("url", "")))
    if not url or url.startswith("$"):
        return {"tool": "notify", "status": "failed", "reason": "target_url_unset"}
    payload = {"task_id": ex.task_id, "turn": turn, "t": now(), "instruction": ex.instruction,
               "message": str(call.args.get("message", "")), "target": name}
    if call.args.get("include_route"):
        payload["route"] = ex.memory.route()
        payload["route_text"] = ex.memory.route_text()
    headers = {str(k): os.path.expandvars(str(v)) for k, v in (t.get("headers") or {}).items()}
    t0 = time.monotonic()
    try:
        client = ex.http or httpx.Client(timeout=float(t.get("timeout_s", 10.0)))
        r = client.post(url, json=payload, headers=headers)
        ok = 200 <= r.status_code < 300
        return {"tool": "notify", "status": "ok" if ok else "failed", "http_status": r.status_code,
                "reason": None if ok else f"http_{r.status_code}", "latency_s": round(time.monotonic() - t0, 2),
                "delivered": {"target": name, "message": payload["message"], "route_text": payload.get("route_text")}}
    except Exception as e:
        return {"tool": "notify", "status": "failed", "reason": f"{type(e).__name__}: {e}"[:200]}

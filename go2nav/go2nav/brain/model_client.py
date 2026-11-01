"""Model clients: the OpenAI Responses API (GPT-6 Astra) and a scripted stand-in for offline tests.

History is managed by the brain (store=False) so that old images can be replaced by text stubs and the whole
session is reproducible from the logs. Reasoning items are passed back with encrypted content.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .guard import CallGuard, GuardTripped
from .tools import Call, parse_function_args

log = logging.getLogger("go2nav.model")


@dataclass
class ModelTurn:
    calls: list[Call]
    text: str = ""
    reasoning_summary: str = ""
    output_items: list[dict] = field(default_factory=list)    # raw output items to append to the history
    usage: dict = field(default_factory=dict)                  # input_tokens, cached_tokens, output_tokens, reasoning_tokens, cost_usd, latency_s
    response_id: Optional[str] = None
    raw: Any = None


def image_part(jpeg: bytes, detail: str = "high") -> dict:
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": detail}


def text_part(text: str) -> dict:
    return {"type": "input_text", "text": text}


class ModelClient:
    name = "base"
    guard: Optional[CallGuard] = None

    def respond(self, instructions: str, history: list[dict], tools: list[dict]) -> ModelTurn:
        raise NotImplementedError


# exceptions worth one retry (transient); anything else (auth, bad request, guard) ends the task at once
def is_transient(e: Exception) -> bool:
    try:
        import openai
        return isinstance(e, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError))
    except Exception:  # pragma: no cover
        return False


class OpenAIResponsesClient(ModelClient):
    name = "openai"

    def __init__(self, cfg: dict, timeout_s: float = 90.0, task_id: str = ""):
        from openai import OpenAI
        self.cfg = cfg
        self.model = cfg.get("model", "gpt-6-astra")
        self.effort = cfg.get("reasoning_effort", "high")
        self.summary = cfg.get("reasoning_summary", "auto")
        self.max_out = int(cfg.get("max_output_tokens", 6000))
        self.price = cfg.get("price_usd_per_1m", {"input": 10.0, "cached_input": 1.0, "output": 50.0})
        # max_retries=0: retries are decided by the executive (one, transient errors only) so every call passes the guard
        self.client = OpenAI(timeout=timeout_s, max_retries=0)
        self.guard = CallGuard(cfg.get("guard"), self.price, task_id=task_id, model=self.model)

    def respond(self, instructions: str, history: list[dict], tools: list[dict]) -> ModelTurn:
        self.guard.before_call(history, instructions, tools)
        t0 = time.monotonic()
        kwargs: dict[str, Any] = dict(model=self.model, instructions=instructions, input=history, tools=tools,
                                      store=False, include=["reasoning.encrypted_content"],
                                      max_output_tokens=self.max_out, parallel_tool_calls=True)
        reasoning = {"effort": self.effort}
        if self.summary:
            reasoning["summary"] = self.summary
        kwargs["reasoning"] = reasoning
        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as e:
            self.guard.after_call(None, ok=False, error=f"{type(e).__name__}: {e}")
            raise
        latency = time.monotonic() - t0
        calls: list[Call] = []
        text = []
        summaries = []
        items = []
        for item in resp.output:
            d = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            items.append(d)
            typ = d.get("type")
            if typ == "function_call":
                calls.append(Call(name=d.get("name", ""), args=parse_function_args(d.get("arguments")), call_id=d.get("call_id", "")))
            elif typ == "message":
                for c in d.get("content", []):
                    if c.get("type") == "output_text":
                        text.append(c.get("text", ""))
            elif typ == "reasoning":
                for s in d.get("summary", []) or []:
                    if isinstance(s, dict) and s.get("text"):
                        summaries.append(s["text"])
        u = getattr(resp, "usage", None)
        usage = {}
        if u is not None:
            ud = u.model_dump() if hasattr(u, "model_dump") else dict(u)
            inp = int(ud.get("input_tokens", 0) or 0)
            cached = int(((ud.get("input_tokens_details") or {}).get("cached_tokens", 0)) or 0)
            out = int(ud.get("output_tokens", 0) or 0)
            rt = int(((ud.get("output_tokens_details") or {}).get("reasoning_tokens", 0)) or 0)
            cost = ((inp - cached) * self.price["input"] + cached * self.price["cached_input"] + out * self.price["output"]) / 1e6
            usage = {"input_tokens": inp, "cached_tokens": cached, "output_tokens": out, "reasoning_tokens": rt,
                     "cost_usd": round(cost, 4), "latency_s": round(latency, 2)}
        self.guard.after_call(usage, ok=True)
        return ModelTurn(calls=calls, text="\n".join(text), reasoning_summary="\n".join(summaries), output_items=items,
                         usage=usage, response_id=getattr(resp, "id", None), raw=resp)


class ScriptedModel(ModelClient):
    """Offline stand-in: returns pre-scripted calls turn by turn. `script` is a list of lists of (name, args),
    or a callable(history) -> list of (name, args)."""
    name = "scripted"

    def __init__(self, script, latency_s: float = 0.0, guard: Optional[CallGuard] = None):
        self.script = script
        self.i = 0
        self.latency_s = latency_s
        self.inputs: list[list[dict]] = []
        self.guard = guard

    def respond(self, instructions: str, history: list[dict], tools: list[dict]) -> ModelTurn:
        if self.guard is not None:
            self.guard.before_call(history, instructions, tools)
        self.inputs.append(list(history))
        if callable(self.script):
            step = self.script(history, self.i)
        else:
            step = self.script[self.i] if self.i < len(self.script) else [
                ("rationale", {"where_am_i": "?", "goal_belief": "?", "options": [], "chosen": "stop", "why": "script exhausted", "confidence": 0.0}),
                ("stop", {"reason": "script_exhausted"})]
        self.i += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        calls = [Call(name=n, args=dict(a), call_id=f"call_{self.i}_{k}") for k, (n, a) in enumerate(step)]
        items = [{"type": "function_call", "name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False), "call_id": c.call_id}
                 for c in calls]
        usage = {"input_tokens": 1000, "cached_tokens": 500, "output_tokens": 200, "reasoning_tokens": 100, "cost_usd": 0.02, "latency_s": self.latency_s}
        if self.guard is not None:
            self.guard.after_call(usage, ok=True)
        return ModelTurn(calls=calls, output_items=items, usage=usage)


def make_model(cfg: dict, timeout_s: float, task_id: str = "") -> ModelClient:
    return OpenAIResponsesClient(cfg, timeout_s=timeout_s, task_id=task_id)

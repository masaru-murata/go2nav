"""Call guard for the paid model API: hard limits that no bug in the turn loop can get around.

Checked right before every model call (and the spend is recorded right after), independently of the
executive's budgets, which the operator can extend. Trips raise GuardTripped; the executive ends the task
with status `api_guard` and never retries.

Limits (config/model.yaml `guard:`):
  calls_max_per_task     total calls in one brain process
  max_calls_per_minute   sliding window; a bug loop calls far faster than a real turn (10-60 s)
  min_interval_s         two calls closer than this are refused
  cost_hard_max_usd      cumulative cost of this task (usage-based, or a conservative estimate when usage is missing)
  daily_cost_max_usd     cumulative cost of all tasks today (persistent ledger, shared by every process)
  max_input_tokens_est   estimated tokens of the request (history growth bug)
  max_images_in_input    images in the request (image stubbing bug)
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULTS = {
    "calls_max_per_task": 60,
    "max_calls_per_minute": 6,
    "min_interval_s": 2.0,
    "cost_hard_max_usd": 20.0,
    "daily_cost_max_usd": 50.0,
    "max_input_tokens_est": 150_000,
    "max_images_in_input": 24,
    "ledger": "~/go2nav/data/api_ledger.jsonl",
}

# conservative per-image token estimate for detail=high (OpenAI: ~85 base + 170 per 512px tile; 768px long edge -> <= 6 tiles)
IMAGE_TOKENS_EST = 1105
CHARS_PER_TOKEN = 2.5    # Japanese text tokenises densely; keep the estimate conservative


class GuardTripped(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def estimate_input(history: list[dict], instructions: str = "", tools: Optional[list] = None) -> tuple[int, int]:
    """(estimated tokens, number of images) of a request; images count at IMAGE_TOKENS_EST each."""
    chars = len(instructions or "") + (len(json.dumps(tools, ensure_ascii=False)) if tools else 0)
    images = 0
    for item in history:
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "input_image":
                    images += 1
                else:
                    chars += len(str(part.get("text", "")))
        elif isinstance(content, str):
            chars += len(content)
        else:
            for k in ("arguments", "output", "text", "summary"):
                v = item.get(k)
                if v:
                    chars += len(v) if isinstance(v, str) else len(json.dumps(v, ensure_ascii=False))
    return int(chars / CHARS_PER_TOKEN) + images * IMAGE_TOKENS_EST, images


class CallGuard:
    def __init__(self, cfg: Optional[dict] = None, price: Optional[dict] = None, task_id: str = "", model: str = "",
                 clock=time.monotonic, today=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.price = price or {"input": 10.0, "cached_input": 1.0, "output": 50.0}
        self.task_id = task_id
        self.model = model
        self.clock = clock
        self.today = today or (lambda: time.strftime("%Y-%m-%d"))
        self.calls = 0
        self.cost_usd = 0.0
        self.recent: collections.deque[float] = collections.deque()
        self.last_t: Optional[float] = None
        self._lock = threading.Lock()
        self.ledger = Path(os.path.expanduser(str(self.cfg["ledger"]))) if self.cfg.get("ledger") else None
        self.tripped: Optional[str] = None
        self._pending_est: Optional[dict] = None

    # -- checks -----------------------------------------------------------------------
    def before_call(self, history: list[dict], instructions: str = "", tools: Optional[list] = None) -> None:
        with self._lock:
            if self.tripped:
                raise GuardTripped(self.tripped, "guard already tripped")
            c = self.cfg
            t = self.clock()
            if self.calls >= int(c["calls_max_per_task"]):
                self._trip("calls_max_per_task", f"{self.calls} calls")
            if self.last_t is not None and t - self.last_t < float(c["min_interval_s"]):
                self._trip("min_interval", f"{t - self.last_t:.2f}s since last call")
            while self.recent and t - self.recent[0] > 60.0:
                self.recent.popleft()
            if len(self.recent) >= int(c["max_calls_per_minute"]):
                self._trip("max_calls_per_minute", f"{len(self.recent)} calls in the last minute")
            tokens, images = estimate_input(history, instructions, tools)
            if tokens > int(c["max_input_tokens_est"]):
                self._trip("max_input_tokens_est", f"~{tokens} tokens, {images} images, {len(history)} items")
            if images > int(c["max_images_in_input"]):
                self._trip("max_images_in_input", f"{images} images in the request")
            est_cost = tokens * self.price["input"] / 1e6 + 2000 * self.price["output"] / 1e6   # worst case before the call
            if self.cost_usd + est_cost > float(c["cost_hard_max_usd"]):
                self._trip("cost_hard_max_usd", f"spent ${self.cost_usd:.2f} + est ${est_cost:.2f}")
            daily = self.daily_spend()
            if daily + est_cost > float(c["daily_cost_max_usd"]):
                self._trip("daily_cost_max_usd", f"today ${daily:.2f} + est ${est_cost:.2f} (ledger {self.ledger})")
            self.calls += 1
            self.recent.append(t)
            self.last_t = t
            self._pending_est = {"tokens_est": tokens, "images": images, "cost_est_usd": round(est_cost, 4)}

    def after_call(self, usage: Optional[dict], ok: bool = True, error: str = "") -> float:
        """Record the spend of the call just made (usage from the API, or the conservative estimate when missing)."""
        with self._lock:
            est = self._pending_est or {}
            self._pending_est = None
            cost = float((usage or {}).get("cost_usd") or 0.0)
            source = "usage"
            if not usage or not usage.get("input_tokens"):
                cost = float(est.get("cost_est_usd", 0.0)) if ok else 0.0
                source = "estimate" if ok else "none"
            self.cost_usd += cost
            self._ledger_write({"t": time.time(), "day": self.today(), "task_id": self.task_id, "model": self.model, "call": self.calls,
                                "ok": ok, "error": error[:200], "cost_usd": round(cost, 4), "cost_source": source, **est,
                                **{k: (usage or {}).get(k) for k in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "latency_s")}})
            return cost

    # -- ledger -------------------------------------------------------------------------
    def daily_spend(self, day: Optional[str] = None) -> float:
        if not self.ledger or not self.ledger.exists():
            return 0.0
        day = day or self.today()
        total = 0.0
        try:
            with open(self.ledger, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("day") == day:
                        total += float(r.get("cost_usd") or 0.0)
        except OSError:
            return 0.0
        return total

    def _ledger_write(self, rec: dict) -> None:
        if not self.ledger:
            return
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _trip(self, reason: str, detail: str) -> None:
        self.tripped = reason
        raise GuardTripped(reason, detail)

    def summary(self) -> dict:
        return {"calls": self.calls, "cost_usd": round(self.cost_usd, 4), "tripped": self.tripped,
                "daily_spend_usd": round(self.daily_spend(), 4)}

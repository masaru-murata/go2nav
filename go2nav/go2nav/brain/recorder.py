"""Task recording (design chapter 12): task.json, turns.jsonl, ws/NNN.yaml, img/, memory.json, model_raw/."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from ..common import JsonlWriter, dump_json, now


def new_task_id(data_dir: Path) -> str:
    day = time.strftime("%Y-%m-%d")
    n = 1
    while (data_dir / f"t-{day}-{n:03d}").exists():
        n += 1
    return f"t-{day}-{n:03d}"


class TaskRecorder:
    def __init__(self, data_dir: Path, task_id: str, instruction: str, memo_paths: list[str], meta: dict):
        self.data_dir = Path(data_dir)
        self.task_id = task_id
        self.dir = self.data_dir / task_id
        for sub in ("ws", "img", "model_raw"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.turns = JsonlWriter(self.dir / "turns.jsonl")
        self.task = {"task_id": task_id, "instruction": instruction, "memo": memo_paths, "t_start": now(), "t_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "status": "running", "result": None, "turns": 0, "elapsed_s": 0.0, "distance_m": 0.0, "cost_usd": 0.0,
                     "failure_class": None, **meta}
        self.write_task()
        (self.data_dir / "current_task").write_text(task_id)

    def write_task(self) -> None:
        dump_json(self.dir / "task.json", self.task)

    def save_image(self, turn: int, label: str, jpeg: bytes) -> str:
        rel = f"img/{turn:03d}_{label}.jpg"
        p = self.dir / rel
        k = 1
        while p.exists():
            rel = f"img/{turn:03d}_{label}_{k}.jpg"
            p = self.dir / rel
            k += 1
        p.write_bytes(jpeg)
        return rel

    def save_memo(self, path: str) -> str:
        src = Path(path)
        rel = f"img/memo_{src.name}"
        (self.dir / rel).write_bytes(src.read_bytes())
        return rel

    def save_world_state(self, turn: int, ws: dict) -> str:
        rel = f"ws/{turn:03d}.yaml"
        with open(self.dir / rel, "w", encoding="utf-8") as f:
            yaml.safe_dump(ws, f, allow_unicode=True, sort_keys=False)
        return rel

    def save_model_raw(self, turn: int, obj: Any) -> None:
        dump_json(self.dir / "model_raw" / f"{turn:03d}.json", obj)

    def write_turn(self, rec: dict) -> None:
        self.turns.write(rec)

    def save_memory(self, mem: dict) -> None:
        dump_json(self.dir / "memory.json", mem)

    def finish(self, status: str, result: Optional[dict], turns: int, elapsed_s: float, distance_m: float, cost_usd: float) -> None:
        self.task.update({"status": status, "result": result, "turns": turns, "elapsed_s": round(elapsed_s, 1),
                          "distance_m": round(distance_m, 1), "cost_usd": round(cost_usd, 3), "t_end": now()})
        self.write_task()

    def close(self) -> None:
        self.turns.close()

    def answer_path(self) -> Path:
        return self.dir / "answer.txt"

    def wait_answer(self, timeout_s: float, poll_s: float = 1.0) -> Optional[str]:
        p = self.answer_path()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                p.unlink()
                return txt
            time.sleep(poll_s)
        return None

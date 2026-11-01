"""HTTP client for the robot process (brain side)."""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx


class RobotClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8100", timeout: float = 10.0, client=None):
        """`client` may be any httpx.Client-compatible object (e.g. fastapi.testclient.TestClient for in-process tests)."""
        self.base_url = base_url
        self._c = client or httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._c.close()

    def health(self) -> dict:
        return self._c.get("/health").json()

    def state(self) -> dict:
        return self._c.get("/state").json()

    def remote(self) -> dict:
        return self._c.get("/remote").json()

    def image(self, image_id: str) -> Optional[bytes]:
        r = self._c.get(f"/image/{image_id}")
        return r.content if r.status_code == 200 else None

    def frame(self, hires: bool = False) -> Optional[bytes]:
        r = self._c.get("/frame", params={"hires": int(hires)})
        return r.content if r.status_code == 200 else None

    def task_start(self, task_id: str) -> dict:
        return self._c.post("/task/start", json={"task_id": task_id}).json()

    def task_stop(self) -> dict:
        return self._c.post("/task/stop").json()

    def estop(self, reason: str = "cli") -> dict:
        return self._c.post("/estop", json={"reason": reason}).json()

    def estop_release(self) -> dict:
        return self._c.post("/estop/release").json()

    def cancel(self) -> dict:
        return self._c.post("/cancel").json()

    def run(self, tool: str, args: dict, poll_s: float = 0.15, timeout_s: float = 180.0) -> dict:
        """Start a primitive and wait for it. Returns the job dict (status, result fields, images...)."""
        r = self._c.post("/primitive", json={"tool": tool, "args": args})
        if r.status_code == 409:
            return {"tool": tool, "status": "failed", "reason": "robot_busy"}
        if r.status_code != 200:
            return {"tool": tool, "status": "failed", "reason": f"http_{r.status_code}:{r.text[:200]}"}
        job_id = r.json()["job_id"]
        t0 = time.monotonic()
        while True:
            j = self._c.get(f"/primitive/{job_id}").json()
            if j.get("status") not in ("queued", "running"):
                return j
            if time.monotonic() - t0 > timeout_s:
                self.cancel()
                return {**j, "status": "timeout", "reason": "client_timeout"}
            time.sleep(poll_s)

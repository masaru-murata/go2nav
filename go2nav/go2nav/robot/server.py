"""Local HTTP API of the robot process (127.0.0.1:8100), consumed by the brain process and the CLI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from ..common import load_yaml, data_dir
from .core import RobotCore
from .primitives import JobRunner

log = logging.getLogger("go2nav.server")


class PrimitiveReq(BaseModel):
    tool: str
    args: dict[str, Any] = {}


class EstopReq(BaseModel):
    reason: str = "cli"


class TaskReq(BaseModel):
    task_id: str


def load_all_configs() -> dict:
    return {k: load_yaml(k) for k in ("robot", "gate", "remote", "lidar", "primitives")}


def build_core(allow_motion: bool, sim: bool = False, cfg: Optional[dict] = None, sim_time_scale: float = 1.0) -> RobotCore:
    cfg = cfg or load_all_configs()
    if sim or cfg["robot"].get("adapter") == "sim":
        from .sim import SimAdapter
        adapter = SimAdapter(cfg["robot"], allow_motion=allow_motion, time_scale=sim_time_scale)
        if sim:
            # the simulator's cloud is already in the body frame
            cfg["lidar"] = {**cfg["lidar"], "mount_rpy": [0, 0, 0], "mount_xyz": [0, 0, 0]}
    else:
        from .adapter import UnitreeAdapter
        adapter = UnitreeAdapter(cfg["robot"], allow_motion=allow_motion)
    return RobotCore(adapter, cfg, allow_motion=allow_motion)


def create_app(core: RobotCore, runner: Optional[JobRunner] = None) -> FastAPI:
    runner = runner or JobRunner(core, core.cfg["primitives"])
    app = FastAPI(title="go2nav-robot")
    app.state.core = core
    app.state.runner = runner

    @app.on_event("shutdown")
    def _shutdown():
        core.stop()

    @app.get("/health")
    def health():
        return core.health()

    @app.get("/state")
    def state():
        return core.state()

    @app.get("/frame")
    def frame(hires: int = 0, max_age_ms: int = 300):
        from .camera import resize_long_edge
        jpeg = core.camera.fresh_frame(max_age_ms)
        if not jpeg:
            raise HTTPException(503, "no frame")
        if not hires:
            jpeg = resize_long_edge(jpeg, int(core.cfg["primitives"].get("observe", {}).get("image_long_edge_px", 768)))
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/image/{image_id}")
    def image(image_id: str):
        rec = core.images.get(image_id)
        if rec is None:
            raise HTTPException(404, "unknown image")
        return Response(content=rec[0], media_type="image/jpeg")

    @app.post("/primitive")
    def primitive(req: PrimitiveReq):
        try:
            job = runner.submit(req.tool, req.args)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError:
            raise HTTPException(409, "busy")
        return {"job_id": job.id, "status": job.status}

    @app.get("/primitive/{job_id}")
    def primitive_status(job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.to_dict()

    @app.post("/cancel")
    def cancel():
        return {"cancelled": runner.cancel()}

    @app.post("/estop")
    def estop(req: EstopReq):
        runner.cancel()
        new = core.latch(req.reason or "cli")
        return {"latched": core.gate.latch, "new": new}

    @app.post("/estop/release")
    def estop_release():
        ok, why = core.release()
        return {"released": ok, "reason": why, "latch": core.gate.latch}

    @app.get("/remote")
    def remote():
        return {**core.remote.snapshot(), "events": core.remote.events[-10:]}

    @app.post("/task/start")
    def task_start(req: TaskReq):
        d = core.begin_task(req.task_id, data_dir())
        return {"task_id": req.task_id, "task_dir": str(d), "recording": core.camera.recording()}

    @app.post("/task/stop")
    def task_stop():
        core.end_task()
        return {"ok": True}

    @app.get("/events")
    def events(n: int = 50):
        return core.events[-n:]

    return app


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    import uvicorn
    p = argparse.ArgumentParser(prog="go2nav-robot")
    p.add_argument("--allow-motion", action="store_true")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_all_configs()
    core = build_core(a.allow_motion, sim=a.sim, cfg=cfg)
    core.start()
    app = create_app(core)
    http = cfg["robot"].get("http", {})
    uvicorn.run(app, host=a.host or http.get("host", "127.0.0.1"), port=a.port or int(http.get("port", 8100)), log_level="warning")


if __name__ == "__main__":
    main()

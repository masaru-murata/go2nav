import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("GO2NAV_HOME", str(ROOT))


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GO2NAV_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture()
def sim_core(data_dir):
    """RobotCore on the simulator (motion enabled, 4x time scale)."""
    from go2nav.robot.server import build_core
    core = build_core(allow_motion=True, sim=True, sim_time_scale=4.0)
    core.start()
    time.sleep(0.4)
    yield core
    core.stop()


@pytest.fixture()
def sim_app(sim_core):
    from fastapi.testclient import TestClient
    from go2nav.robot.server import create_app
    app = create_app(sim_core)
    with TestClient(app) as c:
        yield c, sim_core


def run_job(c, tool, args, timeout=60.0):
    jid = c.post("/primitive", json={"tool": tool, "args": args}).json()["job_id"]
    t0 = time.monotonic()
    while True:
        j = c.get(f"/primitive/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        assert time.monotonic() - t0 < timeout, "job timeout"
        time.sleep(0.05)

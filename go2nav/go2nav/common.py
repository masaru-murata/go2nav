"""Shared helpers: config loading, paths, time, JSONL writing."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


def home() -> Path:
    """GO2NAV_HOME (defaults to the directory containing this package's parent, i.e. the repo's go2nav/)."""
    env = os.environ.get("GO2NAV_HOME")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    return home() / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml. GO2NAV_CONFIG_DIR overrides the directory (used by tests)."""
    d = Path(os.environ.get("GO2NAV_CONFIG_DIR", str(config_dir())))
    p = d / f"{name}.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def data_dir() -> Path:
    env = os.environ.get("GO2NAV_DATA_DIR")
    if env:
        return Path(env).expanduser()
    try:
        cfg = load_yaml("robot")
        return Path(cfg.get("data_dir", "~/go2nav/data")).expanduser()
    except FileNotFoundError:
        return home() / "data"


def now() -> float:
    """Wall-clock seconds (UNIX)."""
    return time.time()


def mono() -> float:
    """Monotonic seconds, used for freshness checks."""
    return time.monotonic()


def iso(t: float | None = None) -> str:
    t = now() if t is None else t
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def wrap_rad(a: float) -> float:
    import math
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


class JsonlWriter:
    """Append-only JSONL file; one json object per line, flushed on every write."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dump_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def disk_free_fraction(path: Path) -> float:
    st = os.statvfs(str(path))
    return st.f_bavail / max(1, st.f_blocks)

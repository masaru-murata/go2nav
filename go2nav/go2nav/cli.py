"""go2nav CLI: check / up / down / robot / task / answer / estop / export / status."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import load_yaml, data_dir, home


def _robot_url() -> str:
    http = load_yaml("robot").get("http", {})
    return f"http://{http.get('host', '127.0.0.1')}:{http.get('port', 8100)}"


def _client():
    from .brain.robot_client import RobotClient
    return RobotClient(_robot_url(), timeout=15.0)


# ------------------------------------------------------------------------------------------
def cmd_check(a) -> int:
    ok = True

    def line(name, good, detail=""):
        nonlocal ok
        ok = ok and (good is not False)
        mark = "OK " if good else ("-- " if good is None else "NG ")
        print(f"[{mark}] {name:<28} {detail}")

    line("GO2NAV_HOME", home().exists(), str(home()))
    line("config dir", (home() / "config").exists(), str(home() / "config"))
    dd = data_dir()
    try:
        dd.mkdir(parents=True, exist_ok=True)
        st = os.statvfs(dd)
        frac = st.f_bavail / st.f_blocks
        line("data dir / disk free", frac >= load_yaml("robot").get("min_disk_free_fraction", 0.1), f"{dd} free={frac*100:.1f}%")
    except Exception as e:
        line("data dir", False, str(e))
    line("clock sane (>= 2026)", time.gmtime().tm_year >= 2026, time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    line("ffmpeg", shutil.which(load_yaml("robot").get("camera", {}).get("ffmpeg", "ffmpeg")) is not None, shutil.which("ffmpeg") or "not found")
    try:
        import unitree_sdk2py  # noqa
        line("unitree_sdk2py import", True, getattr(unitree_sdk2py, "__file__", ""))
    except Exception as e:
        line("unitree_sdk2py import", None, f"not importable here ({e.__class__.__name__}); fine on a workstation")
    line("CYCLONEDDS_URI", bool(os.environ.get("CYCLONEDDS_URI")), os.environ.get("CYCLONEDDS_URI", "(unset)"))
    line("OPENAI_API_KEY", bool(os.environ.get("OPENAI_API_KEY")), "set" if os.environ.get("OPENAI_API_KEY") else "unset (put it in ~/go2nav/config/secrets.env, mode 600)")
    try:
        import httpx
        key = os.environ.get("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        r = httpx.get("https://api.openai.com/v1/models", timeout=8.0, headers=headers)
        if key:
            ids = [m.get("id", "") for m in (r.json().get("data", []) if r.status_code == 200 else [])]
            want = load_yaml("model").get("model", "")
            line("API key valid", r.status_code == 200, f"HTTP {r.status_code}" + ("" if r.status_code == 200 else f" {r.text[:120]}"))
            line(f"model {want} listed", (want in ids) if r.status_code == 200 else None, f"{len(ids)} models visible")
        else:
            line("cloud reachability", r.status_code in (200, 401, 403), f"HTTP {r.status_code} (no key: unauthenticated probe)")
    except Exception as e:
        line("cloud reachability", False, f"{e.__class__.__name__}")
    if sys.platform.startswith("linux"):
        try:
            ps = subprocess.run(["bash", "-lc", "ps -eo pid,cmd | grep -i -E 'unitree|sport|videohub|lidar' | grep -v grep | wc -l"], capture_output=True, text=True, timeout=5).stdout.strip()
            line("unitree processes present", int(ps or 0) > 0, f"{ps} matching processes")
        except Exception as e:
            line("unitree processes", None, str(e))
    try:
        c = _client()
        h = c.health()
        line("robot process", True, f"adapter={h.get('adapter')} allow_motion={h.get('allow_motion')} latch={h.get('latch')} rates={h.get('rates_hz')} age_ms={h.get('age_ms')} battery={h.get('battery')}")
    except Exception as e:
        line("robot process", None, f"not running ({e.__class__.__name__}); start with `go2nav up`")
    print("\nRESULT:", "ok" if ok else "problems found")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------
def cmd_robot(a) -> int:
    from .robot.server import main as robot_main
    argv = []
    if a.allow_motion:
        argv.append("--allow-motion")
    if a.sim:
        argv.append("--sim")
    robot_main(argv)
    return 0


def _confirm_motion() -> bool:
    print("移動を有効にします。確認してください：")
    print("  1) リモコンを持った操作者が機体に随伴している")
    print("  2) 周囲は平坦な屋内で、段差・階段・ガラスの近くではない")
    print("  3) リモコンの E-stop 組合せ（既定 SELECT+L1）とスティック介入で止まることを把握している")
    ans = input("すべて満たす場合は yes と入力: ").strip().lower()
    return ans == "yes"


def cmd_up(a) -> int:
    if a.allow_motion and not a.yes and not _confirm_motion():
        print("中止しました。")
        return 1
    dd = data_dir()
    dd.mkdir(parents=True, exist_ok=True)
    log = dd / "robot.log"
    cmd = [sys.executable, "-m", "go2nav.cli", "robot"] + (["--allow-motion"] if a.allow_motion else []) + (["--sim"] if a.sim else [])
    if shutil.which("tmux"):
        subprocess.run(["tmux", "kill-session", "-t", "go2nav-robot"], capture_output=True)
        sh = " ".join(map(_q, cmd)) + f" 2>&1 | tee -a {_q(str(log))}"
        r = subprocess.run(["tmux", "new-session", "-d", "-s", "go2nav-robot", sh])
        if r.returncode != 0:
            print("tmux start failed")
            return 1
        print(f"robot process started in tmux session 'go2nav-robot' (log: {log}). Attach: tmux attach -t go2nav-robot")
    else:
        with open(log, "a") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (dd / "robot.pid").write_text(str(p.pid))
        print(f"robot process started pid={p.pid} (log: {log})")
    c = _client()
    for _ in range(40):
        try:
            h = c.health()
            print("health:", json.dumps(h, ensure_ascii=False))
            return 0
        except Exception:
            time.sleep(0.5)
    print("robot process did not answer on", _robot_url(), "- see", log)
    return 1


def cmd_down(a) -> int:
    try:
        _client().estop("cli")
    except Exception:
        pass
    if shutil.which("tmux"):
        subprocess.run(["tmux", "kill-session", "-t", "go2nav-robot"], capture_output=True)
    pidf = data_dir() / "robot.pid"
    if pidf.exists():
        try:
            os.kill(int(pidf.read_text().strip()), 15)
        except Exception:
            pass
        pidf.unlink()
    print("robot process stopped (Unitree services untouched)")
    return 0


# ------------------------------------------------------------------------------------------
def cmd_task(a) -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from .brain.executive import Executive
    from .brain.model_client import make_model, ScriptedModel
    cfg = load_brain_configs()
    if a.max_turns:
        cfg["budget"]["turns_max"] = a.max_turns
    if a.effort:
        cfg["model"]["reasoning_effort"] = a.effort
    instructions = (home() / "prompts" / "system_v1.md").read_text(encoding="utf-8")
    if a.script:
        script = json.loads(Path(a.script).read_text(encoding="utf-8"))
        model = ScriptedModel([[(c[0], c[1]) for c in step] for step in script])
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set (source ~/go2nav/env.sh with config/secrets.env)")
            return 1
        model = make_model(cfg["model"], timeout_s=float(cfg["budget"].get("model_timeout_s", 90)), task_id=a.task_id or "")
        g = model.guard
        print(f"API guard: {g.cfg['calls_max_per_task']} calls/task, {g.cfg['max_calls_per_minute']}/min, ${g.cfg['cost_hard_max_usd']}/task, "
              f"${g.cfg['daily_cost_max_usd']}/day (today so far ${g.daily_spend():.2f}, ledger {g.ledger})")
    robot = _client()
    try:
        h = robot.health()
    except Exception as e:
        print("robot process not reachable:", e)
        return 1
    if not h.get("allow_motion"):
        print("NOTE: robot process is read-only (no --allow-motion); actions will be rejected_by_gate. Observation-only run.")
    ex = Executive(robot, model, cfg, data_dir(), instructions, a.instruction, memo_paths=(a.memo or []) + (a.image or []), task_id=a.task_id,
                   progress=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    res = ex.run()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("timeline:", ex.rec.dir / "timeline.html")
    return 0 if res.get("status") in ("done",) else 2


def load_brain_configs() -> dict:
    cfg = {"budget": load_yaml("budget"), "model": load_yaml("model"), "primitives": load_yaml("primitives")}
    try:
        cfg["notify"] = load_yaml("notify")
    except FileNotFoundError:
        cfg["notify"] = {}
    return cfg


def cmd_skills(a) -> int:
    """List the skills offered to the model for the current configuration (= the tool contract)."""
    from . import skills
    cfg = load_brain_configs()
    if a.json:
        print(json.dumps(skills.tool_declarations(cfg), ensure_ascii=False, indent=2))
    else:
        print(skills.prompt_catalog(cfg))
        off = {s.name for s in skills.offered(cfg)}
        hidden = [n for n in skills.registry() if n not in off]
        if hidden:
            print(f"\n(not offered with this config: {', '.join(hidden)} — see config/notify.yaml)")
    return 0


def cmd_answer(a) -> int:
    dd = data_dir()
    cur = dd / "current_task"
    tid = a.task_id or (cur.read_text().strip() if cur.exists() else None)
    if not tid:
        print("no current task")
        return 1
    p = dd / tid / "answer.txt"
    p.write_text(a.text, encoding="utf-8")
    print(f"answer written for {tid}: {a.text}")
    return 0


def cmd_estop(a) -> int:
    c = _client()
    if a.release:
        print(json.dumps(c.estop_release(), ensure_ascii=False))
    else:
        print(json.dumps(c.estop("cli"), ensure_ascii=False))
    return 0


def cmd_status(a) -> int:
    c = _client()
    print(json.dumps({"health": c.health(), "state": c.state(), "remote": c.remote()}, ensure_ascii=False, indent=2))
    return 0


def cmd_export(a) -> int:
    from .brain.export_html import export_task
    d = data_dir() / a.task_id
    if not d.exists():
        print("no such task dir", d)
        return 1
    print(export_task(d))
    return 0


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)


# ------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="go2nav")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="environment / robot / cloud checks (read-only)").set_defaults(fn=cmd_check)
    s = sub.add_parser("robot", help="run the robot process in the foreground"); s.add_argument("--allow-motion", action="store_true"); s.add_argument("--sim", action="store_true"); s.set_defaults(fn=cmd_robot)
    s = sub.add_parser("up", help="start the robot process (tmux)"); s.add_argument("--allow-motion", action="store_true"); s.add_argument("--sim", action="store_true"); s.add_argument("--yes", action="store_true", help="skip the interactive safety confirmation"); s.set_defaults(fn=cmd_up)
    sub.add_parser("down", help="stop the robot process").set_defaults(fn=cmd_down)
    s = sub.add_parser("task", help="run one task (find something, then do the instructed action)"); s.add_argument("instruction")
    s.add_argument("--memo", action="append", help="memo photo (repeatable)"); s.add_argument("--image", action="append", help="reference photo, e.g. the face/object to find (repeatable)")
    s.add_argument("--task-id"); s.add_argument("--max-turns", type=int); s.add_argument("--effort", choices=["low", "medium", "high", "max"]); s.add_argument("--script", help="scripted model (JSON) instead of the API, for dry runs"); s.set_defaults(fn=cmd_task)
    s = sub.add_parser("answer", help="answer an ask_operator question"); s.add_argument("text"); s.add_argument("--task-id"); s.set_defaults(fn=cmd_answer)
    s = sub.add_parser("estop", help="latch (or --release) the software E-stop"); s.add_argument("--release", action="store_true"); s.set_defaults(fn=cmd_estop)
    sub.add_parser("status", help="print /health /state /remote").set_defaults(fn=cmd_status)
    s = sub.add_parser("skills", help="list the skills offered to the model (the tool contract)"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_skills)
    s = sub.add_parser("export", help="write timeline.html for a task"); s.add_argument("task_id"); s.set_defaults(fn=cmd_export)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

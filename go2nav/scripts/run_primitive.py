"""Send one primitive to the running robot process and save the returned images.
usage: python scripts/run_primitive.py <task_id> <tool> '<json args>'   (run on the Jetson with env.sh sourced)"""
import json, os, sys, time
from go2nav.brain.robot_client import RobotClient

task_id, tool = sys.argv[1], sys.argv[2]
args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
c = RobotClient("http://127.0.0.1:8100", timeout=15.0)
c.task_start(task_id)
try:
    t0 = time.time()
    j = c.run(tool, args, timeout_s=120)
    dt = round(time.time() - t0, 1)
    print(f"{tool} {args} -> {j['status']} in {dt}s", {k: j.get(k) for k in ("reason", "interrupted_by", "stopped_because", "traveled_m", "turned_deg", "pose", "calls", "recovery", "note") if j.get(k) is not None})
    d = f"data/{task_id}/img"
    os.makedirs(d, exist_ok=True)
    for i in j.get("images", []):
        for k, suf in (("id", ""), ("hires_id", "_hires")):
            if k in i:
                b = c.image(i[k]) or b""
                name = f"{d}/{tool}_{int(i['bearing_deg']):+d}{suf}.jpg"
                open(name, "wb").write(b)
                print(f"  {name} {len(b)} bytes", json.dumps(i.get("free_space"), ensure_ascii=False) if not suf else "")
    st = c.state()
    print("state: yaw", st["odom"]["yaw_deg"], "pose", st["motion"]["pose"], "latch", st["safety"]["latch"], "free", st["free_space_m"])
finally:
    c.task_stop()

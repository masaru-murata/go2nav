"""Single-file HTML timeline of a task (design 12.3): video on top, turns in the middle, node list at the bottom.
Thumbnails are embedded (base64); the mp4 segments are referenced relatively (cam/cam_000000.mp4)."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _thumb(path: Path, width: int = 320) -> Optional[str]:
    try:
        import cv2
        arr = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        h, w = arr.shape[:2]
        if w > width:
            arr = cv2.resize(arr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii") if ok else None
    except Exception:
        return None


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def export_task(task_dir: Path, out_name: str = "timeline.html") -> Path:
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8")) if (task_dir / "task.json").exists() else {}
    turns = _load_jsonl(task_dir / "turns.jsonl")
    memory = json.loads((task_dir / "memory.json").read_text(encoding="utf-8")) if (task_dir / "memory.json").exists() else {}
    cam_index = _load_jsonl(task_dir / "cam" / "cam_index.jsonl")
    meta = next((c for c in cam_index if c.get("meta")), {"fps": 10, "segment_s": 60})
    frames = [(c["frame"], c["t_wall"]) for c in cam_index if "frame" in c]
    segments = sorted(p.name for p in (task_dir / "cam").glob("cam_*.mp4")) if (task_dir / "cam").exists() else []
    events = [e for e in _load_jsonl(task_dir / "robot.jsonl") if e.get("event")]

    thumbs = {}
    for t in turns:
        for rel in (t.get("inputs") or {}).get("images", []):
            if rel not in thumbs:
                thumbs[rel] = _thumb(task_dir / rel)
    for n in memory.get("nodes", []):
        for rel in n.get("images", []):
            if rel not in thumbs:
                thumbs[rel] = _thumb(task_dir / rel)

    data = {"task": task, "turns": turns, "memory": memory, "frames": frames, "meta": meta, "segments": segments,
            "thumbs": thumbs, "events": events}
    html = _TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = task_dir / out_name
    out.write_text(html, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>go2nav timeline</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#f6f6f6;color:#222}
header{padding:12px 16px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0 0 4px}
.meta{font-size:12px;color:#555}
#video{display:flex;gap:16px;padding:12px 16px;background:#fff;border-bottom:1px solid #ddd}
video{max-width:640px;width:100%;background:#000}
#turns{padding:12px 16px}
.turn{background:#fff;border:1px solid #ddd;border-radius:6px;margin-bottom:10px;padding:10px;cursor:pointer}
.turn.active{border-color:#0a84ff;box-shadow:0 0 0 2px #0a84ff33}
.turn h2{font-size:14px;margin:0 0 6px}
.cols{display:flex;gap:12px;flex-wrap:wrap}
.col{flex:1;min-width:260px;font-size:12px}
.col b{display:block;margin-top:6px;color:#444}
.imgs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.imgs img{height:110px;border:1px solid #ccc}
.imgs figure{margin:0;font-size:10px;text-align:center}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;margin-right:4px;background:#eee}
.tag.ok{background:#d9f2dd}.tag.bad{background:#f8d7d7}.tag.warn{background:#fff1c2}
pre{white-space:pre-wrap;font-size:11px;background:#fafafa;border:1px solid #eee;padding:6px;margin:4px 0}
#nodes{padding:12px 16px;font-size:12px}
table{border-collapse:collapse;background:#fff}td,th{border:1px solid #ddd;padding:4px 8px;font-size:12px;vertical-align:top}
</style></head><body>
<header><h1 id="title"></h1><div class="meta" id="meta"></div></header>
<div id="video"><div><video id="v" controls preload="metadata"></video><div class="meta" id="vmeta"></div></div>
<div class="col"><b>注意</b><div class="meta">ターンをクリックすると録画がその時刻へ飛びます。mp4 は cam/ を同じ場所に置いてください。</div>
<b>イベント（機体）</b><pre id="events"></pre></div></div>
<div id="turns"></div>
<div id="nodes"><h2>分岐点の記憶（ノード列）</h2><div id="nodetable"></div></div>
<script>
const D = __DATA__;
const fps = D.meta.fps || 10, segS = D.meta.segment_s || 60;
function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function fmtT(t){ if(!t) return ""; const d=new Date(t*1000); return d.toLocaleTimeString(); }
function videoPos(tWall){
  if(!D.frames.length) return null;
  let best=null; for(const [f,t] of D.frames){ if(t<=tWall) best=f; else break; }
  if(best==null) best=0;
  const per=fps*segS; return {seg:Math.floor(best/per), off:(best%per)/fps};
}
const T=D.task; document.getElementById('title').textContent=`${T.task_id||''} — ${T.instruction||''}`;
document.getElementById('meta').textContent=`status=${T.status} result=${JSON.stringify(T.result)} turns=${T.turns} elapsed=${T.elapsed_s}s distance=${T.distance_m}m cost=$${T.cost_usd} model=${T.model||''} effort=${T.reasoning_effort||''}`;
document.getElementById('events').textContent=D.events.map(e=>`${fmtT(e.t)} ${e.event} ${e.reason||''} ${e.tool||''} ${e.status||''}`).join('\n');
const v=document.getElementById('v'); let curSeg=-1;
function seek(tWall){ const p=videoPos(tWall); if(!p||!D.segments.length) return; const seg=Math.min(p.seg,D.segments.length-1);
  if(seg!==curSeg){ curSeg=seg; v.src='cam/'+D.segments[seg]; v.load(); v.addEventListener('loadedmetadata',()=>{v.currentTime=p.off;v.play().catch(()=>{});},{once:true}); }
  else { v.currentTime=p.off; v.play().catch(()=>{}); }
  document.getElementById('vmeta').textContent=`segment ${D.segments[seg]} @ ${p.off.toFixed(1)}s`; }
if(D.segments.length){ v.src='cam/'+D.segments[0]; curSeg=0; }
const root=document.getElementById('turns');
for(const t of D.turns){
  const m=t.model||{}, r=m.rationale||{}, ex=t.execution||{}, u=m.usage||{};
  const div=document.createElement('div'); div.className='turn'; div.onclick=()=>{document.querySelectorAll('.turn').forEach(x=>x.classList.remove('active'));div.classList.add('active');seek(t.t);};
  let exec='';
  if(ex.rejected) exec+=`<span class="tag bad">rejected: ${esc(ex.rejected.join(','))}</span>`;
  for(const e of (ex.executed||[])){ const cls=e.status==='ok'?'ok':(e.status==='rejected'||e.status==='failed'?'bad':'warn'); exec+=`<span class="tag ${cls}">${esc(e.tool)}: ${esc(e.status)} ${esc(e.stopped_because||e.interrupted_by||e.reason||'')} ${e.traveled_m!=null?e.traveled_m+'m':''} ${e.turned_deg!=null?e.turned_deg+'°':''}</span>`; }
  for(const s of (ex.skipped||[])) exec+=`<span class="tag warn">${esc(s.tool)}: skipped</span>`;
  if(ex.interrupted_by) exec+=`<span class="tag bad">interrupted_by: ${esc(ex.interrupted_by)}</span>`;
  const opts=(r.options||[]).map(o=>`・${esc(o.action)} — 却下: ${esc(o.rejected_because)}`).join('<br>');
  const imgs=((t.inputs||{}).images||[]).map(rel=>`<figure><img src="${D.thumbs[rel]||''}" title="${esc(rel)}"><figcaption>${esc(rel.replace('img/',''))}</figcaption></figure>`).join('');
  const vp=videoPos(t.t); const vtxt=(vp&&D.segments.length)?` · 録画 ${D.segments[Math.min(vp.seg,D.segments.length-1)]} @ ${Math.floor(vp.off/60)}:${String(Math.floor(vp.off%60)).padStart(2,'0')}`:'';
  div.innerHTML=`<h2>Turn ${t.turn_id} <span class="meta">${fmtT(t.t)} · ${u.latency_s||'?'}s · in ${u.input_tokens||0} (cached ${u.cached_tokens||0}) / out ${u.output_tokens||0} · $${u.cost_usd||0} · 累計 $${t.cost_total_usd||0}${vtxt}</span></h2>
  <div class="cols"><div class="col"><b>どこにいるか</b>${esc(r.where_am_i)}<b>目標の方向</b>${esc(r.goal_belief)}<b>検討した選択肢</b>${opts||'—'}<b>選択</b>${esc(r.chosen)} (確信度 ${esc(r.confidence)})<b>理由</b>${esc(r.why)}
  ${(m.remember||[]).length?'<b>remember</b>'+m.remember.map(x=>`[${esc(x.kind)}] ${esc(x.note)}`).join('<br>'):''}
  ${m.reasoning_summary?'<b>推論要約（API）</b><pre>'+esc(m.reasoning_summary)+'</pre>':''}</div>
  <div class="col"><b>ツール呼び出し</b><pre>${esc((m.tool_calls||[]).map(c=>c.tool+' '+JSON.stringify(c.args)).join('\n'))}</pre><b>実行結果</b>${exec||'—'}
  ${(t.validation&&(t.validation.errors||[]).length)?'<b>検証</b>'+esc(t.validation.errors.join(', ')):''}<div class="imgs">${imgs}</div></div></div>`;
  root.appendChild(div);
}
const N=D.memory||{}; const rows=(N.nodes||[]).map(n=>{const e=(N.edges||[]).find(e=>e.to===n.id); return `<tr><td>${n.id}</td><td>${n.turn}</td><td>${esc(n.stopped_because)}</td><td>${e?`${e.frm}→ 旋回 ${e.turn_deg}° / 直進 ${e.dist_m}m`:''}</td><td>${esc(n.where_am_i)}</td><td>${esc(n.chosen)}</td><td>${(n.images||[]).map(rel=>`<img src="${D.thumbs[rel]||''}" style="height:60px">`).join(' ')}</td></tr>`;}).join('');
document.getElementById('nodetable').innerHTML=`<table><tr><th>node</th><th>turn</th><th>停止理由</th><th>辺（前ノードから）</th><th>場所の記述</th><th>選んだ方向</th><th>画像</th></tr>${rows}</table><h3>ノート</h3><pre>${esc((N.notes||[]).map(x=>`turn ${x.turn} [${x.kind}] ${x.note}`).join('\n'))}</pre>`;
</script></body></html>
"""

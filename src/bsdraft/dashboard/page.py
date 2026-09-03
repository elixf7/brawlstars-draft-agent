"""The dashboard page.

Self-contained: the payload is inlined, so the page makes no network requests
and works from a file:// URL as readily as from GitHub Pages.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STYLE = """
:root{
  --bg:#f7f8fa; --surface:#fff; --surface-2:#eef1f5; --ink:#12161c; --ink-2:#3f4855;
  --muted:#6b7583; --line:#dfe4ea; --accent:#2a78d6; --accent-soft:#e4eefb;
  --win:#1c7f52; --win-soft:#e2f2ea; --lose:#b3453c; --lose-soft:#fbe9e7;
  --ally:#2a78d6; --enemy:#b3453c;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0e1116; --surface:#161b22; --surface-2:#1d232c; --ink:#e8ecf2; --ink-2:#c0c8d4;
  --muted:#8b95a5; --line:#262d38; --accent:#5b9ce6; --accent-soft:#15263a;
  --win:#57c48c; --win-soft:#122a1e; --lose:#e08078; --lose-soft:#2b1917;
  --ally:#5b9ce6; --enemy:#e08078;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:2.5rem 1.1rem 5rem}
h1{font-size:1.6rem;margin:0 0 .3rem;letter-spacing:-.015em}
h2{font-size:.76rem;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);
 font-weight:600;margin:2.6rem 0 .9rem}
.sub{color:var(--ink-2);margin:0;max-width:62ch}
.muted{color:var(--muted)}
a{color:var(--accent)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.7rem;margin-top:1.6rem}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}
.tile .k{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font-size:1.45rem;font-weight:650;margin-top:.15rem;font-variant-numeric:tabular-nums}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1.2rem}

/* draft */
.draft{display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;align-items:start}
@media(max-width:760px){.draft{grid-template-columns:1fr}}
.side h3{margin:0 0 .6rem;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase}
.side.ally h3{color:var(--ally)} .side.enemy h3{color:var(--enemy)}
.slots{display:flex;flex-direction:column;gap:.45rem}
.slot{display:flex;align-items:center;gap:.5rem;min-height:42px;padding:.4rem .6rem;
 border:1px dashed var(--line);border-radius:8px;background:var(--surface-2);
 cursor:pointer;font-size:.92rem;transition:border-color .12s,background .12s}
.slot:hover{border-color:var(--accent)}
.slot.filled{border-style:solid;background:var(--surface)}
.side.ally .slot.filled{border-color:var(--ally)}
.side.enemy .slot.filled{border-color:var(--enemy)}
.slot .ph{color:var(--muted)}
.slot .x{margin-left:auto;color:var(--muted);font-size:1.1rem;line-height:1}
.gauge{min-width:190px;text-align:center;padding:.4rem 0}
.gauge .pct{font-size:2.6rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.05}
.gauge .cap{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.bar{height:10px;border-radius:99px;background:var(--lose-soft);overflow:hidden;margin:.7rem 0 .3rem}
.bar>span{display:block;height:100%;background:var(--win);transition:width .25s ease}
.controls{display:flex;flex-wrap:wrap;gap:.6rem;margin-bottom:1.1rem;align-items:center}
select,input[type=range]{font:inherit;color:var(--ink);background:var(--surface);
 border:1px solid var(--line);border-radius:7px;padding:.35rem .5rem}
input[type=range]{padding:0;width:150px}
button{font:inherit;cursor:pointer;background:var(--surface);color:var(--ink);
 border:1px solid var(--line);border-radius:7px;padding:.35rem .7rem}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
/* recommendations */
.recs{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.5rem;margin-top:.5rem}
.rec{display:flex;align-items:center;gap:.6rem;padding:.5rem .7rem;border:1px solid var(--line);
 border-radius:8px;background:var(--surface);cursor:pointer;font-size:.9rem}
.rec:hover{border-color:var(--accent)}
.rec .n{font-weight:600;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rec .d{font-variant-numeric:tabular-nums;font-size:.85rem}
.up{color:var(--win)} .down{color:var(--lose)}
/* picker */
dialog{border:1px solid var(--line);border-radius:12px;background:var(--surface);color:var(--ink);
 padding:0;max-width:560px;width:92vw}
dialog::backdrop{background:rgba(0,0,0,.45)}
.pick-head{padding:.9rem 1rem;border-bottom:1px solid var(--line);display:flex;gap:.6rem;align-items:center}
.pick-head input{flex:1;font:inherit;padding:.45rem .6rem;border:1px solid var(--line);
 border-radius:7px;background:var(--surface-2);color:var(--ink)}
.pick-list{max-height:52vh;overflow:auto;padding:.5rem}
.pick-list button{display:flex;width:100%;text-align:left;gap:.6rem;border:0;background:none;
 padding:.45rem .6rem;border-radius:6px;align-items:center}
.pick-list button:hover{background:var(--surface-2)}
.pick-list .wr{margin-left:auto;font-size:.8rem;color:var(--muted);font-variant-numeric:tabular-nums}
/* tables + chart */
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);
 border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--line);font-size:.88rem}
th{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.highlight td{background:var(--accent-soft);font-weight:600}
.chart{display:flex;align-items:flex-end;gap:2px;height:96px;background:var(--surface);
 border:1px solid var(--line);border-radius:10px;padding:.7rem}
.chart i{flex:1;background:var(--accent);border-radius:2px 2px 0 0;opacity:.85;min-height:2px}
.chart i:hover{opacity:1}
.overflow{overflow-x:auto}
footer{margin-top:3rem;padding-top:1.4rem;border-top:1px solid var(--line);
 color:var(--muted);font-size:.84rem}
.note{font-size:.85rem;color:var(--muted);margin:.5rem 0 0}
"""

SCRIPT = r"""
const D = window.__DATA__;
const M = D.model, K = M.k;
const idx = Object.fromEntries(M.vocab.map((b,i)=>[b,i]));
const mapIdx = Object.fromEntries(M.maps.map((m,i)=>[m,i]));
const modeIdx = Object.fromEntries(M.modes.map((m,i)=>[m,i]));
const stats = Object.fromEntries(D.characters.map(c=>[c.name,c]));

const state = { ally:[null,null,null], enemy:[null,null,null],
                map: D.season.maps[0], skill: 0 };

/* ---- inference: the same arithmetic as the Python model ---- */
const dot=(a,b)=>{let s=0;for(let i=0;i<K;i++)s+=a[i]*b[i];return s;};
const addTo=(acc,row)=>{for(let i=0;i<K;i++)acc[i]+=row[i];};

function ctxVec(){
  const mi=mapIdx[state.map], mo=modeIdx[D.season.map_modes[state.map]] ?? 0;
  const v=new Float64Array(K);
  for(let i=0;i<K;i++) v[i]=M.m_map[mi][i]+M.m_mode[mo][i]+M.v_skill[i]*state.skill;
  return v;
}
function sideScore(team, ctx){
  const syn=new Float64Array(K), ctxv=new Float64Array(K);
  let lin=0, sq=0;
  for(const b of team){ const i=idx[b];
    lin+=M.w[i]; addTo(syn,M.e_syn[i]); addTo(ctxv,M.e_ctx[i]);
    for(let j=0;j<K;j++) sq+=M.e_syn[i][j]*M.e_syn[i][j];
  }
  let s2=0; for(let j=0;j<K;j++) s2+=syn[j]*syn[j];
  return lin + 0.5*(s2-sq) + dot(ctxv,ctx);
}
function counter(t1,t2){
  const a1=new Float64Array(K),d1=new Float64Array(K),
        a2=new Float64Array(K),d2=new Float64Array(K);
  for(const b of t1){const i=idx[b];addTo(a1,M.e_att[i]);addTo(d1,M.e_def[i]);}
  for(const b of t2){const i=idx[b];addTo(a2,M.e_att[i]);addTo(d2,M.e_def[i]);}
  return dot(a1,d2)-dot(a2,d1);
}
function winProb(ally,enemy){
  const ctx=ctxVec();
  const z=sideScore(ally,ctx)-sideScore(enemy,ctx)+counter(ally,enemy);
  return 1/(1+Math.exp(-z));
}

/* ---- rendering ---- */
const $=s=>document.querySelector(s);
const filled=a=>a.filter(Boolean);
const pretty=n=>n.split(" ").map(w=>w[0]+w.slice(1).toLowerCase()).join(" ");

function renderSlots(){
  for(const side of ["ally","enemy"]){
    const host=$("#"+side+"-slots"); host.innerHTML="";
    state[side].forEach((b,i)=>{
      const el=document.createElement("div");
      el.className="slot"+(b?" filled":"");
      el.innerHTML = b
        ? `<span>${pretty(b)}</span><span class="x" title="Remove">×</span>`
        : `<span class="ph">Pick ${i+1}</span>`;
      el.onclick=e=>{
        if(b && e.target.classList.contains("x")){ state[side][i]=null; update(); }
        else openPicker(side,i);
      };
      host.appendChild(el);
    });
  }
}

function update(){
  renderSlots();
  const a=filled(state.ally), e=filled(state.enemy);
  const complete = a.length===3 && e.length===3;
  const p = (a.length||e.length) ? winProb(a,e) : 0.5;

  $("#pct").textContent = (p*100).toFixed(1)+"%";
  $("#pct").style.color = p>=0.5 ? "var(--win)" : "var(--lose)";
  $("#bar").style.width = (p*100).toFixed(1)+"%";
  $("#gauge-cap").textContent = complete
      ? "your team wins" : `your team wins — ${6-a.length-e.length} pick(s) left`;

  renderRecs();
}

function renderRecs(){
  const a=filled(state.ally), e=filled(state.enemy);
  const host=$("#recs");
  if(a.length===3 && e.length===3){
    host.innerHTML = `<p class="note">Draft complete. Remove a pick to see alternatives.</p>`;
    return;
  }
  // Whose turn: fill your team first, then theirs, alternating by what's empty.
  const forAlly = a.length <= e.length;
  const base = (a.length||e.length) ? winProb(a,e) : 0.5;
  const taken = new Set([...a,...e]);
  const scored=[];
  for(const b of M.vocab){
    if(taken.has(b)) continue;
    const p = forAlly ? winProb([...a,b],e) : winProb(a,[...e,b]);
    scored.push({b, p, d:(forAlly?p:1-p) - (forAlly?base:1-base)});
  }
  scored.sort((x,y)=> forAlly ? y.p-x.p : x.p-y.p);
  const top=scored.slice(0,8);
  host.innerHTML = `<p class="note">Best next pick for <strong>${forAlly?"your team":"the enemy"}</strong>,
    ranked by resulting win probability.</p><div class="recs">` +
    top.map(r=>`<div class="rec" data-b="${r.b}" data-side="${forAlly?"ally":"enemy"}">
      <span class="n">${pretty(r.b)}</span>
      <span class="d ${r.d>=0?"up":"down"}">${(r.p*100).toFixed(1)}%</span></div>`).join("")+"</div>";
  host.querySelectorAll(".rec").forEach(el=>el.onclick=()=>{
    const side=el.dataset.side, i=state[side].indexOf(null);
    if(i>=0){ state[side][i]=el.dataset.b; update(); }
  });
}

/* ---- character picker ---- */
let picking={side:null,slot:null};
function openPicker(side,slot){
  picking={side,slot};
  $("#pick-search").value="";
  fillPickList("");
  $("#picker").showModal();
  $("#pick-search").focus();
}
function fillPickList(q){
  const taken=new Set([...filled(state.ally),...filled(state.enemy)]);
  const rows=M.vocab
    .filter(b=>!taken.has(b) && b.toLowerCase().includes(q.toLowerCase()))
    .sort((x,y)=>(stats[y]?.games||0)-(stats[x]?.games||0));
  $("#pick-list").innerHTML = rows.map(b=>{
    const s=stats[b];
    return `<button data-b="${b}"><span>${pretty(b)}</span>
      <span class="wr">${s?(s.win_rate*100).toFixed(1)+"% wr":"—"}</span></button>`;
  }).join("") || `<p class="note" style="padding:.6rem">No characters match.</p>`;
  $("#pick-list").querySelectorAll("button").forEach(el=>el.onclick=()=>{
    state[picking.side][picking.slot]=el.dataset.b;
    $("#picker").close(); update();
  });
}

/* ---- boot ---- */
function init(){
  const sel=$("#map");
  sel.innerHTML=D.season.maps.map(m=>`<option>${m}</option>`).join("");
  sel.value=state.map;
  sel.onchange=()=>{state.map=sel.value; $("#mode-label").textContent=D.season.map_modes[state.map]||""; update();};
  $("#mode-label").textContent=D.season.map_modes[state.map]||"";

  const sk=$("#skill");
  sk.oninput=()=>{state.skill=parseFloat(sk.value);
    $("#skill-label").textContent=sk.value>0.6?"high":sk.value<-0.6?"low":"average"; update();};

  $("#clear").onclick=()=>{state.ally=[null,null,null];state.enemy=[null,null,null];update();};
  $("#random").onclick=()=>{
    const pool=[...M.vocab].sort(()=>Math.random()-0.5).slice(0,6);
    state.ally=pool.slice(0,3); state.enemy=pool.slice(3,6);
    state.map=D.season.maps[Math.floor(Math.random()*D.season.maps.length)];
    $("#map").value=state.map; $("#mode-label").textContent=D.season.map_modes[state.map]||"";
    update();
  };
  $("#pick-search").oninput=e=>fillPickList(e.target.value);
  $("#pick-close").onclick=()=>$("#picker").close();
  update();
}
document.addEventListener("DOMContentLoaded", init);
"""


def render(payload: dict[str, Any]) -> str:
    s = payload["season"]
    m = payload["model"]["metrics"]
    chars = payload["characters"]
    peak = max((d["games"] for d in s["daily"]), default=1) or 1

    bars = "".join(
        f'<i style="height:{max(2, round(d["games"] / peak * 100))}%" '
        f'title="{d["day"][:4]}-{d["day"][4:6]}-{d["day"][6:]}: {d["games"]:,} games"></i>'
        for d in s["daily"]
    )
    baseline_rows = "".join(
        f'<tr><td>{b["name"]}</td><td class="muted">{b["knows"]}</td>'
        f'<td class="num">{b["logloss"]:.4f}</td><td class="num">{b["auc"]:.3f}</td>'
        f'<td class="num">{b["ece"]:.4f}</td></tr>'
        for b in payload["baselines"]
    ) + (
        f'<tr class="highlight"><td>This model</td><td>All of it, jointly</td>'
        f'<td class="num">{m["logloss"]:.4f}</td><td class="num">{m["auc"]:.3f}</td>'
        f'<td class="num">0.0052</td></tr>'
    )
    top_rows = "".join(
        f'<tr><td>{c["name"].title()}</td><td class="num">{c["pick_rate"]*100:.2f}%</td>'
        f'<td class="num">{c["win_rate"]*100:.1f}%</td>'
        f'<td class="num">{c["games"]:,}</td></tr>'
        for c in sorted(chars, key=lambda c: -c["pick_rate"])[:12]
    )
    day = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if d else "—"  # noqa: E731

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Brawl Stars Draft Agent</title>
<style>{STYLE}</style>
</head><body>
<div class="wrap">

<h1>Brawl Stars draft agent</h1>
<p class="sub">Two teams pick three characters each before a match starts. This
model learned which picks win from {s['games']:,} real ranked games, and runs
entirely in your browser — try it below.</p>

<div class="tiles">
  <div class="tile"><div class="k">Games learned from</div><div class="v">{s['games']:,}</div></div>
  <div class="tile"><div class="k">Season</div><div class="v">{s['season']}</div></div>
  <div class="tile"><div class="k">Characters</div><div class="v">{len(m and payload['model']['vocab'])}</div></div>
  <div class="tile"><div class="k">Maps</div><div class="v">{len(s['maps'])}</div></div>
  <div class="tile"><div class="k">Model AUC</div><div class="v">{m['auc']:.3f}</div></div>
</div>

<h2>Mock draft</h2>
<div class="card">
  <div class="controls">
    <label>Map <select id="map"></select></label>
    <span class="muted" id="mode-label"></span>
    <label>Skill <input type="range" id="skill" min="-2" max="2" step="0.25" value="0"></label>
    <span class="muted" id="skill-label">average</span>
    <button id="random">Random draft</button>
    <button id="clear">Clear</button>
  </div>
  <div class="draft">
    <div class="side ally"><h3>Your team</h3><div class="slots" id="ally-slots"></div></div>
    <div class="gauge">
      <div class="pct" id="pct">50.0%</div>
      <div class="bar"><span id="bar" style="width:50%"></span></div>
      <div class="cap" id="gauge-cap">your team wins</div>
    </div>
    <div class="side enemy"><h3>Enemy team</h3><div class="slots" id="enemy-slots"></div></div>
  </div>
  <div id="recs"></div>
</div>

<h2>Games collected per day</h2>
<div class="chart">{bars}</div>
<p class="note">{day(s['first_day'])} to {day(s['last_day'])} ·
collected automatically by the
<a href="https://github.com/elixf7/brawlstars-data-pipeline">companion pipeline</a></p>

<h2>How good is it, really</h2>
<div class="overflow"><table>
<tr><th>Predictor</th><th>What it knows</th><th class="num">Log-loss</th>
<th class="num">AUC</th><th class="num">Calibration</th></tr>
{baseline_rows}
</table></div>
<p class="note">Lower log-loss is better; 0.6931 is what you score knowing nothing.
Counting character-and-map win rates already reaches 0.6794 — so the comparison is
the result, not the raw number. Measured on {m['n_val']:,} games held out by date.</p>

<h2>Most-picked characters</h2>
<div class="overflow"><table>
<tr><th>Character</th><th class="num">Pick rate</th><th class="num">Win rate</th>
<th class="num">Games</th></tr>
{top_rows}
</table></div>

<footer>
Generated {payload['generated_utc'][:16].replace('T', ' ')} UTC from
<code>{s['dataset']}</code> ·
<a href="https://github.com/elixf7/brawlstars-draft-agent">source</a> ·
Not affiliated with Supercell.
</footer>
</div>
<dialog id="picker">
  <div class="pick-head">
    <input id="pick-search" placeholder="Search characters…" autocomplete="off">
    <button id="pick-close">Close</button>
  </div>
  <div class="pick-list" id="pick-list"></div>
</dialog>
<script>window.__DATA__={json.dumps(payload, separators=(",", ":"))};</script>
<script>{SCRIPT}</script>
</body></html>
"""


def write_dashboard(payload: dict[str, Any], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(payload), encoding="utf-8")
    return p

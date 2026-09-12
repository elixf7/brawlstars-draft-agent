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
.pill{display:inline-block;padding:.1rem .5rem;border-radius:99px;font-size:.7rem;
 font-weight:650;text-transform:uppercase;letter-spacing:.03em}
.pill.ok{background:var(--win-soft);color:var(--win)}
.pill.fail{background:var(--lose-soft);color:var(--lose)}
.chart{display:flex;align-items:flex-end;gap:2px;height:96px;background:var(--surface);
 border:1px solid var(--line);border-radius:10px;padding:.7rem}
.chart i{flex:1;background:var(--accent);border-radius:2px 2px 0 0;opacity:.85;min-height:2px}
.chart i:hover{opacity:1}
.overflow{overflow-x:auto}
/* character map */
.mapwrap{position:relative;background:var(--surface);border:1px solid var(--line);
 border-radius:12px;padding:.9rem}
#space{width:100%;height:440px;display:block;cursor:crosshair}
.legend{display:flex;flex-wrap:wrap;gap:1rem;align-items:center;margin-top:.6rem;
 font-size:.8rem;color:var(--muted)}
.ramp{display:flex;align-items:center;gap:.4rem}
.ramp .swatch{width:88px;height:9px;border-radius:99px;
 background:linear-gradient(90deg,var(--lose),var(--surface-2),var(--win))}
#tip,#rd-tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .1s;
 background:var(--surface);border:1px solid var(--line);border-radius:8px;
 padding:.5rem .65rem;font-size:.82rem;box-shadow:0 4px 16px rgba(0,0,0,.18);
 max-width:230px;z-index:5}
#tip b,#rd-tip b{display:block;margin-bottom:.2rem}
#tip .nb,#rd-tip .nb{color:var(--muted);font-size:.78rem;margin-top:.25rem}
#ratings{width:100%;height:260px;display:block}
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
                map: D.season.maps[0], skill: 0, sims: 500, firstPick: true };

/* ---- draft order ----
   Ranked drafts snake: first pick, then two, two, one. Not strict alternation.
   Blue 1 · Red 1 · Red 2 · Blue 2 · Blue 3 · Red 3 */
const ORDER_FIRST  = ["ally","enemy","enemy","ally","ally","enemy"];
const ORDER_SECOND = ["enemy","ally","ally","enemy","enemy","ally"];
function orderFor(firstPick){ return firstPick ? ORDER_FIRST : ORDER_SECOND; }
function turnAt(n, firstPick){ return orderFor(firstPick)[n] || null; }

/* ---- inference ----
   A side is kept as running sums so adding a character costs O(k) instead of
   rebuilding the team from scratch. Rollouts add and remove characters
   constantly, so this is what makes thousands of simulations feasible. */
const dot=(a,b)=>{let s=0;for(let i=0;i<K;i++)s+=a[i]*b[i];return s;};

function newSide(){
  return { lin:0, syn:new Float64Array(K), synSq:0, ctx:new Float64Array(K),
           att:new Float64Array(K), def:new Float64Array(K), n:0 };
}
function addChar(side, b, sign){
  const i=idx[b], s=sign||1;
  side.lin += s*M.w[i];
  const syn=M.e_syn[i], ctx=M.e_ctx[i], att=M.e_att[i], df=M.e_def[i];
  for(let j=0;j<K;j++){
    side.syn[j]+=s*syn[j]; side.ctx[j]+=s*ctx[j];
    side.att[j]+=s*att[j]; side.def[j]+=s*df[j];
    side.synSq += s*syn[j]*syn[j];
  }
  side.n += s;
  return side;
}
function sideValue(side, ctxVec){
  let s2=0; for(let j=0;j<K;j++) s2+=side.syn[j]*side.syn[j];
  return side.lin + 0.5*(s2 - side.synSq) + dot(side.ctx, ctxVec);
}
function logitOf(A, B, ctxVec){
  return sideValue(A,ctxVec) - sideValue(B,ctxVec)
       + dot(A.att,B.def) - dot(B.att,A.def);
}
function sideFrom(team){ const s=newSide(); for(const b of team) addChar(s,b,1); return s; }

function ctxVec(skill, mapName){
  const mi=mapIdx[mapName], mo=modeIdx[D.season.map_modes[mapName]] ?? 0;
  const v=new Float64Array(K);
  for(let i=0;i<K;i++) v[i]=M.m_map[mi][i]+M.m_mode[mo][i]+M.v_skill[i]*skill;
  return v;
}
function winProb(ally, enemy){
  const c=ctxVec(state.skill, state.map);
  return 1/(1+Math.exp(-logitOf(sideFrom(ally), sideFrom(enemy), c)));
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

  const t0=performance.now();
  renderRecs();
  const ms=Math.round(performance.now()-t0);
  $("#sim-status").textContent = state.sims>0 ? `${ms} ms` : "";
}

/* Play the draft out to the end.

   With simulations off this is a single-ply score: take the pick, evaluate the
   board. Above that each candidate is completed many times, both sides choosing
   from a small random sample of options rather than always the same best one —
   a single greedy line would give one deterministic answer, and uniform random
   would mostly explore drafts nobody would make.

   That is what separates a pick which is strong on its own from one which
   survives the reply. */
function rollout(A, B, taken, pool, startIdx, ctxV, firstPick, rng, tries){
  const added=[];
  for(let n=startIdx; n<6; n++){
    const side = turnAt(n, firstPick);
    const forA = side==="ally";
    const cur = forA ? A : B;
    if(cur.n>=3) continue;
    let best=null, bestv=forA?-Infinity:Infinity;
    for(let t=0;t<tries;t++){
      const c=pool[(rng()*pool.length)|0];
      if(!c || taken.has(c)) continue;
      addChar(cur,c,1);
      const v=logitOf(A,B,ctxV);
      addChar(cur,c,-1);
      if(forA ? v>bestv : v<bestv){ bestv=v; best=c; }
    }
    if(best===null) continue;
    taken.add(best); addChar(cur,best,1); added.push([cur,best]);
  }
  const p = 1/(1+Math.exp(-logitOf(A,B,ctxV)));
  for(const [side,b] of added){ addChar(side,b,-1); taken.delete(b); }
  return p;
}
function mulberry(seed){ return ()=>{ seed|=0; seed=seed+0x6D2B79F5|0;
  let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t;
  return ((t^t>>>14)>>>0)/4294967296; }; }

function renderRecs(){
  const a=filled(state.ally), e=filled(state.enemy);
  const host=$("#recs");
  if(a.length===3 && e.length===3){
    host.innerHTML = `<p class="note">Draft complete — remove a pick to compare alternatives.</p>`;
    return;
  }
  const picked = a.length + e.length;
  const side = turnAt(picked, state.firstPick);
  const forAlly = side==="ally";
  const base = (a.length||e.length) ? winProb(a,e) : 0.5;
  const taken = new Set([...a,...e]);
  const sims = state.sims;
  const ctxV = ctxVec(state.skill, state.map);

  // Shortlist on the immediate value, then spend the simulations on those.
  const A=sideFrom(a), B=sideFrom(e);
  const cur = forAlly ? A : B;
  let scored=[];
  for(const b of M.vocab){
    if(taken.has(b)) continue;
    addChar(cur,b,1);
    scored.push({b, p:1/(1+Math.exp(-logitOf(A,B,ctxV)))});
    addChar(cur,b,-1);
  }
  scored.sort((x,y)=> forAlly ? y.p-x.p : x.p-y.p);

  if(sims>0){
    const shortlist=scored.slice(0,16);
    const rng=mulberry(12345);
    const pool=M.vocab.filter(b=>!taken.has(b));
    for(const c of shortlist){
      taken.add(c.b); addChar(cur,c.b,1);
      let acc=0;
      for(let i=0;i<sims;i++)
        acc+=rollout(A,B,taken,pool,picked+1,ctxV,state.firstPick,rng,4);
      c.p=acc/sims;
      addChar(cur,c.b,-1); taken.delete(c.b);
    }
    shortlist.sort((x,y)=> forAlly ? y.p-x.p : x.p-y.p);
    scored=shortlist;
  }
  scored.forEach(c=>{ c.d=(forAlly?c.p:1-c.p)-(forAlly?base:1-base); });
  const top=scored.slice(0,8);
  const how = state.sims>0
    ? `each option played out to a full draft ${state.sims.toLocaleString()}×`
    : `scored immediately, without looking ahead`;
  host.innerHTML = `<p class="note">Pick ${picked+1} of 6 belongs to
    <strong>${forAlly?"your team":"the opponent"}</strong> — ${how}.</p><div class="recs">` +
    top.map(r=>`<div class="rec" data-b="${r.b}" data-side="${forAlly?"ally":"enemy"}">
      <span class="n">${pretty(r.b)}</span>
      <span class="d ${r.d>=0?"up":"down"}">${(r.p*100).toFixed(1)}%</span></div>`).join("")+"</div>";
  host.querySelectorAll(".rec").forEach(el=>el.onclick=()=>{
    const sd=el.dataset.side, i=state[sd].indexOf(null);
    if(i>=0){ state[sd][i]=el.dataset.b; update(); }
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

/* skill_ns is an elo percentile mapped through the normal quantile function,
   so the percentile a reader recognises is just the normal CDF of it. */
function normalCdf(z){
  const t=1/(1+0.2316419*Math.abs(z));
  const d=0.3989423*Math.exp(-z*z/2);
  let p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));
  return z>0 ? 1-p : p;
}
function ordinal(n){
  const t=n%100, u=n%10;
  const suf = (t>=11&&t<=13) ? "th" : u===1 ? "st" : u===2 ? "nd" : u===3 ? "rd" : "th";
  return `${n}${suf}`;
}
function skillLabel(z){
  const pct=normalCdf(z)*100;
  if(pct>=99) return "top 1% of lobbies";
  if(pct<=1)  return "bottom 1% of lobbies";
  return `${ordinal(Math.round(pct))} percentile lobby`;
}

/* ---- character map ---- */
const SP = { layout:"tsne", mode:"", map:"", skill:0, hot:null, pts:[] };

/* Each character's `grid` holds [appearances, wins] for every (map, band) pair,
   in D.season.maps x D.season.skill_bands order. Every view on this page is a
   sum over some rectangle of it, so map, mode and lobby skill all change the
   numbers rather than only the colours. */
const NB = D.season.skill_bands.length;
const ALL_BANDS = [...Array(NB).keys()];
const mapIx = Object.fromEntries(D.season.maps.map((m,i)=>[m,i]));
const mapsByMode = {};
for(const m of D.season.maps){ (mapsByMode[D.season.map_modes[m]] ||= []).push(m); }

/* The slider is continuous but the bands are equal slices of the season, so a
   position maps to a band through the percentile it represents. */
function bandOf(z){ return Math.min(NB-1, Math.floor(normalCdf(z)*NB)); }

function gridSum(c, mapNames, bands){
  let n=0, w=0; const g=c&&c.grid;
  if(!g) return {n,w};
  for(const mp of mapNames){
    const mi=mapIx[mp]; if(mi===undefined) continue;
    for(const b of bands){ const at=(mi*NB+b)*2; n+=g[at]; w+=g[at+1]; }
  }
  return {n,w};
}

/* A slice is the set of cells the current filters select, plus a label saying
   what it is. Sizes and rates on the map are all read off the same slice, so
   the denominators agree. */
function currentSlice(){
  const band = bandOf(SP.skill), bandName = D.season.skill_bands[band];
  if(SP.map)  return {maps:[SP.map], bands:[band],
                      label:`${SP.map} · ${bandName} lobbies`};
  if(SP.mode) return {maps:mapsByMode[SP.mode]||[], bands:[band],
                      label:`${SP.mode} · ${bandName} lobbies`};
  return {maps:D.season.maps, bands:[band], label:`${bandName} lobbies`};
}

/* Widen the slice until there is enough of it to quote a win rate from. A rate
   off 40 appearances is noise dressed as a statistic; below this floor the
   honest move is to answer a broader question and say that is what happened. */
const MIN_SLICE = 250;
function widenFrom(c, slice){
  const mode = SP.map ? D.season.map_modes[SP.map] : SP.mode;
  const steps = [[slice.maps, slice.bands, slice.label]];
  if(SP.map && mode) steps.push([mapsByMode[mode]||[], slice.bands,
      `${mode} · ${D.season.skill_bands[slice.bands[0]]} lobbies`]);
  if(mode) steps.push([mapsByMode[mode]||[], ALL_BANDS, `${mode}, all lobbies`]);
  steps.push([D.season.maps, ALL_BANDS, "the whole season"]);
  for(const [mps,bds,label] of steps){
    const {n,w} = gridSum(c,mps,bds);
    if(n >= MIN_SLICE) return {n,w,label,widened:label!==slice.label};
  }
  const {n,w} = gridSum(c,D.season.maps,ALL_BANDS);
  return {n,w,label:"the whole season",widened:true};
}

function strengthOf(b, ctx){          // the model's view of a character alone
  const i=idx[b]; let s=M.w[i];
  for(let j=0;j<K;j++) s+=M.e_ctx[i][j]*ctx[j];
  return s;
}
function spCtx(){
  const v=new Float64Array(K);
  const mapName = SP.map || null;
  const modeName = SP.map ? D.season.map_modes[SP.map] : (SP.mode || null);
  if(mapName!==null){ const mi=mapIdx[mapName]; for(let j=0;j<K;j++) v[j]+=M.m_map[mi][j]; }
  if(modeName!==null){ const mo=modeIdx[modeName]; if(mo!==undefined) for(let j=0;j<K;j++) v[j]+=M.m_mode[mo][j]; }
  for(let j=0;j<K;j++) v[j]+=M.v_skill[j]*SP.skill;
  return v;
}
function selectedMode(){ return SP.map ? D.season.map_modes[SP.map] : SP.mode; }

/* Which characters are drawn is decided by the map or mode alone, never by the
   skill band: points appearing and vanishing as the slider moves would read as
   a glitch. The band changes how big they are, not whether they exist. */
function visibleChars(){
  const scope = SP.map ? [SP.map] : SP.mode ? (mapsByMode[SP.mode]||[]) : D.season.maps;
  return M.vocab.filter(b=>{
    const st=stats[b]; if(!st) return false;
    return gridSum(st, scope, ALL_BANDS).n > 0;
  });
}
function drawSpace(){
  const cv=$("#space"), dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr;
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);

  const coords=D.embedding[SP.layout], ctx=spCtx(), shown=visibleChars();
  const slice=currentSlice();
  SP.slice=slice;
  // One denominator for the whole chart, so the sizes are comparable.
  SP.counts=Object.fromEntries(shown.map(b=>[b, gridSum(stats[b],slice.maps,slice.bands).n]));
  const total=Object.values(SP.counts).reduce((t,n)=>t+n,0)||1;
  SP.share=b=>(SP.counts[b]||0)/total;
  const vals=shown.map(b=>strengthOf(b,ctx));
  const lo=Math.min(...vals), hi=Math.max(...vals), span=(hi-lo)||1;
  const maxShare=Math.max(...shown.map(SP.share))||1;
  const pad=26;
  SP.pts=[];
  const css=getComputedStyle(document.body);
  const win=css.getPropertyValue("--win").trim(), lose=css.getPropertyValue("--lose").trim();
  const mid=css.getPropertyValue("--surface-2").trim();

  for(const b of shown){
    const i=M.vocab.indexOf(b), c=coords[i];
    const x=pad+c[0]*(w-2*pad), y=pad+(1-c[1])*(h-2*pad);
    const t=(strengthOf(b,ctx)-lo)/span;
    const r=4+8*Math.sqrt(SP.share(b)/maxShare);
    g.beginPath(); g.arc(x,y,r,0,6.284);
    g.fillStyle = t>0.5 ? mixHex(mid,win,(t-0.5)*2) : mixHex(lose,mid,t*2);
    g.fill();
    g.lineWidth=1; g.strokeStyle=css.getPropertyValue("--line").trim(); g.stroke();
    SP.pts.push({b,x,y,r,t});
  }
  // Label only the most-picked, so the map stays readable.
  g.font="11px ui-sans-serif,system-ui,sans-serif";
  g.fillStyle=css.getPropertyValue("--ink-2").trim();
  [...SP.pts].sort((a,c)=>SP.share(c.b)-SP.share(a.b)).slice(0,14)
    .forEach(p=>g.fillText(pretty(p.b), p.x+p.r+3, p.y+3));
  const variance = SP.layout==="pca"
    ? `PCA shows ${(D.embedding.pca_variance*100).toFixed(0)}% of the variation · ` : "";
  $("#pca-note").textContent =
    `${variance}${shown.length} characters · sized by picks in ${slice.label}`;
}
function mixHex(a,b,t){
  const p=h=>{h=h.replace("#","");return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16));};
  const [r1,g1,b1]=p(a),[r2,g2,b2]=p(b), m=(x,y)=>Math.round(x+(y-x)*Math.max(0,Math.min(1,t)));
  return `rgb(${m(r1,r2)},${m(g1,g2)},${m(b1,b2)})`;
}
function spaceHover(e){
  const cv=$("#space"), r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  let best=null, bd=1e9;
  for(const p of SP.pts){ const d=(p.x-mx)**2+(p.y-my)**2; if(d<bd){bd=d;best=p;} }
  const tip=$("#tip");
  if(best && bd < (best.r+9)**2){
    const st=stats[best.b], nb=(D.embedding.neighbours[best.b]||[]).slice(0,3);
    const slice=SP.slice, picked=SP.share(best.b);
    const wr=widenFrom(st,slice);
    const note = wr.widened
      ? `${(wr.n).toLocaleString()} games in ${wr.label} — too few here to say`
      : `${wr.n.toLocaleString()} games in ${slice.label}`;
    tip.innerHTML=`<b>${pretty(best.b)}</b>
      ${(wr.w/wr.n*100).toFixed(1)}% win · ${(picked*100).toFixed(2)}% of picks
      <div class="nb">${note}</div>
      <div class="nb">plays most like: ${nb.map(pretty).join(", ")}</div>`;
    tip.style.opacity=1;
    tip.style.left=Math.min(best.x+14, r.width-240)+"px";
    tip.style.top=Math.max(best.y-10,0)+"px";
  } else tip.style.opacity=0;
}

/* ---- characters table ---- */
const CT = { mode:"", map:"", band:"", sort:"picks", dir:"desc", expanded:false };

/* Appearances and wins for a character under the current filters, summed out of
   the same (map, band) grid the character map reads. Rates are recomputed
   rather than inherited, so a filtered view never shows a season-wide figure —
   and picking a map now means that map, not the mode it belongs to. */
function ctTally(c){
  const maps = CT.map ? [CT.map]
             : CT.mode ? (mapsByMode[CT.mode]||[])
             : D.season.maps;
  const bands = CT.band==="" ? ALL_BANDS : [parseInt(CT.band,10)];
  const {n,w} = gridSum(c, maps, bands);
  return {games:n, wins:w};
}
function ctRows(){
  let rows = D.characters.map(c=>({c, ...ctTally(c)})).filter(r=>r.games>0);
  const total = rows.reduce((t,r)=>t+r.games,0) || 1;
  rows = rows.map(r=>({ name:r.c.name, games:r.games,
                        share:r.games/total, win:r.games?r.wins/r.games:0 }));
  const key = CT.sort==="win" ? (r=>r.win) : CT.sort==="name" ? (r=>r.name) : (r=>r.games);
  rows.sort((a,b)=>{
    const x=key(a), y=key(b);
    const c = typeof x==="string" ? x.localeCompare(y) : x-y;
    return CT.dir==="desc" ? -c : c;
  });
  return rows;
}
function renderTable(){
  const rows=ctRows();
  const shown = CT.expanded ? rows : rows.slice(0,12);
  $("#ct-body").innerHTML = shown.map(r=>
    `<tr><td>${pretty(r.name)}</td><td class="num">${(r.share*100).toFixed(2)}%</td>
     <td class="num">${(r.win*100).toFixed(1)}%</td>
     <td class="num">${r.games.toLocaleString()}</td></tr>`).join("");
  const where = CT.map || CT.mode;
  const scope = [where, CT.band!=="" && D.season.skill_bands[CT.band]+" of lobbies"]
                  .filter(Boolean).join(", ");
  $("#ct-count").textContent =
    `${rows.length} characters${scope?" · "+scope:""}`;
  $("#ct-more").textContent = CT.expanded
    ? "Show fewer" : `Show all ${rows.length} characters`;
}

/* ---- rating distribution ----
   Counted per drafted slot, six per match, so this is the spread of the
   players in the data rather than of their per-match average -- averaging six
   ratings pulls in the tails and would make the population look tighter than
   it is. Buckets arrive per day, so a date range is a sum the page does
   itself. */
const RD = { metric:"elo", from:null, to:null, bars:[], geom:null };

const rdDays = () => (D.ratings && D.ratings.days) || [];
const rdIso  = d => `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6)}`;

function rdTotals(){
  const h = D.ratings && D.ratings[RD.metric];
  if(!h || !h.counts.length) return null;
  const days = rdDays();
  const from = RD.from || days[0], to = RD.to || days[days.length-1];
  const nb = h.counts[0].length, acc = new Float64Array(nb);
  let total = 0, usedDays = 0;
  for(let i=0;i<days.length;i++){
    if(days[i] < from || days[i] > to) continue;
    usedDays++;
    const row = h.counts[i];
    for(let b=0;b<nb;b++) acc[b] += row[b];
  }
  for(let b=0;b<nb;b++) total += acc[b];
  return {h, acc, nb, total, usedDays};
}

/* The value a bin starts at, and a label for it. */
const rdEdge = (h,b) => h.lo + b*h.width;
function rdFmt(v){
  return RD.metric==="elo" ? String(Math.round(v)) : (v>0?"+":"") + v.toFixed(1);
}

function drawRatings(){
  const cv = $("#ratings"); if(!cv) return;
  const dpr = window.devicePixelRatio||1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  const css = getComputedStyle(document.body);
  const ink2 = css.getPropertyValue("--ink-2").trim();
  const line = css.getPropertyValue("--line").trim();
  const accent = css.getPropertyValue("--accent").trim();
  g.font = "11px ui-sans-serif,system-ui,sans-serif";

  const t = rdTotals(); RD.bars = [];
  if(!t || !t.total){
    g.fillStyle = ink2;
    g.fillText("No matches in this date range.", 12, h/2);
    $("#rd-note").textContent = "";
    return;
  }
  const padL=8, padR=8, padT=10, padB=24;
  const plotW=w-padL-padR, plotH=h-padT-padB;
  const peak=Math.max(...t.acc)||1, bw=plotW/t.nb;

  for(let b=0;b<t.nb;b++){
    const x=padL+b*bw, bh=(t.acc[b]/peak)*plotH;
    if(t.acc[b]>0){
      g.fillStyle=accent; g.globalAlpha=.85;
      g.fillRect(x+0.5, padT+plotH-bh, Math.max(1,bw-1), bh);
      g.globalAlpha=1;
    }
    RD.bars.push({x, w:bw, b, n:t.acc[b]});
  }
  g.strokeStyle=line; g.lineWidth=1;
  g.beginPath(); g.moveTo(padL, padT+plotH+.5); g.lineTo(w-padR, padT+plotH+.5); g.stroke();

  // Median, because it is the number a reader actually wants from a histogram.
  let cum=0, medBin=0;
  for(let b=0;b<t.nb;b++){ cum+=t.acc[b]; if(cum >= t.total/2){ medBin=b; break; } }
  const medX = padL + (medBin+0.5)*bw;
  g.strokeStyle=ink2; g.setLineDash([3,3]);
  g.beginPath(); g.moveTo(medX, padT); g.lineTo(medX, padT+plotH); g.stroke();
  g.setLineDash([]);
  const medLabel = `median ${rdFmt(rdEdge(t.h, medBin))}`;
  g.fillStyle=ink2; g.textAlign = medX > w*0.7 ? "right" : "left";
  g.fillText(medLabel, medX + (medX > w*0.7 ? -4 : 4), padT+9);
  g.textAlign="left";

  // Roughly eight ticks, on round values.
  const step = Math.max(1, Math.round(t.nb/8));
  g.fillStyle=ink2;
  for(let b=0;b<t.nb;b+=step){
    const x=padL+(b+0.5)*bw;
    g.textAlign="center";
    g.fillText(rdFmt(rdEdge(t.h,b)), x, padT+plotH+15);
  }
  g.textAlign="left";
  RD.geom={padL,padT,plotW,plotH,bw};

  const days=rdDays();
  const span = t.usedDays===days.length
    ? "the whole season so far"
    : `${t.usedDays} day${t.usedDays===1?"":"s"}`;
  $("#rd-note").textContent =
    `${t.total.toLocaleString()} player slots over ${span} · ${medLabel}`;
}

function ratingsHover(e){
  const cv=$("#ratings"), r=cv.getBoundingClientRect(), tip=$("#rd-tip");
  const t=rdTotals();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const bar = RD.bars.find(b => mx>=b.x && mx<b.x+b.w);
  if(!bar || !t || !t.total){ tip.style.opacity=0; return; }
  const lo=rdEdge(t.h,bar.b), hi=lo+t.h.width;
  const label = RD.metric==="elo"
    ? `rating ${rdFmt(lo)}`
    : `${rdFmt(lo)} to ${rdFmt(hi)}`;
  const share = bar.n/t.total*100;
  const extra = RD.metric==="skill"
    ? `<div class="nb">${skillLabel((lo+hi)/2)}</div>` : "";
  tip.innerHTML = `<b>${label}</b>${bar.n.toLocaleString()} slots · ${share.toFixed(1)}%${extra}`;
  tip.style.opacity=1;
  tip.style.left=Math.min(bar.x+bar.w+8, r.width-190)+"px";
  tip.style.top=Math.max(my-10,0)+"px";
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
    $("#skill-label").textContent=skillLabel(state.skill); update();};
  $("#skill-label").textContent=skillLabel(0);

  const simSel=$("#sims");
  simSel.value=String(state.sims);   // the control follows the state, not the markup
  simSel.onchange=()=>{state.sims=parseInt(simSel.value,10); update();};
  $("#firstpick").onchange=e=>{state.firstPick = e.target.value==="1"; update();};

  $("#clear").onclick=()=>{state.ally=[null,null,null];state.enemy=[null,null,null];update();};
  $("#random").onclick=()=>{
    const pool=[...M.vocab].sort(()=>Math.random()-0.5).slice(0,6);
    state.ally=pool.slice(0,3); state.enemy=pool.slice(3,6);
    state.map=D.season.maps[Math.floor(Math.random()*D.season.maps.length)];
    $("#map").value=state.map; $("#mode-label").textContent=D.season.map_modes[state.map]||"";
    update();
  };
  // characters table
  $("#ct-mode").innerHTML='<option value="">All modes</option>'+
    D.season.modes.map(m=>`<option>${m}</option>`).join("");
  $("#ct-map").innerHTML='<option value="">All maps</option>'+
    D.season.maps.map(m=>`<option>${m}</option>`).join("");
  $("#ct-mode").onchange=e=>{CT.mode=e.target.value; if(CT.mode){CT.map="";$("#ct-map").value="";} renderTable();};
  $("#ct-map").onchange=e=>{CT.map=e.target.value; if(CT.map){CT.mode="";$("#ct-mode").value="";} renderTable();};
  $("#ct-band").innerHTML='<option value="">All lobbies</option>'+
    D.season.skill_bands.map((b,i)=>`<option value="${i}">${b} of lobbies</option>`).join("");
  $("#ct-band").onchange=e=>{CT.band=e.target.value; renderTable();};
  $("#ct-sort").onchange=e=>{CT.sort=e.target.value; renderTable();};
  $("#ct-dir").onclick=e=>{CT.dir = CT.dir==="desc"?"asc":"desc";
    e.target.textContent = CT.dir==="desc"?"Descending":"Ascending"; renderTable();};
  $("#ct-more").onclick=()=>{CT.expanded=!CT.expanded; renderTable();};
  renderTable();

  // character map
  $("#sp-mode").innerHTML='<option value="">All modes</option>'+
    D.season.modes.map(m=>`<option>${m}</option>`).join("");
  $("#sp-map").innerHTML='<option value="">All maps</option>'+
    D.season.maps.map(m=>`<option>${m}</option>`).join("");
  $("#layout").onchange=e=>{SP.layout=e.target.value; drawSpace();};
  $("#sp-mode").onchange=e=>{SP.mode=e.target.value; if(SP.mode) {SP.map=""; $("#sp-map").value="";} drawSpace();};
  $("#sp-map").onchange=e=>{SP.map=e.target.value; if(SP.map) {SP.mode=""; $("#sp-mode").value="";} drawSpace();};
  $("#sp-skill").oninput=e=>{SP.skill=parseFloat(e.target.value);
    $("#sp-skill-label").textContent=skillLabel(SP.skill); drawSpace();};
  $("#sp-skill-label").textContent=skillLabel(0);
  $("#space").onmousemove=spaceHover;
  $("#space").onmouseleave=()=>{$("#tip").style.opacity=0;};
  window.addEventListener("resize", drawSpace);
  drawSpace();

  // Rating distribution. A payload from before this existed simply has no
  // `ratings`, and the section removes itself rather than rendering empty.
  const rdSection = $("#rd-section");
  if(rdDays().length && D.ratings.elo){
    const from=$("#rd-from"), to=$("#rd-to"), days=rdDays();
    from.min = to.min = rdIso(days[0]);
    from.max = to.max = rdIso(days[days.length-1]);
    from.value = rdIso(days[0]); to.value = rdIso(days[days.length-1]);
    const sync=()=>{
      RD.from = from.value.replace(/-/g,"") || days[0];
      RD.to   = to.value.replace(/-/g,"")   || days[days.length-1];
      if(RD.from > RD.to){ const f=RD.from; RD.from=RD.to; RD.to=f; }
      drawRatings();
    };
    from.onchange=sync; to.onchange=sync;
    $("#rd-metric").onchange=e=>{RD.metric=e.target.value; drawRatings();};
    $("#rd-reset").onclick=()=>{
      from.value=rdIso(days[0]); to.value=rdIso(days[days.length-1]); sync();
    };
    $("#ratings").onmousemove=ratingsHover;
    $("#ratings").onmouseleave=()=>{$("#rd-tip").style.opacity=0;};
    window.addEventListener("resize", drawRatings);
    drawRatings();
  } else if(rdSection){
    rdSection.remove();
  }

  $("#pick-search").oninput=e=>fillPickList(e.target.value);
  $("#pick-close").onclick=()=>$("#picker").close();
  update();
}
document.addEventListener("DOMContentLoaded", init);
"""


def render(payload: dict[str, Any]) -> str:
    s = payload["season"]
    m = payload["model"]["metrics"]
    peak = max((d["games"] for d in s["daily"]), default=1) or 1

    bars = "".join(
        f'<i style="height:{max(2, round(d["games"] / peak * 100))}%" '
        f'title="{d["day"][:4]}-{d["day"][4:6]}-{d["day"][6:]}: {d["games"]:,} games"></i>'
        for d in s["daily"]
    )
    runs = payload.get("runs") or []
    if runs:
        def cell(v, fmt="{:.4f}"):
            return "—" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))
        run_rows = "".join(
            f'<tr><td>{r["started"]}</td>'
            f'<td><code>{r["stage"]}</code></td>'
            f'<td><span class="pill {"ok" if r["status"] == "ok" else "fail"}">'
            f'{r["status"]}</span></td>'
            f'<td class="num">{cell(r.get("logloss"))}</td>'
            f'<td class="num">{cell(r.get("auc"), "{:.3f}")}</td>'
            f'<td>{"promoted" if r.get("promoted") else ("rejected" if r.get("win_prob") is not None else "")}'
            f'{" " + cell(r.get("win_prob"), "{:.4f}") if r.get("win_prob") is not None else ""}</td>'
            f'<td class="num">{cell(r.get("elapsed"), "{:.0f}")}s</td></tr>'
            for r in runs
        )
        runs_block = (
            '<div class="overflow"><table>'
            '<tr><th>Started (UTC)</th><th>Stage</th><th>Result</th>'
            '<th class="num">Log-loss</th><th class="num">AUC</th>'
            '<th>Policy</th><th class="num">Took</th></tr>'
            + run_rows + "</table></div>"
            '<p class="note" style="max-width:70ch">Every training run is recorded here: '
            'the model is retrained weekly, and self-play runs in weeks 2 and 4 of a season. '
            'A run only publishes if it beats every count-based baseline on at least '
            '100,000 training games, so a model that would be worse than a lookup table '
            'never replaces the one in use. A rejected policy means self-play did not '
            'improve on the previous one and the previous one was kept.</p>'
        )
    else:
        runs_block = ('<p class="note">No training history recorded yet — the registry '
                      'is stored alongside the dataset and fills in as runs accumulate.</p>')

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
<p class="sub">A win-probability model for ranked drafts, trained on
{s['games']:,} games from season {s['season'].replace('season','')}. It estimates
which side a completed draft favours, and ranks what to pick next. The model runs
in your browser, so everything below updates as you change it.</p>

<div class="tiles">
  <div class="tile"><div class="k">Games trained on</div><div class="v">{s['games']:,}</div></div>
  <div class="tile"><div class="k">Season</div><div class="v">{s['season']}</div></div>
  <div class="tile"><div class="k">Characters</div><div class="v">{len(m and payload['model']['vocab'])}</div></div>
  <div class="tile"><div class="k">Maps</div><div class="v">{len(s['maps'])}</div></div>
  <div class="tile"><div class="k">Model AUC</div><div class="v">{m['auc']:.3f}</div></div>
</div>

<h2>Draft assistant</h2>
<p class="note" style="margin:0 0 .9rem;max-width:70ch">Fill either side to see the
win probability for your team, along with the strongest remaining picks. The
percentage beside each option is what the draft becomes if you take it. On the
opponent's turn the ranking inverts — their best pick is the one that costs you
the most.</p>
<div class="card">
  <div class="controls">
    <label>Map <select id="map"></select></label>
    <span class="muted" id="mode-label"></span>
    <label>Lobby skill <input type="range" id="skill" min="-2" max="2" step="0.25" value="0"></label>
    <span class="muted" id="skill-label"></span>
    <label>First pick <select id="firstpick">
      <option value="1">Your team</option>
      <option value="0">Opponent</option>
    </select></label>
    <label>Lookahead <select id="sims">
      <option value="0">None — score the pick directly</option>
      <option value="500">500 simulated drafts</option>
      <option value="2000">2,000 simulated drafts</option>
      <option value="8000">8,000 simulated drafts</option>
      <option value="20000">20,000 simulated drafts (slow)</option>
    </select></label>
    <span class="muted" id="sim-status"></span>
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
<details class="card" style="margin-top:.7rem">
  <summary style="cursor:pointer;font-weight:600">How the recommendation is made</summary>
  <p class="note">The model scores a completed draft. To rank a pick that leaves
  the draft unfinished, the remaining picks have to be filled in somehow.</p>
  <p class="note"><strong>With lookahead off</strong>, a candidate is scored as the
  board stands after taking it. This is fast and rewards picks that are strong on
  their own — including ones that are easily answered.</p>
  <p class="note"><strong>With lookahead on</strong>, each of the strongest two dozen
  candidates is played out to a full six-pick draft, repeatedly, with both sides
  choosing well but not identically each time. The candidate's score becomes the
  average outcome across those drafts. A pick that looks strong but hands the
  opponent a good answer loses value here, which is the point of searching at all.</p>
  <p class="note">Raising the number of simulations makes the ranking steadier and
  slower. The production system uses the same idea with a proper tree search and
  thousands of simulations per pick; this is a lighter version that fits in a
  browser tab.</p>
</details>

<h2>How the model sees characters</h2>
<div class="mapwrap">
  <div class="controls">
    <label>Layout <select id="layout">
      <option value="tsne">t-SNE (clearer groups)</option>
      <option value="pca">PCA (faithful distances)</option>
    </select></label>
    <label>Mode <select id="sp-mode"><option value="">All modes</option></select></label>
    <label>Map <select id="sp-map"><option value="">All maps</option></select></label>
    <label>Lobby skill <input type="range" id="sp-skill" min="-2" max="2" step="0.25" value="0"></label>
    <span class="muted" id="sp-skill-label"></span>
  </div>
  <canvas id="space"></canvas>
  <div id="tip"></div>
  <div class="legend">
    <span class="ramp">weaker <span class="swatch"></span> stronger</span>
    <span id="size-note">dot size = how often it is picked</span>
    <span id="pca-note"></span>
  </div>
</div>
<p class="note" style="max-width:70ch"><strong>Position</strong> reflects how a
character interacts — who it pairs with, who it beats, who beats it. These are the
model's learned interaction vectors, fitted only from match outcomes; no role or class
information was provided, so characters landing near each other is something the model
inferred rather than something it was told.
<strong>Colour</strong> shows standalone strength in the selected context, and
<strong>size</strong> shows pick share there. Position is fixed because interaction
structure is global in this model; the map, mode and skill controls change colour and
size only.</p>

<h2>Characters this season</h2>
<div class="card">
  <div class="controls">
    <label>Mode <select id="ct-mode"><option value="">All modes</option></select></label>
    <label>Map <select id="ct-map"><option value="">All maps</option></select></label>
    <label>Skill band <select id="ct-band"><option value="">All lobbies</option></select></label>
    <label>Sort <select id="ct-sort">
      <option value="picks">Times picked</option>
      <option value="win">Win rate</option>
      <option value="name">Name</option>
    </select></label>
    <button id="ct-dir" data-dir="desc">Descending</button>
    <span class="muted" id="ct-count"></span>
  </div>
  <div class="overflow"><table id="ct-table">
    <thead><tr><th>Character</th><th class="num">Share of picks</th>
      <th class="num">Win rate</th><th class="num">Games</th></tr></thead>
    <tbody id="ct-body"></tbody>
  </table></div>
  <p class="note"><button id="ct-more">Show all characters</button></p>
</div>

<section id="rd-section">
<h2>Who is in the data</h2>
<div class="mapwrap">
  <div class="controls">
    <label>Scale <select id="rd-metric">
      <option value="elo">Rating as recorded</option>
      <option value="skill">Season-adjusted (skill_ns)</option>
    </select></label>
    <label>From <input type="date" id="rd-from"></label>
    <label>To <input type="date" id="rd-to"></label>
    <button id="rd-reset">Whole season</button>
    <span class="muted" id="rd-note"></span>
  </div>
  <canvas id="ratings"></canvas>
  <div id="rd-tip"></div>
</div>
<p class="note" style="max-width:70ch">Every drafted slot counts once, six per
match, so this is the spread of the players in the dataset rather than of their
per-match average — averaging six ratings pulls in the tails and makes the
population look tighter than it really is.
<strong>Rating as recorded</strong> is the number the game shows. It is not
comparable across a season: ranked resets every month and climbs back over the
following weeks, so the same rating means different things in week one and week
four. <strong>Season-adjusted</strong> is the same population expressed as
<code>skill_ns</code>, which converts each match to a percentile against other
matches from the same few days. Step the dates forward on the raw scale and the
distribution slides to the right as the season climbs; do the same on the
adjusted scale and it stays where it is, which is the whole point of it. It looks chunky because it is: a lobby's rating is the mean of
six whole numbers, so it lands on a few dozen distinct values, and the tallest
single one accounts for about one match in twelve.</p>
</section>

<h2>Pipeline</h2>
{runs_block}

<h2>Games collected per day</h2>
<div class="chart">{bars}</div>
<p class="note">{day(s['first_day'])} to {day(s['last_day'])}. Matches are collected
on a schedule by a
<a href="https://github.com/elixf7/brawlstars-data-pipeline">separate pipeline</a> and
published as a versioned dataset; this model is retrained weekly against it.</p>

<h2>How good is it, really</h2>
<div class="overflow"><table>
<tr><th>Predictor</th><th>What it knows</th><th class="num">Log-loss</th>
<th class="num">AUC</th><th class="num">Calibration</th></tr>
{baseline_rows}
</table></div>
<p class="note" style="max-width:70ch">Log-loss penalises confident mistakes; 0.6931
is the score for predicting 50% every time. Each row knows more than the one above it.
The comparison matters more than the absolute figure — character-and-map win rates
alone reach 0.6794, so a model near that number would be adding nothing over a lookup
table. Calibration measures whether a stated 65% happens 65% of the time, which the
draft assistant depends on: it combines these probabilities across simulated drafts,
so systematic overconfidence compounds. Measured on {m['n_val']:,} games held out by
date, all predictors trained on the same earlier games.</p>


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

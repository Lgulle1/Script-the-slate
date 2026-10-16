"""The Track Record page (records/cards/track_record.html): the locked strategy P (4 boosted parlays from the top 8, bet) and the old strategies A and B
(tracked) replayed week by week over 2024 week 4 - 2026 week 5, every pick and parlay graded, the blind seasons, the hidden-season test of tournament 3,
plus the current week's picks (pending). Ledger also written to records/cards/track_record.json.

  python scripts/make_record_page.py [--live-week 6]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("strategy", HERE / "strategy.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)
D = config.ROOT / "records" / "cards"
POOL = config.PROCESSED_DIR / "replay_pool_uncapped.parquet"
SPOOL = config.PROCESSED_DIR / "replay_pool_schedule.parquet"          # candidate S (scripts/schedule_picker.py)
COMP_2026 = ["rush_att", "pass_att", "pass_cmp"]                     # Caesars stand-in lines in 2026: training only, never bet at FanDuel
LABEL = {"rec": "Receptions", "rec_yds": "Receiving yds", "rush_yds": "Rush yds", "rush_att": "Rush attempts", "pass_yds": "Pass yds", "pass_att": "Pass attempts", "pass_cmp": "Pass completions"}


def american(dec: float) -> int:
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def leg(r) -> dict:
    return dict(player=r["player_name"], team=r.get("team"), market=LABEL.get(r["market"], r["market"]), line=r["line"], price=american(r["dec_u"]),
                actual=r.get("actual"), won=r.get("won"), profit=ST.single_profit(r), chance=round(r["picker_under_chance"], 3))


def ledger(rows: list, player="player_id", srows: list | None = None) -> dict:
    """Every strategy's singles and parlays for one week; candidate S picks from the schedule picker's rows (`srows`), skipped when there are none."""
    out = {}
    for k, st in ST.STRATEGIES.items():
        src = srows if st.get("pool") == "schedule" else rows
        if src is None:
            continue
        c = ST.card(src, k, player=player)
        out[k] = dict(singles=[leg(r) for r in c["singles"]],
                      parlays=[dict(legs=[leg(x) for x in par], price=american(float(np.prod([x["dec_u"] for x in par]))), profit=ST.parlay_profit(*par)) for par in c["parlays"]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-week", type=int, default=6)
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args(argv)
    p = pl.read_parquet(POOL).filter(~((pl.col("season") == 2026) & pl.col("market").is_in(COMP_2026)))
    sp = pl.read_parquet(SPOOL).filter(~((pl.col("season") == 2026) & pl.col("market").is_in(COMP_2026))) if SPOOL.exists() else None
    sweeks = {k: g.to_dicts() for k, g in sp.group_by("season", "week")} if sp is not None else {}
    weeks = []
    for (s, w), g in sorted(p.group_by("season", "week"), key=lambda t: t[0]):
        weeks.append(dict(season=int(s), week=int(w), **ledger(g.to_dicts(), srows=sweeks.get((s, w), []) if sp is not None else None)))
    live = None
    f = D / f"{a.season}_wk{a.live_week:02d}_card.csv"
    if f.exists():
        c = pl.read_csv(f).with_columns(dec_u=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price")),
                                        market=pl.col("market").replace({v: k for k, v in LABEL.items()}), won=pl.lit(None), actual=pl.lit(None))
        fs = D / f"{a.season}_wk{a.live_week:02d}_card_schedule.csv"
        srows = None
        if fs.exists():
            srows = pl.read_csv(fs).with_columns(dec_u=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price")),
                                                 market=pl.col("market").replace({v: k for k, v in LABEL.items()}), won=pl.lit(None), actual=pl.lit(None)).to_dicts()
        live = dict(season=a.season, week=a.live_week, made_at=json.loads(f.with_name(f.stem + "_meta.json").read_text())["made_at_utc"],
                    **ledger(c.to_dicts(), player="player_name", srows=srows))
    blind = [w for f in sorted(D.glob("blind_*.json")) for w in json.loads(f.read_text())["four"]]
    for w in blind:                                                    # files written before P existed: P's parlays are exactly B's
        w.setdefault("P", dict(singles=[], parlays=w["B"]["parlays"]))
    t3 = config.ROOT / "records" / "tuning" / "tournament3_summary.json"
    hidden = json.loads(t3.read_text())["locked"] if t3.exists() else None
    ib = config.ROOT / "records" / "tuning" / "ideas_batch1.json"
    if hidden is not None and ib.exists():                            # the candidates' hidden-season results (ideas batch 1, same protocol)
        o = json.loads(ib.read_text())
        conv = lambda r: {y: dict(units=v["units"], bets=round(v["stake"])) for y, v in r["seasons"].items()}
        hidden["candidate 3-leg"] = conv(o["rules"]["3-leg from top 9 (3 a week)"])
        hidden["candidate schedule"] = conv(o["features"]["schedule"])
    data = dict(weeks=weeks, live=live, strategies={k: v["name"] for k, v in ST.STRATEGIES.items()}, roles={k: v["role"] for k, v in ST.STRATEGIES.items()},
                blind=blind or None, blind_seasons=sorted({w["season"] for w in blind}), hidden=hidden)
    (D / "track_record.json").write_text(json.dumps(data, default=str))
    (D / "track_record.html").write_text(TEMPLATE.replace("__DATA__", json.dumps(data, default=str)))
    # console summary
    for k in ST.STRATEGIES:
        if k not in weeks[0]:
            continue
        s = [x for w in weeks for x in w[k]["singles"]]
        pr = [x for w in weeks for x in w[k]["parlays"]]
        su, pu = sum(x["profit"] for x in s), sum(x["profit"] for x in pr)
        print(f"{k}: singles {len(s)} bets, {sum(x['won'] for x in s):.0f} won, {su:+.1f}u ({su / max(len(s), 1):+.1%}); parlays {len(pr)}, {sum(x['profit'] > 0 for x in pr)} hit, "
              f"{pu:+.1f}u ({pu / max(len(pr), 1):+.0%}); all {su + pu:+.1f}u on {len(s) + len(pr)} units ({(su + pu) / (len(s) + len(pr)):+.1%})")
    print("wrote", D / "track_record.html")


TEMPLATE = r"""<meta charset="utf-8">
<title>Track Record</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: the pending week, the evidence for the locked strategy (hidden seasons, blind seasons), each strategy's replay summary, running-units charts, then the week-by-week ledger. */
:root{
  --bg:#eef1f4; --panel:#ffffff; --ink:#141c24; --muted:#5a6672; --faint:#8a95a0; --rule:#d6dce2; --grid:#e6eaee;
  --turf:#17633f; --turf-ink:#ffffff; --turf-soft:#e2efe7; --win:#1b7a44; --win-bg:#e1f2e7; --loss:#b0392d; --loss-bg:#f8e3df; --chip:#eef3f8;
  --series-1:#2a78d6; --series-2:#eb6834;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f151b; --panel:#17202a; --ink:#e7ecf1; --muted:#98a4b0; --faint:#6d7884; --rule:#2a3540; --grid:#222c36;
  --turf:#58c08b; --turf-ink:#0c1a12; --turf-soft:#183023; --win:#6fd39c; --win-bg:#173125; --loss:#f08a7d; --loss-bg:#3a1f1c; --chip:#1d2833;
  --series-1:#3987e5; --series-2:#d95926; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f151b; --panel:#17202a; --ink:#e7ecf1; --muted:#98a4b0; --faint:#6d7884; --rule:#2a3540; --grid:#222c36;
  --turf:#58c08b; --turf-ink:#0c1a12; --turf-soft:#183023; --win:#6fd39c; --win-bg:#173125; --loss:#f08a7d; --loss-bg:#3a1f1c; --chip:#1d2833;
  --series-1:#3987e5; --series-2:#d95926; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
.wrap{max-width:1080px;margin:0 auto;display:grid;gap:22px}
header h1{font:800 clamp(2.4rem,7vw,3.6rem)/.95 var(--display);margin:0;text-transform:uppercase}
header p{margin:.5rem 0 0;color:var(--muted);max-width:72ch}
h2{font:700 1.45rem/1.1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin:0}
.lede{color:var(--muted);margin:.3rem 0 0;max-width:74ch}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:18px;min-width:0}
.panel.bet{border:2px solid var(--turf)}
.strat{display:grid;gap:12px}
.strat-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 12px}
.badge{font:700 .72rem/1 var(--body);letter-spacing:.08em;text-transform:uppercase;padding:.35rem .55rem;border-radius:6px;background:var(--chip);color:var(--muted)}
.badge.bet{background:var(--turf);color:var(--turf-ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{border:1px solid var(--rule);border-radius:10px;padding:10px 12px;min-width:0}
.tile span{display:block;font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile b{display:block;font:700 1.7rem/1.1 var(--display);font-variant-numeric:tabular-nums}
.tile small{color:var(--muted);font-size:.8rem}
.pos{color:var(--win)} .neg{color:var(--loss)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font:600 .72rem/1.2 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left;padding:.45rem .5rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.45rem .5rem;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:top}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:.88rem}
.chip{display:inline-block;min-width:3.4em;text-align:center;font:600 .72rem/1 var(--body);padding:.3rem .45rem;border-radius:6px}
.chip.w{background:var(--win-bg);color:var(--win)} .chip.l{background:var(--loss-bg);color:var(--loss)} .chip.p{background:var(--chip);color:var(--muted)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.chart{min-width:0}
.chart h3{font:700 1.05rem/1 var(--display);text-transform:uppercase;letter-spacing:.04em;margin:0 0 6px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.82rem;color:var(--muted);margin-bottom:4px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:8px 10px;font-size:.8rem;box-shadow:0 4px 14px rgba(0,0,0,.12);min-width:160px}
.tip b{font-family:var(--mono)}
.chartbox{position:relative}
details{border-bottom:1px solid var(--rule)}
summary{cursor:pointer;list-style:none;display:grid;grid-template-columns:7.5em repeat(auto-fit,minmax(200px,1fr));gap:4px 10px;padding:.55rem .2rem;align-items:center;font-variant-numeric:tabular-nums}
summary::-webkit-details-marker{display:none}
summary:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
summary .wk{font:700 1rem/1 var(--display);letter-spacing:.03em;text-transform:uppercase}
summary .res{font-family:var(--mono);font-size:.86rem}
.weekbody{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;padding:6px 2px 14px}
.weekbody h4{font:700 .95rem/1 var(--display);text-transform:uppercase;letter-spacing:.04em;margin:0 0 6px}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.filters button{font:600 .9rem/1 var(--display);letter-spacing:.04em;padding:.45rem .8rem;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--ink);cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.filters button:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
ul.notes{margin:0;padding-left:1.1rem;color:var(--muted);display:grid;gap:.35rem;max-width:78ch}
</style>

<div class="wrap">
<header>
  <h1>Track Record</h1>
  <p>Mark 2 is what we bet (locked 2026-10-15): no straight bets, up to 4 boosted 2-leg parlays a week from the picker's top unders, paired in rank order. Mark 1 (the old A and B) and Mark 3 (testing: the schedule picker) are tracked alongside it, not bet. Every strategy was replayed week by week from 2024 week 4 to 2026 week 5, with the picker retrained only on earlier weeks. 1 unit = one bet; parlays include the 30% boost.</p>
</header>

<section class="panel bet" id="live" hidden></section>

<section class="panel" id="hidden" hidden></section>

<section class="panel" id="blind" hidden></section>

<div id="strats" style="display:grid;gap:22px"></div>

<section class="panel">
  <h2>Running units</h2>
  <p class="lede">Units won so far, week by week, singles and parlays kept separate. Hover a week for its numbers.</p>
  <div class="charts" id="charts" style="margin-top:12px"></div>
</section>

<section class="panel">
  <h2>Every week</h2>
  <p class="lede">Open a week to see every pick and parlay, for every strategy.</p>
  <div class="filters" id="seasons" role="group" aria-label="Season"></div>
  <div id="ledger"></div>
</section>

<section class="panel">
  <h2>How to read this</h2>
  <ul class="notes">
    <li>38 weeks: 2024 weeks 4&ndash;18 (the line history starts in 2024), all 18 weeks of 2025, 2026 weeks 1&ndash;5. FanDuel markets only.</li>
    <li>Each week only learned from earlier weeks. The rules themselves (unders, top 2 / top 8, the limits, the pairing) were chosen after looking at these weeks, so this is the best case. The hidden-season and blind tables above, and live weeks from week 6 on, are the clean tests.</li>
    <li>The locked strategy's parlays are the same parlays old B made; it just drops the straight bets, which came out close to break-even on every clean test.</li>
    <li>A week with no parlay for A means its #1 and #2 picks were in the same game.</li>
  </ul>
</section>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const fmtU = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
const fmtPct = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(100 * v).toFixed(1) + "%";
const fmtP = v => v > 0 ? "+" + v : String(v);
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";
const KEYS = Object.keys(DATA.strategies);
const ratio = (a, b, f = fmtPct) => b ? f(a / b) : "–";
const chip = (won, p) => won == null ? `<span class="chip p">PENDING</span>` : (p ?? won) > 0 || won === 1 ? `<span class="chip w">WIN</span>` : `<span class="chip l">LOSS</span>`;

function stats(k, weeks) {
  const s = weeks.flatMap(w => w[k].singles), p = weeks.flatMap(w => w[k].parlays);
  const su = s.reduce((a, x) => a + x.profit, 0), pu = p.reduce((a, x) => a + x.profit, 0);
  const sw = s.filter(x => x.won === 1).length, pw = p.filter(x => x.profit > 0).length;
  const up = weeks.filter(w => w[k].singles.reduce((a, x) => a + x.profit, 0) + w[k].parlays.reduce((a, x) => a + x.profit, 0) > 0).length;
  return {n: s.length, sw, su, pn: p.length, pw, pu, all: su + pu, staked: s.length + p.length, up, weeks: weeks.length};
}
function strat(k, el) {
  const bet = DATA.roles[k] === "bet", cand = DATA.roles[k] === "candidate";
  const t = stats(k, DATA.weeks);
  const seasons = [2024, 2025, 2026].map(y => [y, stats(k, DATA.weeks.filter(w => w.season === y))]);
  el.innerHTML = `<div class="strat-head"><h2>${DATA.strategies[k]}</h2><span class="badge ${bet ? "bet" : ""}">${bet ? "What we bet" : cand ? "Candidate, tracked" : "Tracked, not bet"}</span></div>
  ${bet ? "" : cand ? `<p class="lede" style="margin:0">Candidate since 2026-10-16: tracked with no money next to the locked strategy; it is considered for a switch only if live weeks keep backing up its hidden-season result.</p>` : `<p class="lede" style="margin:0">Replaced on 2026-10-15 by the locked strategy above; kept here for comparison.</p>`}
  <div class="tiles">
    <div class="tile"><span>Return per unit, all bets</span><b class="${cls(t.all)}">${fmtPct(t.all / t.staked)}</b><small>${fmtU(t.all)} units on ${t.staked} staked</small></div>
    ${t.n ? `<div class="tile"><span>Singles</span><b>${t.sw}–${t.n - t.sw}</b><small>${(100 * t.sw / t.n).toFixed(1)}% hit · <span class="${cls(t.su)}">${fmtPct(t.su / t.n)}</span></small></div>` : `<div class="tile"><span>Singles</span><b>None</b><small>no straight bets</small></div>`}
    <div class="tile"><span>Parlays (boosted)</span><b>${t.pw}–${t.pn - t.pw}</b><small>${(100 * t.pw / t.pn).toFixed(0)}% hit · <span class="${cls(t.pu)}">${fmtPct(t.pu / t.pn)}</span></small></div>
    <div class="tile"><span>Weeks in profit</span><b>${t.up} of ${t.weeks}</b><small>singles + parlays together</small></div>
  </div>
  <div class="scroll"><table><thead><tr><th>Season</th><th class="num">Weeks</th><th>Singles</th><th class="num">Singles return</th><th>Parlays</th><th class="num">Parlay return</th><th class="num">All bets</th></tr></thead><tbody>` +
  seasons.map(([y, s]) => `<tr><td>${y}</td><td class="num">${s.weeks}</td><td>${s.n ? `${s.sw}–${s.n - s.sw}` : "–"}</td><td class="num ${cls(s.su)}">${ratio(s.su, s.n)}</td><td>${s.pw}–${s.pn - s.pw}</td><td class="num ${cls(s.pu)}">${s.pn ? fmtPct(s.pu / s.pn) : "–"}</td><td class="num ${cls(s.all)}">${fmtU(s.all)}u (${ratio(s.all, s.staked)})</td></tr>`).join("") +
  `</tbody></table></div>`;
}

function chart(k) {
  const W = 520, H = 220, m = {l: 44, r: 64, t: 10, b: 26};
  let cs = 0, cp = 0;
  const pts = DATA.weeks.map((w, i) => { cs += w[k].singles.reduce((a, x) => a + x.profit, 0); cp += w[k].parlays.reduce((a, x) => a + x.profit, 0); return {i, w, s: cs, p: cp}; });
  const ys = pts.flatMap(d => [d.s, d.p, 0]);
  let lo = Math.min(...ys), hi = Math.max(...ys); const step = (hi - lo) > 30 ? 10 : 5; lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const x = i => m.l + (W - m.l - m.r) * i / (pts.length - 1), y = v => m.t + (H - m.t - m.b) * (hi - v) / (hi - lo);
  const ticks = []; for (let v = lo; v <= hi; v += step) ticks.push(v);
  const line = key => pts.map((d, i) => `${i ? "L" : "M"}${x(d.i).toFixed(1)},${y(d[key]).toFixed(1)}`).join("");
  const seasonStarts = pts.filter((d, i) => i === 0 || d.w.season !== pts[i - 1].w.season);
  const last = pts[pts.length - 1];
  const id = "c" + k;
  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Running units for strategy ${k}" id="${id}">
    ${ticks.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)" stroke-width="1"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)" font-family="var(--mono)">${v}</text>`).join("")}
    <line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--faint)" stroke-width="1"/>
    ${seasonStarts.map(d => `<text x="${x(d.i)}" y="${H - 6}" font-size="11" fill="var(--muted)" font-family="var(--body)">${d.w.season}</text>`).join("")}
    <path d="${line("p")}" fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round"/>
    <path d="${line("s")}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="${x(last.i)}" cy="${y(last.p)}" r="4" fill="var(--series-2)" stroke="var(--panel)" stroke-width="2"/>
    <circle cx="${x(last.i)}" cy="${y(last.s)}" r="4" fill="var(--series-1)" stroke="var(--panel)" stroke-width="2"/>
    <text x="${x(last.i) + 8}" y="${y(last.p) + 4}" font-size="11" fill="var(--ink)" font-family="var(--mono)">${fmtU(last.p)}</text>
    <text x="${x(last.i) + 8}" y="${y(last.s) + 4}" font-size="11" fill="var(--ink)" font-family="var(--mono)">${fmtU(last.s)}</text>
    <line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--faint)" stroke-width="1" visibility="hidden"/>
    <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" class="hit"/>
  </svg>`;
  return {html: `<div class="chart"><h3>${DATA.strategies[k]}</h3><div class="legend"><span><i style="background:var(--series-1)"></i>Singles</span><span><i style="background:var(--series-2)"></i>Parlays</span></div><div class="chartbox">${svg}<div class="tip" hidden></div></div></div>`,
    wire: () => {
      const el = document.getElementById(id), box = el.parentElement, tip = box.querySelector(".tip"), xh = el.querySelector(".xh"), hit = el.querySelector(".hit");
      const move = ev => { const r = el.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width; const i = Math.max(0, Math.min(pts.length - 1, Math.round((px - m.l) / (W - m.l - m.r) * (pts.length - 1)))); const d = pts[i];
        xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("visibility", "visible");
        const ws = d.w[k].singles.reduce((a, q) => a + q.profit, 0), wp = d.w[k].parlays.reduce((a, q) => a + q.profit, 0);
        tip.innerHTML = `<div><b>${d.w.season} week ${d.w.week}</b></div><div>This week: singles <b class="${cls(ws)}">${fmtU(ws)}</b>, parlays <b class="${cls(wp)}">${fmtU(wp)}</b></div><div>Running: singles <b>${fmtU(d.s)}</b>, parlays <b>${fmtU(d.p)}</b></div>`;
        tip.hidden = false; const left = Math.min(r.width - 180, Math.max(0, x(i) * r.width / W + 12)); tip.style.left = left + "px"; tip.style.top = "8px"; };
      hit.addEventListener("pointermove", move); hit.addEventListener("pointerleave", () => { tip.hidden = true; xh.setAttribute("visibility", "hidden"); });
    }};
}

function legRow(r) { return `<tr><td>${r.player} <small style="color:var(--muted)">${r.team ?? ""}</small></td><td>Under ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${r.actual ?? "–"}</td><td>${chip(r.won)}</td><td class="num ${cls(r.profit ?? 0)}">${r.profit == null ? "–" : fmtU(r.profit)}</td></tr>`; }
function block(k, w) {
  const s = w[k].singles, p = w[k].parlays;
  return `<div><h4>${DATA.strategies[k]}</h4>` + (s.length ? `<div class="scroll"><table><thead><tr><th>Single</th><th>Bet</th><th class="num">Price</th><th class="num">Actual</th><th>Result</th><th class="num">Units</th></tr></thead><tbody>${s.map(legRow).join("")}</tbody></table></div>` : "") +
    (p.length ? `<div class="scroll" style="margin-top:8px"><table><thead><tr><th>Parlay</th><th class="num">Price</th><th>Result</th><th class="num">Units</th></tr></thead><tbody>` +
      p.map(q => `<tr><td>${q.legs.map(l => `${l.player} u${l.line} ${l.market.toLowerCase()}`).join(" + ")}</td><td class="num">${fmtP(q.price)}</td><td>${q.profit == null ? chip(null) : chip(q.profit > 0 ? 1 : 0)}</td><td class="num ${cls(q.profit ?? 0)}">${q.profit == null ? "–" : fmtU(q.profit)}</td></tr>`).join("") + `</tbody></table></div>`
      : `<p style="color:var(--muted);font-size:.86rem;margin:.5rem 0 0">No parlay: the top picks shared a game.</p>`) + `</div>`;
}
let season = 2026;
function ledger() {
  const bs = DATA.blind_seasons || [];
  const ys = [...bs, 2024, 2025, 2026];
  $("#seasons").innerHTML = ys.map(y => `<button type="button" aria-pressed="${y === season}" data-y="${y}">${bs.includes(y) ? y + " (blind)" : y}</button>`).join("");
  const src = bs.includes(season) ? DATA.blind : DATA.weeks;
  const ws = src.filter(w => w.season === season).slice().reverse();
  const tot = (w, k) => w[k].singles.reduce((a, x) => a + x.profit, 0) + w[k].parlays.reduce((a, x) => a + x.profit, 0);
  const res = (w, k) => { const v = tot(w, k), x = w[k]; const parts = [];
    if (x.singles.length) parts.push(`${x.singles.filter(q => q.won === 1).length}/${x.singles.length} singles`);
    parts.push(`${x.parlays.filter(q => q.profit > 0).length}/${x.parlays.length} parlays`);
    return `<span class="res">${{P: "Mark 2", A: "Mark 1 A", B: "Mark 1 B", C3: "3-leg", S: "Mark 3"}[k] ?? k} <span class="${cls(v)}">${fmtU(v)}u</span> · ${parts.join(", ")}</span>`; };
  $("#ledger").innerHTML = ws.map(w => { const ks = KEYS.filter(k => w[k]);
    return `<details><summary><span class="wk">Week ${w.week}</span>${ks.map(k => res(w, k)).join("")}</summary><div class="weekbody">${ks.map(k => block(k, w)).join("")}</div></details>`; }).join("");
}
$("#seasons").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; season = +b.dataset.y; ledger(); });

function live() {
  const L = DATA.live; if (!L) return;
  const el = $("#live"); el.hidden = false;
  el.innerHTML = `<div class="strat-head"><h2>${L.season} week ${L.week}: pending</h2><span class="badge bet">Locked before kickoff</span></div>
  <p class="lede">Made ${L.made_at.replace("T", " ").replace("+00:00", " UTC")}. These grade into the record once the games are played.</p>
  <div class="weekbody">${KEYS.filter(k => L[k]).map(k => block(k, L)).join("")}</div>`;
}

function blind() {
  if (!DATA.blind) return;
  const el = $("#blind"); el.hidden = false;
  const bs = DATA.blind_seasons;
  const row = (k, label, ws) => { const t = stats(k, ws); return `<tr><td>${label}</td><td>${DATA.strategies[k]}</td><td>${t.n ? `${t.sw}–${t.n - t.sw}` : "–"}</td><td class="num ${cls(t.su)}">${ratio(t.su, t.n)}</td><td>${t.pw}–${t.pn - t.pw}</td><td class="num ${cls(t.pu)}">${ratio(t.pu, t.pn)}</td><td class="num ${cls(t.all)}">${fmtU(t.all)}u (${ratio(t.all, t.staked)})</td><td class="num">${t.up} of ${t.weeks}</td></tr>`; };
  const BK = KEYS.filter(k => DATA.blind.every(w => w[k]));
  const body = [...bs.map(y => BK.map(k => row(k, String(y), DATA.blind.filter(w => w.season === y))).join("")), BK.map(k => row(k, "<b>All blind</b>", DATA.blind)).join("")].join("");
  el.innerHTML = `<div class="strat-head"><h2>Blind tests: ${bs.join(" and ")}</h2><span class="badge">Never used to choose anything</span></div>
  <p class="lede">The rules were locked first. For each blind season the picker learned only from later seasons (2023 from 2024&ndash;2026, 2022 from 2023&ndash;2026, 2021 from 2022&ndash;2026), then was applied unchanged to every week at FanDuel, in the four markets we bet now. These are the cleanest tests we have until live weeks build up.</p>
  <div class="scroll"><table><thead><tr><th>Season</th><th>Strategy</th><th>Singles</th><th class="num">Singles return</th><th>Parlays</th><th class="num">Parlay return</th><th class="num">All bets</th><th class="num">Weeks up</th></tr></thead><tbody>${body}</tbody></table></div>
  <p class="lede">Across the blind seasons the straight bets came out close to break-even; the profit came from the boosted parlays (without the boost they were also close to break-even). That is why the locked strategy keeps only the four boosted parlays. Open a blind season under Every week below for each pick.</p>`;
}

function hidden() {
  const H = DATA.hidden; if (!H) return;
  const el = $("#hidden"); el.hidden = false;
  const rows = [["parlays top 8", "Mark 2: boosted 2-leg parlays from the top 8"], ["candidate 3-leg", "Mark 2 variant: 3-leg parlays from the top 9"], ["candidate schedule", "Mark 3 (testing): schedule picker, 4 parlays"], ["parlays top 2", "Mark 1 A's parlay (#1 + #2)"], ["singles top 2", "Mark 1 A's straight bets (top 2)"], ["singles top 8", "Mark 1 B's straight bets (top 8)"]].filter(([k]) => H[k]);
  const ys = Object.keys(H["parlays top 8"]);
  const cell = x => x.bets ? `<td class="num ${cls(x.units)}">${fmtPct(x.units / x.bets)}<br><small style="color:var(--muted)">${x.bets} bets</small></td>` : `<td class="num">–</td>`;
  const body = rows.map(([key, label]) => { const r = H[key]; const u = ys.reduce((a, y) => a + r[y].units, 0), b = ys.reduce((a, y) => a + r[y].bets, 0);
    return `<tr><td>${label}</td>${ys.map(y => cell(r[y])).join("")}${cell({units: u, bets: b})}</tr>`; }).join("");
  el.innerHTML = `<div class="strat-head"><h2>Hidden-season test, 2021&ndash;2026</h2><span class="badge">Tournament 3</span></div>
  <p class="lede">Each season was hidden in turn: the picker was retrained on the other five seasons only, then the rules were applied to the hidden season. The locked parlays were positive in all six; the straight bets were not. The two candidates (ideas batch 1) are shown on the same test. Return per unit staked.</p>
  <div class="scroll"><table><thead><tr><th>Rule</th>${ys.map(y => `<th class="num">${y}</th>`).join("")}<th class="num">All six</th></tr></thead><tbody>${body}</tbody></table></div>
  <p class="lede">Also tested: re-tuning the best rule from scratch for each hidden season (126,000 setups) returned +8% on the hidden seasons, and retraining the projection model 14 ways (tournament 4) did not beat the current model. The locked parlays were positive on all 14 projection models.</p>`;
}
live(); hidden(); blind();
$("#strats").innerHTML = KEYS.map(k => `<section class="panel strat${DATA.roles[k] === "bet" ? " bet" : ""}" id="strat${k}"></section>`).join("");
KEYS.forEach(k => strat(k, $("#strat" + k)));
const cs = KEYS.map(chart); $("#charts").innerHTML = cs.map(c => c.html).join(""); cs.forEach(c => c.wire());
ledger();
</script>
"""

if __name__ == "__main__":
    main()

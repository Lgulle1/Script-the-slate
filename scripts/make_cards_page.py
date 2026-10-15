"""Build the Slate Cards web page (records/cards/slate_cards.html) from the card backtest files written by scripts/card_fixes.py.

  python scripts/make_cards_page.py
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

D = config.ROOT / "records" / "cards"
MARKET = {"rec": "Receptions", "rec_yds": "Receiving yds", "rush_yds": "Rush yds", "rush_att": "Rush attempts", "pass_yds": "Pass yds", "pass_att": "Pass attempts"}


def main():
    s = pl.read_csv(D / "fixes_summary.csv")
    r = pl.read_csv(D / "replay_summary.csv")
    c = pl.read_csv(D / "replay_cards.csv").with_columns(market_label=pl.col("market").replace_strict(MARKET, default=pl.col("market")))
    p = pl.read_csv(D / "replay_parlays.csv")
    data = dict(summary=s.to_dicts(), replay=r.to_dicts(),
                cards=c.filter(pl.col("tier") != "Rest").select("variant", "season", "week", "rank", "tier", "player_name", "team", "market_label", "line", "price", "mean",
                                                                 "picker_under_chance", "actual", "won", "profit", "game_id").to_dicts(),
                rest=c.filter(pl.col("tier") == "Rest").group_by("variant", "season", "week").agg(n=pl.len(), w=pl.col("won").sum(), u=pl.col("profit").sum()).to_dicts(),
                parlays=p.to_dicts())
    live = sorted(D.glob("*_wk*_card.csv"))
    if live:
        f = live[-1]
        data["live"] = dict(card=pl.read_csv(f).to_dicts(), parlays=pl.read_csv(f.with_name(f.stem + "_parlays.csv")).to_dicts(),
                            meta=json.loads(f.with_name(f.stem + "_meta.json").read_text()))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=str))
    (D / "slate_cards.html").write_text(html)
    print("wrote", D / "slate_cards.html", len(html), "bytes")


TEMPLATE = r"""<title>Slate Cards</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a single column of scoreboard panels; summary first, then the weekly card you flip through, then its parlays. */
:root{
  --bg:#f3f5f7; --panel:#ffffff; --ink:#18212b; --muted:#5b6774; --rule:#d9dfe5;
  --turf:#1d6b46; --turf-soft:#e3f0e8; --win:#1d7a43; --win-bg:#e2f3e8; --loss:#b03a2e; --loss-bg:#f8e4e1; --pick:#eef3f8;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#10161c; --panel:#18212a; --ink:#e6ebf0; --muted:#97a3af; --rule:#2b3640;
  --turf:#5fc290; --turf-soft:#1b3328; --win:#6fd39c; --win-bg:#173125; --loss:#f08a7d; --loss-bg:#3a1f1c; --pick:#1d2833; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#10161c; --panel:#18212a; --ink:#e6ebf0; --muted:#97a3af; --rule:#2b3640;
  --turf:#5fc290; --turf-soft:#1b3328; --win:#6fd39c; --win-bg:#173125; --loss:#f08a7d; --loss-bg:#3a1f1c; --pick:#1d2833; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:24px 48px}
.wrap{max-width:1040px;margin:0 auto;display:grid;gap:20px}
header h1{font:700 2.6rem/1 var(--display);letter-spacing:.01em;margin:0;text-transform:uppercase}
header p{margin:.4rem 0 0;color:var(--muted);max-width:68ch}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:18px;min-width:0}
h2{font:700 1.35rem/1.1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin:0 0 .6rem}
.lede{color:var(--muted);margin:0 0 .9rem;max-width:70ch}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font:600 .72rem/1.2 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left;padding:.5rem .55rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.5rem .55rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:.88rem}
tr.rec td{background:var(--turf-soft)}
.tag{display:inline-block;font:600 .7rem/1 var(--body);letter-spacing:.05em;text-transform:uppercase;padding:.3rem .5rem;border-radius:999px;background:var(--turf);color:var(--panel)}
.chip{display:inline-block;min-width:3.2em;text-align:center;font:600 .74rem/1 var(--body);padding:.32rem .5rem;border-radius:6px}
.chip.w{background:var(--win-bg);color:var(--win)} .chip.l{background:var(--loss-bg);color:var(--loss)}
.pos{color:var(--win)} .neg{color:var(--loss)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:14px}
.controls label{font:600 .74rem/1 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
select{font:inherit;padding:.45rem .6rem;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--ink)}
.weeks{display:flex;gap:6px;flex-wrap:wrap}
.weeks button{font:600 1rem/1 var(--display);letter-spacing:.04em;padding:.5rem .85rem;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--ink);cursor:pointer}
.weeks button[aria-pressed="true"]{background:var(--ink);color:var(--panel);border-color:var(--ink)}
button:focus-visible,select:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.tierhead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin:16px 0 6px}
.tierhead h3{font:700 1.1rem/1 var(--display);text-transform:uppercase;letter-spacing:.04em;margin:0}
.tierhead .rec{font-family:var(--mono);font-size:.88rem;color:var(--muted)}
.under{font-family:var(--mono);font-size:.88rem}
.muted{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:6px}
.stat{border:1px solid var(--rule);border-radius:8px;padding:10px 12px}
.stat b{display:block;font:700 1.5rem/1.1 var(--display)}
.stat span{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
ul.notes{margin:0;padding-left:1.1rem;color:var(--muted);max-width:75ch;display:grid;gap:.35rem}
@media (max-width:560px){header h1{font-size:2rem} td,th{padding:.45rem .4rem}}
</style>

<div class="wrap">
<header>
  <h1>Slate Cards</h1>
  <p>2024 week 4 through 2026 week 5, replayed week by week. Before each week the picker was retrained on every line before it, ranked every under it liked across all markets, and was graded against what happened. 1 unit = one flat bet. Parlays are $10 two-leg tickets with a 30% profit boost.</p>
</header>

<section class="panel" id="livepanel" hidden>
  <h2 id="livetitle">This week</h2>
  <p class="lede" id="livemeta"></p>
  <div class="tierhead"><h3>Top 10 &mdash; bet these</h3></div>
  <div class="scroll"><table id="livetop"></table></div>
  <div class="tierhead"><h3>Boosted 2-leg parlays from the Top 10</h3></div>
  <div class="scroll"><table id="liveparlays"></table></div>
  <div class="tierhead"><h3>Next 10 &mdash; watch only</h3></div>
  <div class="scroll"><table id="livenext"></table></div>
</section>

<section class="panel">
  <h2>Season records</h2>
  <p class="lede">The card we would bet (all three fixes) next to the original card. 2024 starts at week 4 because the line history starts in 2024.</p>
  <div class="scroll"><table id="replay"></table></div>
</section>

<section class="panel">
  <h2>How we chose the fixes (2026 weeks 1&ndash;5)</h2>
  <p class="lede">The same five weeks, run five ways. A caps picks at one per player and two per game. B skips players with fewer than 3 games for their current team. C averages six picker versions instead of trusting one.</p>
  <div class="scroll"><table id="summary"></table></div>
</section>

<section class="panel">
  <h2>The weekly card</h2>
  <div class="controls">
    <label for="variant">Version</label>
    <select id="variant"></select>
    <label for="season">Season</label>
    <select id="season"></select>
    <div class="weeks" id="weeks" role="group" aria-label="Week"></div>
  </div>
  <div class="stats" id="weekstats"></div>
  <div id="tiers"></div>
</section>

<section class="panel">
  <h2>Boosted parlays this week</h2>
  <p class="lede">Built from the Top 10 only: each leg paired with the next unused leg from a different game.</p>
  <div class="scroll"><table id="parlays"></table></div>
</section>

<section class="panel">
  <h2>Read this before betting</h2>
  <ul class="notes">
    <li>Each week only learned from earlier weeks, but the picker setup and the three fixes were chosen after looking at these same seasons. Treat the records as the best case. Week 6 on is the clean test.</li>
    <li>Parlay payouts assume FanDuel pays the two prices multiplied, which it does for legs from different games. Real boosts are capped in stake and count.</li>
    <li>The 2026 files only carry receptions, receiving yards, rush yards and pass yards.</li>
    <li>Heavy-favourite unders (&minus;180 and worse) win often but pay little; units show the money, the record shows the hit rate.</li>
  </ul>
</section>
</div>

<script>
const DATA = __DATA__;
const REC = "A + B + C";
const $ = s => document.querySelector(s);
const fmtU = v => (v > 0 ? "+" : "") + v.toFixed(2);
const fmtP = v => v > 0 ? "+" + Math.round(v) : String(Math.round(v));
const cls = v => v > 0 ? "pos" : (v < 0 ? "neg" : "");
const variants = ["A + B + C", "as is"];
const WEEKS = {2024: [...Array(15)].map((_, i) => i + 4), 2025: [...Array(18)].map((_, i) => i + 1), 2026: [1, 2, 3, 4, 5]};
let state = {variant: REC, season: 2026, week: 5};
try { const s = JSON.parse(localStorage.getItem("slatecards2") || "{}"); if (variants.includes(s.variant)) state.variant = s.variant; if (WEEKS[s.season]) { state.season = s.season; if (WEEKS[s.season].includes(s.week)) state.week = s.week; } } catch (e) {}
const save = () => { try { localStorage.setItem("slatecards2", JSON.stringify(state)); } catch (e) {} };

function replay() {
  const h = `<thead><tr><th>Card</th><th>Season</th><th class="num">Weeks</th><th>Top 10</th><th class="num">Hit</th><th class="num">Top 10 units</th><th class="num">Weeks up</th><th>Next 10</th><th class="num">Parlays hit</th><th class="num">Parlay profit (boost)</th><th class="num">Without boost</th></tr></thead>`;
  const money = v => `<span class="${cls(v)}">${v >= 0 ? "+" : "−"}$${Math.abs(v).toFixed(0)}</span>`;
  const b = DATA.replay.map(r => { const n = r["Top 10 W"] + r["Top 10 L"]; const all = r.season === "all";
    return `<tr class="${r.variant === REC && all ? "rec" : ""}"><td>${r.variant === REC ? "With fixes" : "Original"}</td><td>${all ? "<b>All</b>" : r.season}</td><td class="num">${r.weeks}</td><td>${r["Top 10 W"]}–${r["Top 10 L"]}</td><td class="num">${(100 * r["Top 10 W"] / n).toFixed(1)}%</td><td class="num ${cls(r["Top 10 units"])}">${fmtU(r["Top 10 units"])}</td><td class="num">${r.top10_weeks_up} of ${r.top10_weeks}</td><td>${r["Next 10 W"]}–${r["Next 10 L"]}, ${fmtU(r["Next 10 units"])}u</td><td class="num">${r.parlays_hit} of ${r.parlays}</td><td class="num">${money(r.parlay_profit)}</td><td class="num">${money(r.parlay_profit_no_boost)}</td></tr>`; }).join("");
  $("#replay").innerHTML = h + "<tbody>" + b + "</tbody>";
}

function summary() {
  const h = `<thead><tr><th>Version</th><th>Picks per week</th><th>Top 10</th><th>Next 10</th><th>Rest</th><th class="num">Top-10 weeks up</th><th>Parlays hit</th><th class="num">Parlay profit</th><th class="num">Most picks from one game</th></tr></thead>`;
  const b = DATA.summary.map(r => `<tr class="${r.variant === REC ? "rec" : ""}"><td>${r.variant}${r.variant === REC ? ' <span class="tag">Best</span>' : ""}</td><td class="num">${r.picks_per_week}</td><td>${r["Top 10"]}</td><td>${r["Next 10"]}</td><td>${r.Rest}</td><td class="num">${r.top10_weeks_up} of 5</td><td class="num">${r.parlays}</td><td class="num ${cls(r["parlay_$"])}">${r["parlay_$"] >= 0 ? "+" : "−"}$${Math.abs(r["parlay_$"]).toFixed(0)}</td><td class="num">${r.max_same_game_top10}</td></tr>`).join("");
  $("#summary").innerHTML = h + "<tbody>" + b + "</tbody>";
}

function rows(list) {
  if (!list.length) return `<p class="muted">No picks in this tier.</p>`;
  const h = `<thead><tr><th class="num">#</th><th>Player</th><th>Team</th><th>Market</th><th>Bet</th><th class="num">Price</th><th class="num">Our prediction</th><th class="num">Picker chance</th><th class="num">Actual</th><th>Result</th><th class="num">Units</th></tr></thead>`;
  const b = list.map(r => `<tr><td class="num">${r.rank ?? "–"}</td><td>${r.player_name}</td><td>${r.team ?? ""}</td><td>${r.market_label}</td><td class="under">Under ${r.line}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(+r.mean).toFixed(1)}</td><td class="num">${(100 * r.picker_under_chance).toFixed(0)}%</td><td class="num">${r.actual}</td><td><span class="chip ${r.won == 1 ? "w" : "l"}">${r.won == 1 ? "WIN" : "LOSS"}</span></td><td class="num ${cls(r.profit)}">${fmtU(+r.profit)}</td></tr>`).join("");
  return `<div class="scroll"><table>${h}<tbody>${b}</tbody></table></div>`;
}
const record = l => { const w = l.filter(r => r.won == 1).length; const u = l.reduce((a, r) => a + (+r.profit), 0); return {w, l: l.length - w, u}; };

function card() {
  const c = DATA.cards.filter(r => r.variant === state.variant && +r.season === state.season && +r.week === state.week);
  const rest = DATA.rest.find(r => r.variant === state.variant && +r.season === state.season && +r.week === state.week);
  const tiers = ["Top 10", "Next 10", "Skipped (cap)"];
  let html = "";
  const top = record(c.filter(r => r.tier === "Top 10"));
  const tw = record(c.filter(r => r.tier !== "Skipped (cap)"));
  const all = {w: tw.w + (rest ? +rest.w : 0), l: tw.l + (rest ? +rest.n - +rest.w : 0), u: tw.u + (rest ? +rest.u : 0)};
  $("#weekstats").innerHTML = [["Top 10 record", `${top.w}–${top.l}`], ["Top 10 units", `<span class="${cls(top.u)}">${fmtU(top.u)}</span>`], ["All picks", `${all.w}–${all.l}`], ["All units", `<span class="${cls(all.u)}">${fmtU(all.u)}</span>`]]
    .map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  for (const t of tiers) {
    const l = c.filter(r => r.tier === t).sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999));
    if (!l.length && t !== "Top 10") continue;
    const r = record(l);
    html += `<div class="tierhead"><h3>${t}</h3><span class="rec">${r.w}–${r.l}, ${fmtU(r.u)} units</span></div>` + rows(l);
  }
  if (rest) html += `<div class="tierhead"><h3>Rest</h3><span class="rec">${rest.w}–${rest.n - rest.w}, ${fmtU(+rest.u)} units · not bet</span></div>`;
  $("#tiers").innerHTML = html;
  const p = DATA.parlays.filter(r => r.variant === state.variant && +r.season === state.season && +r.week === state.week);
  $("#parlays").innerHTML = p.length ? `<thead><tr><th>Leg 1</th><th>Leg 2</th><th class="num">Price</th><th class="num">Picker chance</th><th>Result</th><th class="num">Profit ($10, boosted)</th></tr></thead><tbody>` +
    p.map(r => `<tr><td>${r.leg1}</td><td>${r.leg2}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(100 * r.chance).toFixed(0)}%</td><td><span class="chip ${r.hit === "true" || r.hit === true ? "w" : "l"}">${r.hit === "true" || r.hit === true ? "HIT" : "MISS"}</span></td><td class="num ${cls(r.profit)}">${r.profit >= 0 ? "+" : "−"}$${Math.abs(+r.profit).toFixed(2)}</td></tr>`).join("") + "</tbody>"
    : `<tbody><tr><td class="muted">Not enough Top-10 picks from different games for a parlay this week.</td></tr></tbody>`;
}

function controls() {
  $("#variant").innerHTML = variants.map(v => `<option value="${v}" ${v === state.variant ? "selected" : ""}>${v === REC ? "With fixes (A + B + C)" : "Original"}</option>`).join("");
  $("#variant").onchange = e => { state.variant = e.target.value; save(); card(); };
  $("#season").innerHTML = [2024, 2025, 2026].map(y => `<option ${y === state.season ? "selected" : ""}>${y}</option>`).join("");
  $("#season").onchange = e => { state.season = +e.target.value; if (!WEEKS[state.season].includes(state.week)) state.week = WEEKS[state.season][0]; save(); weekButtons(); card(); };
  weekButtons();
}
function weekButtons() {
  $("#weeks").innerHTML = WEEKS[state.season].map(w => `<button type="button" data-w="${w}" aria-pressed="${w === state.week}">${w}</button>`).join("");
  $("#weeks").onclick = e => { const b = e.target.closest("button"); if (!b) return; state.week = +b.dataset.w; save(); document.querySelectorAll("#weeks button").forEach(x => x.setAttribute("aria-pressed", String(+x.dataset.w === state.week))); card(); };
}
function live() {
  const L = DATA.live; if (!L) return;
  $("#livepanel").hidden = false;
  $("#livetitle").textContent = `Week ${L.meta.week} card`;
  $("#livemeta").textContent = `Made ${L.meta.made_at_utc.replace("T", " ").replace("+00:00", " UTC")}, before kickoff. Unders only. Book is where the line comes from: FanDuel, or Caesars for rush attempts, pass attempts and completions (FanDuel does not post them). Best price is the best under at the same line across regular sportsbooks.`;
  const day = g => new Date(g + "T12:00:00Z").toLocaleDateString("en-US", {weekday: "short", month: "short", day: "numeric", timeZone: "UTC"});
  const t = l => `<thead><tr><th class="num">#</th><th>Player</th><th>Team</th><th>Bet</th><th>Book</th><th class="num">Price</th><th>Best price</th><th class="num">Our prediction</th><th class="num">Picker chance</th><th>Game day</th></tr></thead><tbody>` +
    l.map(r => `<tr><td class="num">${r.rank}</td><td>${r.player_name}</td><td>${r.team ?? ""}</td><td class="under">${r.market} under ${r.line}</td><td>${r.book}</td><td class="num">${fmtP(r.price)}</td><td>${r.best_book ? `${fmtP(r.best_price)} ${r.best_book}` : "–"}</td><td class="num">${(+r.our_prediction).toFixed(1)}</td><td class="num">${(100 * r.picker_under_chance).toFixed(0)}%</td><td>${day(r.gameday)}</td></tr>`).join("") + "</tbody>";
  $("#livetop").innerHTML = t(L.card.filter(r => r.tier === "Top 10"));
  $("#livenext").innerHTML = t(L.card.filter(r => r.tier === "Next 10"));
  $("#liveparlays").innerHTML = `<thead><tr><th>Leg 1</th><th>Leg 2</th><th class="num">Price</th><th class="num">Picker chance both hit</th><th class="num">Win on $10 (boosted)</th></tr></thead><tbody>` +
    L.parlays.map(r => `<tr><td>${r.leg1}</td><td>${r.leg2}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(100 * r.picker_chance).toFixed(0)}%</td><td class="num">$${(+r.payout_on_10_boosted).toFixed(2)}</td></tr>`).join("") + "</tbody>";
}
live(); replay(); summary(); controls(); card();
</script>
"""

if __name__ == "__main__":
    main()

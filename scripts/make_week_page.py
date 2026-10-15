"""The week's picks page (records/cards/<season>_wkNN_picks.html) from the locked card (scripts/week_card.py output):
the locked strategy P (scripts/strategy.py): up to 4 boosted 2-leg parlays from the top 8 unders, no straight bets; then the 8 legs with the model's numbers,
then everything else the picker liked, by time slot, with the reason it was not taken.

  python scripts/make_week_page.py --season 2026 --week 6
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

D = config.ROOT / "records" / "cards"
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("strategy", Path(__file__).resolve().parent / "strategy.py")
ST = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(ST)


def slot(weekday: str, gametime: str) -> tuple:
    h, m = (int(x) for x in gametime.split(":"))
    t = f"{(h - 1) % 12 + 1}:{m:02d} {'PM' if h >= 12 else 'AM'} ET"
    if weekday == "Thursday":
        return 0, "Thursday Night", t
    if weekday == "Monday":
        return 5, "Monday Night", t
    if weekday == "Sunday":
        if h < 12:
            return 1, "Sunday Morning (London)", t
        if h < 15:
            return 2, "Sunday 1 PM", t
        if h < 19:
            return 3, "Sunday 4 PM", t
        return 4, "Sunday Night", t
    return 6, weekday, t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=6)
    a = ap.parse_args(argv)
    stem = f"{a.season}_wk{a.week:02d}_card"
    card = pl.read_csv(D / f"{stem}.csv")
    meta = json.loads((D / f"{stem}_meta.json").read_text())
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute(f"SELECT DISTINCT ON (game_id) game_id, weekday, gametime, away_team, home_team FROM schedules WHERE season={a.season} AND week={a.week} ORDER BY game_id, pulled_at DESC").pl()
    con.close()
    sc = sc.with_columns(s=pl.struct("weekday", "gametime").map_elements(lambda r: [str(v) for v in slot(r["weekday"], r["gametime"])], return_dtype=pl.List(pl.String)))
    sc = sc.with_columns(slot_order=pl.col("s").list.get(0).cast(pl.Int32), slot=pl.col("s").list.get(1), kickoff=pl.col("s").list.get(2),
                         matchup=pl.col("away_team") + " @ " + pl.col("home_team")).drop("s")
    card = card.join(sc, on="game_id", how="left")
    # the locked strategy (scripts/strategy.py): P is bet (parlays only)
    card = card.with_columns(dec_u=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price")))
    rows = card.to_dicts()
    for r in rows:
        r["needs"] = (-r["price"] / (-r["price"] + 100)) if r["price"] < 0 else 100 / (r["price"] + 100)
    P = ST.card(rows, "P", player="player_name")
    legs = [x for pr in P["parlays"] for x in pr]
    leg_of = {(r["player_name"], r["market"]): (n, j) for n, pr in enumerate(P["parlays"], 1) for j, r in enumerate(pr)}
    in_top8 = {(r["player_name"], r["market"]) for r in ST.top(rows, ST.STRATEGIES["P"]["parlay_from"], ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], player="player_name")}
    ranked = sorted(rows, key=lambda r: -r["picker_under_chance"])
    taken_game, taken_player = {}, {}
    for i, r in enumerate(ranked, 1):
        r["overall_rank"] = i
        key = (r["player_name"], r["market"])
        if key in leg_of:
            r["status"], r["why_not"] = f"Parlay {leg_of[key][0]}", ""
        elif key in in_top8:
            r["status"], r["why_not"] = "Unpaired", "In the top 8 but no partner from a different game"
        elif taken_player.get(r["player_name"], 0) >= ST.PARLAY_CAPS["per_player"]:
            r["status"], r["why_not"] = "Cut", "Same player already in a parlay"
        elif taken_game.get(r["game_id"], 0) >= ST.PARLAY_CAPS["per_game"]:
            r["status"], r["why_not"] = "Cut", "Three picks from this game already ranked above it"
        else:
            r["status"], r["why_not"] = "Watch", f"Rank {i}: below the top 8"
        if key in in_top8:
            taken_game[r["game_id"]] = taken_game.get(r["game_id"], 0) + 1
            taken_player[r["player_name"]] = taken_player.get(r["player_name"], 0) + 1
    def par(pairs):
        out = []
        for a_, b_ in pairs:
            dec = a_["dec_u"] * b_["dec_u"]
            out.append(dict(legs=[a_, b_], price=round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1)), picker_chance=a_["picker_under_chance"] * b_["picker_under_chance"],
                            payout_on_10_boosted=round(10 * (dec - 1) * (1 + ST.BOOST), 2), boosted_price=round((dec - 1) * (1 + ST.BOOST) * 100)))
        return out
    for i, r in enumerate(legs, 1):
        r["rank"] = r["overall_rank"]
    # the bet list as a file: the locked parlays, in order (replaces week_card.py's older parlay list)
    pl.DataFrame([dict(parlay=n, leg1=f"#{x['rank']} {x['player_name']} {x['market']} under {x['line']:g} ({x['price']:+.0f})", leg2=f"#{y['rank']} {y['player_name']} {y['market']} under {y['line']:g} ({y['price']:+.0f})",
                       price=p_["price"], boosted_price=p_["boosted_price"], picker_chance=round(p_["picker_chance"], 3), payout_on_10_boosted=p_["payout_on_10_boosted"])
                  for n, (p_, (x, y)) in enumerate(zip(par(P["parlays"]), P["parlays"]), 1)]).write_csv(D / f"{stem}_parlays.csv")
    data = dict(meta=meta, parlays=par(P["parlays"]), legs=legs, liked=ranked,
                slots=sc.sort("slot_order", "kickoff").select("slot_order", "slot", "kickoff", "matchup", "game_id").to_dicts())
    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=str)).replace("__WEEK__", str(a.week))
    out = D / f"{a.season}_wk{a.week:02d}_picks.html"
    out.write_text(html)
    print("wrote", out)


TEMPLATE = r"""<meta charset="utf-8">
<title>Week __WEEK__ Picks</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a bet slip. Big header, the four boosted parlays as tickets (the only bets), the eight legs with the model's numbers, then the rest of the slate by kickoff slot. */
:root{
  --bg:#eef1f4; --panel:#ffffff; --ink:#141c24; --muted:#5a6672; --rule:#d6dce2;
  --turf:#17633f; --turf-ink:#ffffff; --turf-soft:#e2efe7; --warn:#9a5b00; --warn-soft:#fbefdc; --chip:#eef3f8;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f151b; --panel:#17202a; --ink:#e7ecf1; --muted:#98a4b0; --rule:#2a3540; --turf:#58c08b; --turf-ink:#0c1a12; --turf-soft:#183023; --warn:#f0b35c; --warn-soft:#33261a; --chip:#1d2833; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f151b; --panel:#17202a; --ink:#e7ecf1; --muted:#98a4b0; --rule:#2a3540; --turf:#58c08b; --turf-ink:#0c1a12; --turf-soft:#183023; --warn:#f0b35c; --warn-soft:#33261a; --chip:#1d2833; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
.wrap{max-width:1040px;margin:0 auto;display:grid;gap:22px}
.hero{background:var(--turf);color:var(--turf-ink);border-radius:14px;padding:22px 22px 20px;display:grid;gap:10px}
.hero .eyebrow{font:600 .78rem/1 var(--body);letter-spacing:.12em;text-transform:uppercase;opacity:.85}
.hero h1{font:800 clamp(2.6rem,8vw,4.4rem)/.95 var(--display);margin:0;text-transform:uppercase;letter-spacing:.01em}
.hero .sum{display:flex;flex-wrap:wrap;gap:8px}
.hero .sum span{font:600 .9rem/1 var(--body);background:color-mix(in srgb,var(--turf-ink) 16%,transparent);padding:.5rem .7rem;border-radius:999px}
.hero p{margin:0;max-width:70ch;opacity:.92}
h2{font:700 1.5rem/1.1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin:0}
.lede{color:var(--muted);margin:.3rem 0 0;max-width:72ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.ticket{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:16px;display:grid;grid-template-columns:auto 1fr auto;gap:4px 14px;align-items:start;min-width:0}
.slip{background:var(--panel);border:2px solid var(--turf);border-radius:12px;padding:16px;display:grid;gap:10px;min-width:0}
.slip-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.slip-head .n{font:800 1.6rem/1 var(--display);text-transform:uppercase;color:var(--turf);letter-spacing:.02em}
.slip-head .odds{font:600 1.15rem/1 var(--mono)}
.legrow{display:grid;grid-template-columns:auto 1fr auto;gap:2px 10px;align-items:baseline;padding-top:8px;border-top:1px dashed var(--rule)}
.legrow .r{font:800 1.3rem/1 var(--display);color:var(--turf);min-width:1.6em}
.legrow .who{font-size:1.1rem}
.legrow .bet{grid-column:2/4}
.legrow .meta{grid-column:2/4}
.slip-foot{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.88rem;color:var(--muted);padding-top:8px;border-top:1px solid var(--rule)}
.slip-foot b{color:var(--ink);font-family:var(--mono);font-weight:600}
.rank{grid-row:1/4;font:800 2.4rem/1 var(--display);color:var(--turf);min-width:1.4em}
.who{font:700 1.25rem/1.15 var(--display);text-transform:uppercase;letter-spacing:.02em;min-width:0}
.team{font:500 .8rem/1 var(--body);color:var(--muted);margin-left:.35rem;letter-spacing:.04em}
.price{font:600 1.15rem/1 var(--mono);text-align:right}
.bet{grid-column:2/4;font:600 1rem/1.3 var(--body)}
.bet b{font-family:var(--display);font-size:1.2rem;letter-spacing:.03em;color:var(--turf)}
.meta{grid-column:2/4;font-size:.84rem;color:var(--muted)}
.why{grid-column:1/4;margin-top:8px;padding-top:10px;border-top:1px dashed var(--rule);font-size:.88rem;display:grid;gap:4px}
.why .num{font-family:var(--mono)}
.flag{grid-column:1/4;margin-top:6px;background:var(--warn-soft);color:var(--warn);border-radius:8px;padding:.45rem .6rem;font-size:.84rem}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:18px;min-width:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font:600 .72rem/1.2 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left;padding:.5rem .55rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.55rem .55rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:.9rem}
.boost{display:inline-block;font:700 .72rem/1 var(--body);letter-spacing:.06em;text-transform:uppercase;padding:.3rem .5rem;border-radius:6px;background:var(--turf);color:var(--turf-ink)}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}
.how div{border:1px solid var(--rule);border-radius:10px;padding:12px}
.how b{display:block;font:700 1.05rem/1.1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin-bottom:4px}
.slot{display:grid;gap:8px}
.slot + .slot{margin-top:18px}
.slothead{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:baseline}
.slothead h3{font:700 1.2rem/1 var(--display);text-transform:uppercase;letter-spacing:.04em;margin:0}
.slothead .games{color:var(--muted);font-size:.86rem}
.tag{display:inline-block;font:600 .7rem/1 var(--body);letter-spacing:.04em;text-transform:uppercase;padding:.28rem .45rem;border-radius:6px;background:var(--chip);color:var(--muted)}
.tag.bet{background:var(--turf);color:var(--turf-ink)}
.none{color:var(--muted);font-size:.9rem}
small.note{color:var(--muted)}
button:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
</style>

<div class="wrap">
  <section class="hero">
    <div class="eyebrow">Locked strategy · FanDuel · made before kickoff</div>
    <h1>Week __WEEK__ Picks</h1>
    <div class="sum" id="sum"></div>
    <p>These are the bets: up to four 2-leg parlays, each with a 30% FanDuel profit boost. No straight bets. Place them in order: if you only have one boost, bet Parlay 1; two boosts, Parlays 1 and 2; and so on. No boost, no bet.</p>
  </section>

  <section>
    <h2>The parlays</h2>
    <p class="lede">The picker's top 8 unders, paired in rank order (#1 with #2, #3 with #4, ...), never two legs from the same game. Same stake on each.</p>
    <div class="grid" id="slips" style="margin-top:12px"></div>
  </section>

  <section class="panel">
    <h2>The eight legs</h2>
    <p class="lede">Each leg with our stat model's number, the line, and the picker's chance it goes under against what the FanDuel price needs.</p>
    <div class="scroll" style="margin-top:10px"><table id="legs"></table></div>
  </section>

  <section class="panel">
    <h2>Why this and not straight bets</h2>
    <div class="how">
      <div><b>Parlays only</b>On every season we hid from the picker (2021&ndash;2026), these four parlays a week returned +50% per unit and were positive in all six. Straight bets on the same picks came out about break-even.</div>
      <div><b>The boost is the edge</b>Without the 30% boost the same parlays return about +1&ndash;3%. With it, the picks' small edge over FanDuel's prices turns into profit. No boost, no bet.</div>
      <div><b>Four, not one</b>One parlay a week (#1 + #2) returned +28% on hidden seasons and lost money on the fully blind 2021&ndash;2023; four a week returned +50% (+38% blind).</div>
      <div><b>Unders, proven roles</b>Unders only (overs lost money in every test). Players need 3+ games with their current team. Up to 3 legs per game, each player once.</div>
    </div>
  </section>

  <section class="panel">
    <h2>Everything else it liked, by kickoff</h2>
    <p class="lede">Every under the picker rated as good value this week, grouped by time slot, with the reason it is not a bet. Bets are marked so you can see the whole slot.</p>
    <div id="slots" style="margin-top:14px"></div>
  </section>

  <p><small class="note" id="foot"></small></p>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const fmtP = v => v > 0 ? "+" + Math.round(v) : String(Math.round(v));
const pct = v => Math.round(100 * v) + "%";
const unit = {"Receptions": "rec", "Receiving yds": "rec yds", "Rush yds": "rush yds", "Pass yds": "pass yds"};

$("#sum").innerHTML = [`${DATA.parlays.length} boosted parlays`, "30% boost on each", "No straight bets", "Flat stake", "Unders only"].map(s => `<span>${s}</span>`).join("");

const legHtml = r => `<div class="legrow">
    <div class="r">#${r.rank}</div>
    <div class="who">${r.player_name}<span class="team">${r.team ?? ""}</span></div>
    <div class="price">${fmtP(r.price)}</div>
    <div class="bet"><b>UNDER ${r.line}</b> ${r.market.toLowerCase()}</div>
    <div class="meta">${r.matchup} · ${r.slot}, ${r.kickoff}</div>
  </div>`;
$("#slips").innerHTML = DATA.parlays.map((p, i) => `<article class="slip">
    <div class="slip-head"><span class="n">Parlay ${i + 1}</span><span class="odds">${fmtP(p.price)} <small class="note">→ ${fmtP(p.boosted_price)} boosted</small></span></div>
    ${p.legs.map(legHtml).join("")}
    <div class="slip-foot"><span>Both hit (picker): <b>${pct(p.picker_chance)}</b></span><span>$10 wins <b>$${(+p.payout_on_10_boosted).toFixed(2)}</b> with boost</span></div>
  </article>`).join("");

$("#legs").innerHTML = `<thead><tr><th class="num">#</th><th>Parlay</th><th>Player</th><th>Bet</th><th class="num">FanDuel</th><th class="num">Our model</th><th class="num">Picker chance</th><th class="num">Price needs</th><th>Kickoff</th></tr></thead><tbody>` +
  DATA.legs.map(r => `<tr><td class="num">${r.rank}</td><td><span class="boost">${r.status}</span></td><td>${r.player_name} <small class="note">${r.team ?? ""}</small></td><td>Under ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(+r.our_prediction).toFixed(1)}${r.our_prediction >= r.line ? " ⚠" : ""}</td><td class="num">${pct(r.picker_chance ?? r.picker_under_chance)}</td><td class="num">${pct(r.needs)}</td><td>${r.slot}</td></tr>`).join("") +
  `</tbody>` + (DATA.legs.some(r => r.our_prediction >= r.line) ? `<caption style="caption-side:bottom;text-align:left;padding-top:8px;color:var(--muted);font-size:.85rem">⚠ Our stat model has this player above the line; the picker still takes the under on price and game setup.</caption>` : "");

const all = DATA.liked;
const order = [...new Map(DATA.slots.map(s => [s.slot, s])).values()];
$("#slots").innerHTML = order.map(s => {
  const games = DATA.slots.filter(g => g.slot === s.slot);
  const rows = all.filter(r => r.slot === s.slot).sort((a, b) => b.picker_under_chance - a.picker_under_chance);
  const head = `<div class="slothead"><h3>${s.slot}</h3><span class="games">${games.map(g => `${g.matchup} (${g.kickoff})`).join(" · ")}</span></div>`;
  if (!rows.length) return `<div class="slot">${head}<p class="none">Nothing in this slot rated as value.</p></div>`;
  return `<div class="slot">${head}<div class="scroll"><table><thead><tr><th>Status</th><th>Player</th><th>Bet</th><th class="num">FanDuel</th><th class="num">Our model</th><th class="num">Picker chance</th><th>Why not taken</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.status.startsWith("Parlay") ? `<span class="tag bet">${r.status}</span>` : `<span class="tag">${r.status}</span>`}</td><td>${r.player_name} <small class="note">${r.team ?? ""}</small></td><td>Under ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(+r.our_prediction).toFixed(1)}</td><td class="num">${pct(r.picker_under_chance)}</td><td>${r.why_not || "On the card"}</td></tr>`).join("") +
    `</tbody></table></div></div>`;
}).join("");

$("#foot").textContent = `Card made ${DATA.meta.made_at_utc.replace("T", " ").replace("+00:00", " UTC")} from FanDuel prices at that time; lines may have moved since. The picker was retrained on every graded line before this week. Record of these parlays: hidden seasons 2021-2026 +50% per unit (positive all six); fully blind 2021-2023 +38%; replay 2024 wk 4 to 2026 wk 5 +69% (best case).`;
</script>
"""

if __name__ == "__main__":
    main()

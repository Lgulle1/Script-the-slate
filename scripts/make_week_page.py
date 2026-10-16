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
    top8 = ST.top(rows, ST.STRATEGIES["P"]["parlay_from"], ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], player="player_name")
    in_top8 = {(r["player_name"], r["market"]) for r in top8}
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
    def par(groups):
        out = []
        for grp in groups:
            dec = 1.0
            ch = 1.0
            for l in grp:
                dec *= l["dec_u"]
                ch *= l["picker_under_chance"]
            out.append(dict(legs=list(grp), price=round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1)), picker_chance=ch,
                            payout_on_10_boosted=round(10 * (dec - 1) * (1 + ST.BOOST), 2), boosted_price=round((dec - 1) * (1 + ST.BOOST) * 100)))
        return out
    for r in top8:
        r["rank"] = r["overall_rank"]
    # the bet list as a file: the locked parlays, in order (replaces week_card.py's older parlay list)
    pl.DataFrame([dict(parlay=n, leg1=f"#{x['rank']} {x['player_name']} {x['market']} under {x['line']:g} ({x['price']:+.0f})", leg2=f"#{y['rank']} {y['player_name']} {y['market']} under {y['line']:g} ({y['price']:+.0f})",
                       price=p_["price"], boosted_price=p_["boosted_price"], picker_chance=round(p_["picker_chance"], 3), payout_on_10_boosted=p_["payout_on_10_boosted"])
                  for n, (p_, (x, y)) in enumerate(zip(par(P["parlays"]), P["parlays"]), 1)]).write_csv(D / f"{stem}_parlays.csv")
    # Mark 1 (the retired setup, shown on the switch): A = top 2 straight + #1/#2 parlay (was bet), B = top 8 straight + 4 parlays (tracked)
    A = ST.card(rows, "A", player="player_name")
    B = ST.card(rows, "B", player="player_name")
    a_keys = {(r["player_name"], r["market"]) for r in A["singles"]}
    b_keys = {(r["player_name"], r["market"]) for r in B["singles"]}
    taken = {}
    for r in ranked:
        key = (r["player_name"], r["market"])
        if key in a_keys:
            r["status1"], r["why1"] = "Bet (A)", ""
        elif key in b_keys:
            r["status1"], r["why1"] = "Tracked (B)", "Ranks 3-8: in B, not bet"
        elif taken.get(r["game_id"], 0) >= ST.SINGLES_CAPS["per_game"]:
            r["status1"], r["why1"] = "Cut", "Three picks from this game already ranked above it"
        else:
            r["status1"], r["why1"] = "Watch", f"Rank {r['overall_rank']}: below the top 8"
        taken[r["game_id"]] = taken.get(r["game_id"], 0) + 1
    mark1 = dict(a_singles=[dict(r, rank=r["overall_rank"]) for r in A["singles"]], a_parlays=par(A["parlays"]),
                 b_singles=[dict(r, rank=r["overall_rank"]) for r in B["singles"]], b_parlays=par(B["parlays"]))
    # Mark 2 variant, tracked: C3 = 3-leg parlays from the top 9
    C3 = ST.card(rows, "C3", player="player_name")
    cand = dict(c3=par(C3["parlays"]))
    # Mark 3 (testing, tracked): the schedule picker's own picks (scripts/schedule_picker.py): top 8 straight + 4 parlays
    mark3 = dict(singles=[], parlays=[], liked=[], built=False)
    fs = D / f"{a.season}_wk{a.week:02d}_card_schedule.csv"
    if fs.exists():
        sr = pl.read_csv(fs).join(sc, on="game_id", how="left").with_columns(
            dec_u=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price"))).to_dicts()
        for r in sr:
            r["needs"] = (-r["price"] / (-r["price"] + 100)) if r["price"] < 0 else 100 / (r["price"] + 100)
        S = ST.card(sr, "S", player="player_name")
        s_leg = {(r["player_name"], r["market"]): n for n, grp in enumerate(S["parlays"], 1) for r in grp}
        s_single = {(r["player_name"], r["market"]): n for n, r in enumerate(S["singles"], 1)}
        sr = sorted(sr, key=lambda r: -r["picker_under_chance"])
        for i, r in enumerate(sr, 1):
            key = (r["player_name"], r["market"])
            tags = ([f"Straight {s_single[key]}"] if key in s_single else []) + ([f"Parlay {s_leg[key]}"] if key in s_leg else [])
            r["status3"], r["why3"] = (" + ".join(tags), "") if tags else ("Cut", "Same player already in a parlay, or below its top 8")
        mark3 = dict(singles=S["singles"], parlays=par(S["parlays"]), liked=sr, built=True)
    data = dict(meta=meta, parlays=par(P["parlays"]), legs=top8, liked=ranked, mark1=mark1, mark3=mark3, cand=cand,
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
.switch{display:inline-flex;gap:4px;background:color-mix(in srgb,var(--turf-ink) 14%,transparent);padding:4px;border-radius:999px;justify-self:start}
.switch button{font:700 .95rem/1 var(--display);letter-spacing:.05em;text-transform:uppercase;border:0;border-radius:999px;padding:.55rem 1rem;background:transparent;color:var(--turf-ink);cursor:pointer}
.switch button[aria-pressed="true"]{background:var(--turf-ink);color:var(--turf)}
.switch button:focus-visible{outline:2px solid var(--turf-ink);outline-offset:2px}
.markwrap{display:grid;gap:22px}
.markwrap[hidden]{display:none}
.retired{background:var(--warn-soft);color:var(--warn);border-radius:10px;padding:.7rem .9rem;font-size:.92rem}
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
    <div class="eyebrow" id="eyebrow"></div>
    <h1>Week __WEEK__ Picks</h1>
    <div class="switch" role="group" aria-label="Setup">
      <button type="button" data-m="2" aria-pressed="true">Mark 2 · current</button>
      <button type="button" data-m="1" aria-pressed="false">Mark 1 · old A + B</button>
      <button type="button" data-m="3" aria-pressed="false">Mark 3 · testing</button>
    </div>
    <div class="sum" id="sum"></div>
    <p id="herop"></p>
  </section>

  <div class="markwrap" data-mark="1" hidden>
  <p class="retired">Mark 1 is the old setup, retired as the bet on 2026-10-15 and still tracked every week. Same picker and same picks as Mark 2; only the betting rule differs. On seasons the picker never saw, its straight bets lost money (A's top 2: &minus;11%; B's top 8: &minus;2%) and A's single parlay returned +28%, against +50% for Mark 2's parlays. Not bet.</p>
  <section>
    <h2>A: the straight bets</h2>
    <p class="lede">The top 2 unders as straight bets. Same stake on each.</p>
    <div class="grid" id="m1picks" style="margin-top:12px"></div>
  </section>
  <section class="panel">
    <h2>A: the parlay</h2>
    <p class="lede">#1 and #2 together, with the 30% profit boost.</p>
    <div class="scroll" style="margin-top:10px"><table id="m1parlay"></table></div>
  </section>
  <section class="panel">
    <h2>B: top 8 straight + 4 parlays</h2>
    <p class="lede">The broader old version. B's four parlays are the same four parlays Mark 2 bets; B adds the eight straight bets.</p>
    <div class="scroll" style="margin-top:10px"><table id="m1b"></table></div>
    <div class="scroll" style="margin-top:10px"><table id="m1bparlays"></table></div>
  </section>
  </div>

  <div class="markwrap" data-mark="3" hidden>
  <p class="retired">Mark 3 is being tested: tracked every week, not bet. It is Mark 2's picker with schedule inputs added (rest days for both teams, short week, division game, primetime, grass or turf), and it makes its own picks. On seasons it never saw its parlays returned +54% against Mark 2's +50% (better in 4 of 6); on the week-by-week replay +77% against +40%; and its straight bets came out ahead (+3.7% hidden, +5.7% replay) where Mark 2's did not. It becomes the bet only if live weeks back that up.</p>
  <section>
    <h2>Mark 3 parlays</h2>
    <p class="lede">Its own top 8 unders, paired in rank order, never two legs from one game, 30% boost. Not bet.</p>
    <div class="grid" id="m3slips" style="margin-top:12px"></div>
  </section>
  <section class="panel">
    <h2>Mark 3 straight bets</h2>
    <p class="lede">Its top 8 unders as straight bets (up to 3 per game). Tracked because its straight bets were the first to come out positive on hidden seasons.</p>
    <div class="scroll" style="margin-top:10px"><table id="m3s"></table></div>
  </section>
  </div>

  <div class="markwrap" data-mark="2">

  <section>
    <h2>The parlays</h2>
    <p class="lede">The picker's top unders (up to 8 a week), paired in rank order (#1 with #2, #3 with #4, ...), never two legs from the same game. Same stake on each.</p>
    <div class="grid" id="slips" style="margin-top:12px"></div>
  </section>

  <section class="panel">
    <h2 id="legsTitle">The legs</h2>
    <p class="lede">Every pick that qualified this week (4+ points of edge, 3+ games with the team), with our stat model's number, the line, and the picker's chance it goes under against what the FanDuel price needs. A pick with no partner from a different game is listed as unpaired.</p>
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
    <h2>Variant tracked: 3-leg parlays</h2>
    <p class="lede">The same picks as 3-leg parlays from the top 9. Hidden seasons: +65% against +50%, better in 5 of 6, but one season lost 73%; replay +34% against +40%. Tracked, not bet.</p>
    <div class="scroll" style="margin-top:8px"><table id="c3"></table></div>
  </section>
  </div>

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

const HERO = {
  2: {eyebrow: "Mark 2 · locked strategy · FanDuel · made before kickoff",
      sum: [`${DATA.parlays.length} boosted parlays`, "30% boost on each", "No straight bets", "Flat stake", "Unders only"],
      p: "These are the bets: up to four 2-leg parlays, each with a 30% FanDuel profit boost. No straight bets. Place them in order: if you only have one boost, bet Parlay 1; two boosts, Parlays 1 and 2; and so on. No boost, no bet."},
  1: {eyebrow: "Mark 1 · old A + B · retired 2026-10-15 · for comparison",
      sum: [`A: ${DATA.mark1.a_singles.length} straight + ${DATA.mark1.a_parlays.length} parlay`, `B: ${DATA.mark1.b_singles.length} straight + ${DATA.mark1.b_parlays.length} parlays`, "Unders only"],
      p: "The old setup: A bet the two strongest unders straight plus the same two as one boosted parlay; B tracked the top 8 straight and four parlays. Not what we bet now."},
  3: {eyebrow: "Mark 3 · testing · schedule picker · tracked, not bet",
      sum: [`${DATA.mark3.parlays.length} parlays`, `${DATA.mark3.singles.length} straight`, "Not bet", "Unders only"],
      p: "Mark 2's picker plus schedule inputs (rest, short week, division, primetime, turf), making its own picks. Tracked every week to see whether it should replace Mark 2. Do not place these."}
};
let mark = 2;
try { const m = localStorage.getItem("wk-mark"); if (m === "1" || m === "2" || m === "3") mark = +m; } catch (e) {}
function setMark(m) {
  mark = m;
  try { localStorage.setItem("wk-mark", String(m)); } catch (e) {}
  document.querySelectorAll(".switch button").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.m === m)));
  document.querySelectorAll("[data-mark]").forEach(el => { el.hidden = +el.dataset.mark !== m; });
  $("#eyebrow").textContent = HERO[m].eyebrow;
  $("#sum").innerHTML = HERO[m].sum.map(s => `<span>${s}</span>`).join("");
  $("#herop").textContent = HERO[m].p;
  renderSlots();
}
document.querySelector(".switch").addEventListener("click", e => { const b = e.target.closest("button"); if (b) setMark(+b.dataset.m); });

const M1 = DATA.mark1;
$("#m1picks").innerHTML = M1.a_singles.map((r, i) => `<article class="ticket">
    <div class="rank">${i + 1}</div>
    <div class="who">${r.player_name}<span class="team">${r.team ?? ""}</span></div>
    <div class="price">${fmtP(r.price)}</div>
    <div class="bet"><b>UNDER ${r.line}</b> ${r.market.toLowerCase()}</div>
    <div class="meta">${r.matchup} · ${r.slot}, ${r.kickoff}</div>
    <div class="why">
      <div>Our model: <span class="num">${(+r.our_prediction).toFixed(1)}</span> ${unit[r.market] ?? ""} vs line <span class="num">${r.line}</span></div>
      <div>Picker: <span class="num">${pct(r.picker_under_chance)}</span> to go under; the price needs <span class="num">${pct(r.needs)}</span></div>
    </div>
  </article>`).join("");
const ptab = (list, label) => !list.length ? `<tbody><tr><td class="note">None this week.</td></tr></tbody>` : `<thead><tr><th>${label}</th>${list[0].legs.map((_, j) => `<th>Leg ${j + 1}</th>`).join("")}<th class="num">Price</th><th class="num">Chance all hit</th><th class="num">Win on $10 with boost</th></tr></thead><tbody>` +
  list.map((p, i) => `<tr><td><span class="boost">Parlay ${i + 1}</span></td>${p.legs.map(l => `<td>${l.player_name} under ${l.line} ${l.market.toLowerCase()} (${fmtP(l.price)})</td>`).join("")}<td class="num">${fmtP(p.price)}</td><td class="num">${pct(p.picker_chance)}</td><td class="num">$${(+p.payout_on_10_boosted).toFixed(2)}</td></tr>`).join("") + "</tbody>";
const strTab = list => `<thead><tr><th class="num">#</th><th>Player</th><th>Bet</th><th class="num">FanDuel</th><th class="num">Our model</th><th class="num">Picker chance</th><th>Kickoff</th></tr></thead><tbody>` +
  list.map((r, i) => `<tr><td class="num">${i + 1}</td><td>${r.player_name} <small class="note">${r.team ?? ""}</small></td><td>Under ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(+r.our_prediction).toFixed(1)}</td><td class="num">${pct(r.picker_under_chance)}</td><td>${r.slot ?? ""}</td></tr>`).join("") + "</tbody>";
$("#c3").innerHTML = ptab(DATA.cand.c3, "3-leg");
$("#m3s").innerHTML = DATA.mark3.singles.length ? strTab(DATA.mark3.singles) : `<tbody><tr><td class="note">${DATA.mark3.built ? "No qualifying picks this week." : "Mark 3 card not built for this week."}</td></tr></tbody>`;
$("#m1parlay").innerHTML = ptab(M1.a_parlays, "A parlay");
$("#m1b").innerHTML = strTab(M1.b_singles);
$("#m1bparlays").innerHTML = ptab(M1.b_parlays, "B parlay");

const legHtml = r => `<div class="legrow">
    <div class="r">#${r.rank}</div>
    <div class="who">${r.player_name}<span class="team">${r.team ?? ""}</span></div>
    <div class="price">${fmtP(r.price)}</div>
    <div class="bet"><b>UNDER ${r.line}</b> ${r.market.toLowerCase()}</div>
    <div class="meta">${r.matchup} · ${r.slot}, ${r.kickoff}</div>
  </div>`;
const slips = list => list.length ? list.map((p, i) => `<article class="slip">
    <div class="slip-head"><span class="n">Parlay ${i + 1}</span><span class="odds">${fmtP(p.price)} <small class="note">→ ${fmtP(p.boosted_price)} boosted</small></span></div>
    ${p.legs.map(legHtml).join("")}
    <div class="slip-foot"><span>Both hit (picker): <b>${pct(p.picker_chance)}</b></span><span>$10 wins <b>$${(+p.payout_on_10_boosted).toFixed(2)}</b> with boost</span></div>
  </article>`).join("") : `<p class="none">No parlays this week.</p>`;
$("#slips").innerHTML = slips(DATA.parlays);
$("#m3slips").innerHTML = slips(DATA.mark3.parlays.map(p => ({...p, legs: p.legs.map((l, j) => ({...l, rank: DATA.mark3.liked.findIndex(x => x.player_name === l.player_name && x.market === l.market) + 1}))})));
$("#legsTitle").textContent = `The ${DATA.legs.length} legs`;

$("#legs").innerHTML = `<thead><tr><th class="num">#</th><th>Parlay</th><th>Player</th><th>Bet</th><th class="num">FanDuel</th><th class="num">Our model</th><th class="num">Picker chance</th><th class="num">Price needs</th><th>Kickoff</th></tr></thead><tbody>` +
  DATA.legs.map(r => `<tr><td class="num">${r.rank}</td><td><span class="boost">${r.status}</span></td><td>${r.player_name} <small class="note">${r.team ?? ""}</small></td><td>Under ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(+r.our_prediction).toFixed(1)}${r.our_prediction >= r.line ? " ⚠" : ""}</td><td class="num">${pct(r.picker_chance ?? r.picker_under_chance)}</td><td class="num">${pct(r.needs)}</td><td>${r.slot}</td></tr>`).join("") +
  `</tbody>` + (DATA.legs.some(r => r.our_prediction >= r.line) ? `<caption style="caption-side:bottom;text-align:left;padding-top:8px;color:var(--muted);font-size:.85rem">⚠ Our stat model has this player above the line; the picker still takes the under on price and game setup.</caption>` : "");

const order = [...new Map(DATA.slots.map(s => [s.slot, s])).values()];
function renderSlots() {
const st = r => mark === 2 ? [r.status, r.why_not] : mark === 1 ? [r.status1, r.why1] : [r.status3, r.why3];
const isBet = t => mark === 2 ? t.startsWith("Parlay") : mark === 1 ? t === "Bet (A)" : /Parlay|Straight/.test(t);
const all = mark === 3 ? DATA.mark3.liked : DATA.liked;
$("#slots").innerHTML = order.map(s => {
  const games = DATA.slots.filter(g => g.slot === s.slot);
  const rows = all.filter(r => r.slot === s.slot).sort((a, b) => b.picker_under_chance - a.picker_under_chance);
  const head = `<div class="slothead"><h3>${s.slot}</h3><span class="games">${games.map(g => `${g.matchup} (${g.kickoff})`).join(" · ")}</span></div>`;
  if (!rows.length) return `<div class="slot">${head}<p class="none">Nothing in this slot rated as value.</p></div>`;
  return `<div class="slot">${head}<div class="scroll"><table><thead><tr><th>Status</th><th>Player</th><th>Bet</th><th class="num">FanDuel</th><th class="num">Our model</th><th class="num">Picker chance</th><th>Why not taken</th></tr></thead><tbody>` +
    rows.map(r => { const [t, why] = st(r); return `<tr><td>${isBet(t) ? `<span class="tag bet">${t}</span>` : `<span class="tag">${t}</span>`}</td><td>${r.player_name} <small class="note">${r.team ?? ""}</small></td><td>Under ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${(+r.our_prediction).toFixed(1)}</td><td class="num">${pct(r.picker_under_chance)}</td><td>${why || "On the card"}</td></tr>`; }).join("") +
    `</tbody></table></div></div>`;
}).join("");
}
setMark(mark);

$("#foot").textContent = `Card made ${DATA.meta.made_at_utc.replace("T", " ").replace("+00:00", " UTC")} from FanDuel prices at that time; lines may have moved since. The picker was retrained on every graded line before this week. Record of these parlays: hidden seasons 2021-2026 +50% per unit (positive all six); fully blind 2021-2023 +38%; replay 2024 wk 4 to 2026 wk 5 +51% (picker trained only on earlier weeks).`;
</script>
"""

if __name__ == "__main__":
    main()

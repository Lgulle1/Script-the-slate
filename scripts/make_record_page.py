"""The Track Record page (records/cards/track_record.html), one tab per Mark after an Overview that answers which Mark is best: the locked strategy P (4 boosted parlays from the top 8, bet) and the old strategies A and B
(tracked) replayed week by week over 2024 week 4 - 2026 week 5, every pick and parlay graded, the blind seasons, the hidden-season test of tournament 3,
plus the current week's picks (pending). Ledger also written to records/cards/track_record.json.

  python scripts/make_record_page.py [--live-week 7]     # live record: every week from 2026 week 6 to --live-week
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
LIVE_FIRST_WEEK = 6                                                   # 2026 week 6 is live week 1: the first week whose card was made before the games
FOUR = ["rec", "rec_yds", "rush_yds", "pass_yds"]                     # the markets bet (records/cards/locked_setup.json), every season
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


def grade(df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Fill actual and won (under) on a card's rows from the played games; a game not played yet stays pending (won empty). Players are matched by name inside the game."""
    import sys
    sys.path.insert(0, str(config.ROOT))
    from live import results as RS
    act = RS.played(season)[0].filter(pl.col("player_id").is_not_null()).select("game_id", "player_id", "market", "actual")
    log = (pl.read_csv(config.ROOT / "records" / "lines" / "prop_lines_log.csv", infer_schema_length=0).select("game_id", "player_id", "player_name")
           .unique(["game_id", "player_name"]))
    x = df.drop("actual", "won").join(log, on=["game_id", "player_name"], how="left").join(act, on=["game_id", "player_id", "market"], how="left")
    played = act.select("game_id").unique()["game_id"].to_list()
    miss = x.filter(pl.col("game_id").is_in(played) & pl.col("actual").is_null())
    if miss.height:
        print("WARNING: played game but no result for", miss.select("player_name", "market", "game_id").rows(), "(left pending: void or unmatched)")
    return x.with_columns(won=pl.when(pl.col("actual").is_null()).then(None).otherwise((pl.col("actual") < pl.col("line")).cast(pl.Float64)))


def mark4_weeks() -> dict:
    """Mark 4 (Mark 4i) replay: {(season, week): ledger} of the exact live card (mark4/live.py PLAN) built each week from the forward chances
    (tournament.load_p("Mark 4i"): picker and mix weights never saw the week they bet). Legs carry their side; parlays their stake and boosted profit."""
    sys.path.insert(0, str(config.ROOT))
    from mark4 import forward as F, tournament as T, live as LV
    d, A, *_ = F.frame()
    p = T.load_p(LV.VERSION)
    if len(p) != d.height:
        print("WARNING: Mark 4 chances do not match the line table (rerun the forward build); Mark 4 replay left out")
        return {}
    p = p.copy()
    p[d["season"].to_numpy() < 2022] = np.nan
    W = T.week_pools(d, A, p, "both", "all", 0.0, "ev")
    look = {}
    for i, r in enumerate(d.select("game_id", "player_id", "market", "player_name", "team", "line", "actual").iter_rows()):
        look.setdefault((r[0], r[1], r[2]), (i, r[3], r[4], r[5], r[6]))
    out = {}
    for (season, week), rows in W.items():
        legs = T.pool(rows, 40)
        used, pars = set(), []
        for k, c, stake in LV.PLAN:
            free = [r for j, r in enumerate(legs) if j not in used]
            for par in T.build(free, k, c, "rank", "any"):
                for r in par:
                    used.add(next(j for j, y in enumerate(legs) if y is r))
                ls = []
                for r in par:
                    i, name, team, line, actual = look[(r[4], r[5], r[6])]
                    over = (p[i] - A.be_o[i]) > ((1 - p[i]) - A.be_u[i])
                    ls.append(dict(player=name, team=team, market=LABEL.get(r[6], r[6]), line=line, side="Over" if over else "Under", price=american(r[2]),
                                   actual=actual, won=float(r[3]), profit=None, chance=round(float(r[0]), 3)))
                win = all(r[3] == 1 for r in par)
                dec = float(np.prod([r[2] for r in par]))
                pars.append(dict(legs=ls, price=american(dec), stake=stake, profit=stake * ((dec - 1) * (1 + T.BOOST) if win else -1.0)))
        out[(int(season), int(week))] = dict(singles=[], parlays=pars)
    return out


def mark4_live(season: int, week: int):
    """The live Mark 4 card (mark4/live.py), graded leg by leg on its own side; a parlay is settled once a leg loses or every leg is in."""
    f = D / f"{season}_wk{week:02d}_card_mark4.csv"
    if not f.exists():
        return None
    sys.path.insert(0, str(config.ROOT))
    from mark4 import live as LV, tournament as T
    MK = {v: k for k, v in LV.MARKET_LABEL.items()}
    c = pl.read_csv(f).with_columns(player_name=pl.col("player"), game_id=pl.col("game"), market=pl.col("market").replace(MK), actual=pl.lit(None, dtype=pl.Float64),
                                    won=pl.lit(None, dtype=pl.Float64))
    c = grade(c, season).with_columns(won=pl.when(pl.col("actual").is_null()).then(None)
                                      .when(pl.col("side") == "Over").then((pl.col("actual") > pl.col("line")).cast(pl.Float64))
                                      .otherwise((pl.col("actual") < pl.col("line")).cast(pl.Float64)))
    pars = []
    for (pid,), g in c.group_by("parlay", maintain_order=True):
        r0 = g.row(0, named=True)
        ws = g["won"].to_list()
        stake, dec = float(r0["stake_units"]), float(np.prod([1 + p / 100 if p > 0 else 1 + 100 / -p for p in g["price"].to_list()]))
        profit = -stake if 0.0 in ws else (stake * (dec - 1) * (1 + T.BOOST) if all(w == 1.0 for w in ws) else None)
        pars.append(dict(legs=[dict(player=r["player"], team=r["team"], market=LABEL.get(r["market"], r["market"]), line=r["line"], side=r["side"], price=int(r["price"]),
                                    actual=r["actual"], won=r["won"], profit=None, chance=r["chance"]) for r in g.iter_rows(named=True)],
                         price=int(r0["parlay_price"]), stake=stake, profit=profit))
    meta = f.with_name(f.stem + "_meta.json")
    return dict(singles=[], parlays=pars, made_at=json.loads(meta.read_text())["made_at_utc"] if meta.exists() else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-week", type=int, default=6)
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args(argv)
    p = pl.read_parquet(POOL).filter(pl.col("market").is_in(FOUR))
    sp = pl.read_parquet(SPOOL).filter(pl.col("market").is_in(FOUR)) if SPOOL.exists() else None
    sweeks = {k: g.to_dicts() for k, g in sp.group_by("season", "week")} if sp is not None else {}
    weeks = []
    for (s, w), g in sorted(p.group_by("season", "week"), key=lambda t: t[0]):
        weeks.append(dict(season=int(s), week=int(w), **ledger(g.to_dicts(), srows=sweeks.get((s, w), []) if sp is not None else None)))
    m4 = mark4_weeks()
    for w in weeks:
        if m4:
            w["M4"] = m4.get((w["season"], w["week"]), dict(singles=[], parlays=[]))
    live = []                                                          # every live week so far, live week 1 (2026 week 6) to --live-week
    for wk in range(LIVE_FIRST_WEEK, a.live_week + 1):
        f = D / f"{a.season}_wk{wk:02d}_card.csv"
        if not f.exists():
            continue
        c = grade(pl.read_csv(f).with_columns(dec_u=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price")),
                                              market=pl.col("market").replace({v: k for k, v in LABEL.items()}), won=pl.lit(None, dtype=pl.Float64), actual=pl.lit(None, dtype=pl.Float64)), a.season)
        fs = D / f"{a.season}_wk{wk:02d}_card_schedule.csv"
        srows = None
        if fs.exists():
            srows = grade(pl.read_csv(fs).with_columns(dec_u=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price")),
                                                       market=pl.col("market").replace({v: k for k, v in LABEL.items()}), won=pl.lit(None, dtype=pl.Float64), actual=pl.lit(None, dtype=pl.Float64)), a.season).to_dicts()
        L = dict(season=a.season, week=wk, live_no=wk - LIVE_FIRST_WEEK + 1, made_at=json.loads(f.with_name(f.stem + "_meta.json").read_text())["made_at_utc"],
                 **ledger(c.to_dicts(), player="player_name", srows=srows))
        m4l = mark4_live(a.season, wk)
        if m4l is not None:
            L["M4"] = m4l
        live.append(L)
    live = live or None
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
    mc = config.ROOT / "records" / "tuning" / "mark_compare.json"      # Mark 2 vs Mark 3 head to head (scripts/compare_marks.py)
    se = config.ROOT / "records" / "tuning" / "seed_ensembles.json"    # seed-averaged hidden seasons (scripts/seed_ensembles.py)
    data = dict(weeks=weeks, live=live, compare=json.loads(mc.read_text()) if mc.exists() else None, seeded=json.loads(se.read_text()) if se.exists() else None, strategies={**{k: v["name"] for k, v in ST.STRATEGIES.items()}, "M4": "Mark 4: 5/3/2/2 boosted parlays"}, roles={**{k: v["role"] for k, v in ST.STRATEGIES.items()}, "M4": "bet"},
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
/* Layout: a results book. Tabs across the top (Overview, Mark 2, Mark 1, Mark 3); the Overview answers "which Mark is best" first,
   then each Mark's tab goes from its headline numbers down to every graded week. Same turf-green identity as the weekly picks page. */
:root{
  --bg:#eef1f4; --panel:#ffffff; --ink:#141c24; --muted:#5a6672; --faint:#8a95a0; --rule:#d6dce2; --grid:#e6eaee;
  --turf:#17633f; --turf-ink:#ffffff; --turf-soft:#e2efe7; --win:#1b7a44; --win-bg:#e1f2e7; --loss:#b0392d; --loss-bg:#f8e3df; --chip:#eef3f8;
  --test:#7a4e00; --test-bg:#fbefdc; --series-1:#2a78d6; --series-2:#eb6834;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f151b; --panel:#17202a; --ink:#e7ecf1; --muted:#98a4b0; --faint:#6d7884; --rule:#2a3540; --grid:#222c36;
  --turf:#58c08b; --turf-ink:#0c1a12; --turf-soft:#183023; --win:#6fd39c; --win-bg:#173125; --loss:#f08a7d; --loss-bg:#3a1f1c; --chip:#1d2833;
  --test:#f0b35c; --test-bg:#33261a; --series-1:#3987e5; --series-2:#d95926; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f151b; --panel:#17202a; --ink:#e7ecf1; --muted:#98a4b0; --faint:#6d7884; --rule:#2a3540; --grid:#222c36;
  --turf:#58c08b; --turf-ink:#0c1a12; --turf-soft:#183023; --win:#6fd39c; --win-bg:#173125; --loss:#f08a7d; --loss-bg:#3a1f1c; --chip:#1d2833;
  --test:#f0b35c; --test-bg:#33261a; --series-1:#3987e5; --series-2:#d95926; color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
.wrap{max-width:1080px;margin:0 auto;display:grid;gap:22px}
header{display:grid;gap:12px}
header h1{font:800 clamp(2.4rem,7vw,3.6rem)/.95 var(--display);margin:0;text-transform:uppercase}
header p{margin:0;color:var(--muted);max-width:72ch}
.tabs{display:flex;flex-wrap:wrap;gap:6px;border-bottom:2px solid var(--rule);padding-bottom:0}
.tabs button{font:700 1rem/1 var(--display);letter-spacing:.05em;text-transform:uppercase;padding:.7rem 1rem .6rem;border:0;border-bottom:3px solid transparent;margin-bottom:-2px;background:transparent;color:var(--muted);cursor:pointer;display:flex;gap:8px;align-items:center}
.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--turf)}
.tabs button:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.role{font:600 .66rem/1 var(--body);letter-spacing:.08em;text-transform:uppercase;padding:.3rem .45rem;border-radius:5px;background:var(--chip);color:var(--muted)}
.role.bet{background:var(--turf);color:var(--turf-ink)} .role.test{background:var(--test-bg);color:var(--test)}
h2{font:700 1.45rem/1.1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin:0;text-wrap:balance}
h3{font:700 1.1rem/1.1 var(--display);text-transform:uppercase;letter-spacing:.03em;margin:0}
.lede{color:var(--muted);margin:.3rem 0 0;max-width:74ch}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:18px;min-width:0;display:grid;gap:12px}
.verdict{background:var(--turf);color:var(--turf-ink);border-radius:14px;padding:22px;display:grid;gap:10px}
.verdict .eyebrow{font:600 .78rem/1 var(--body);letter-spacing:.12em;text-transform:uppercase;opacity:.85}
.verdict h2{font:800 clamp(1.8rem,5vw,2.8rem)/1 var(--display);text-transform:uppercase}
.verdict p{margin:0;max-width:72ch;opacity:.95}
.marks{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}
.mark{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:16px;display:grid;gap:10px;align-content:start;min-width:0}
.mark.is-bet{border:2px solid var(--turf)}
.mark-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.mark-head b{font:800 1.6rem/1 var(--display);text-transform:uppercase}
.mark p{margin:0;color:var(--muted);font-size:.9rem}
.kv{display:grid;gap:2px;font-size:.9rem}
.kv span{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em} .kv b{font-family:var(--mono);font-weight:500;overflow-wrap:anywhere;margin-bottom:6px}
.open{justify-self:start;font:700 .9rem/1 var(--display);letter-spacing:.05em;text-transform:uppercase;padding:.55rem .8rem;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--ink);cursor:pointer}
.open:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.tile{border:1px solid var(--rule);border-radius:10px;padding:10px 12px;min-width:0}
.tile span{display:block;font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile b{display:block;font:700 1.7rem/1.1 var(--display);font-variant-numeric:tabular-nums}
.tile small{color:var(--muted);font-size:.8rem}
.pos{color:var(--win)} .neg{color:var(--loss)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font:600 .72rem/1.2 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left;padding:.45rem .5rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.5rem .5rem;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:top}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:.88rem}
td small{color:var(--muted);font-family:var(--body)}
.meter{display:flex;align-items:center;gap:8px;justify-content:flex-end}
.meter i{display:block;width:90px;height:8px;border-radius:4px;background:var(--grid);position:relative;overflow:hidden}
.meter i::after{content:"";position:absolute;inset:0 auto 0 0;width:var(--w);background:var(--turf)}
.chip{display:inline-block;min-width:3.4em;text-align:center;font:600 .72rem/1 var(--body);padding:.3rem .45rem;border-radius:6px}
.chip.w{background:var(--win-bg);color:var(--win)} .chip.l{background:var(--loss-bg);color:var(--loss)} .chip.p{background:var(--chip);color:var(--muted)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.chart{min-width:0;max-width:640px}
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
.filters{display:flex;gap:6px;flex-wrap:wrap}
.filters button{font:600 .9rem/1 var(--display);letter-spacing:.04em;padding:.45rem .8rem;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--ink);cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.filters button:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.note{background:var(--test-bg);color:var(--test);border-radius:10px;padding:.7rem .9rem;font-size:.92rem;margin:0}
ul.notes{margin:0;padding-left:1.1rem;color:var(--muted);display:grid;gap:.35rem;max-width:78ch}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <h1>Track Record</h1>
  <p>Every betting setup we run, graded. Mark 4 (Mark 4i: Mark 3's picker mixed with FanDuel's price, in 5/3/2/2 boosted parlays) is the bet from week 7 of 2026; its first graded week shows here after week 7 is played. Mark 3 (the bet 2026-10-16 to week 6), Mark 2 and Mark 1 are tracked beside it with no money on them. 1 unit = one bet; parlays include the 30% FanDuel boost; FanDuel's four markets (receptions, receiving yards, rush yards, pass yards).</p>
  <nav class="tabs" role="tablist" aria-label="Marks" id="tabs"></nav>
</header>
<main id="view"></main>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const fmtU = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
const fmtPct = v => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(100 * v).toFixed(1) + "%";
const fmtP = v => v > 0 ? "+" + v : String(v);
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";
const ratio = (a, b) => b ? fmtPct(a / b) : "–";
const chip = won => won == null ? `<span class="chip p">PENDING</span>` : won === 1 ? `<span class="chip w">WIN</span>` : `<span class="chip l">LOSS</span>`;
const has = k => DATA.weeks.length && DATA.weeks[0][k];

const MARKS = {
  mark4: {label: "Mark 4", role: "Bet", rc: "bet", keys: ["M4"], hidden: [],
          what: "The bet from 2026 week 7. Mark 4i: Mark 3's picker mixed with FanDuel's own no-vig price (weights chosen from earlier seasons only). Legs ranked by expected value, over or under per leg, never two legs of one game; one 30% boost per parlay: 1 x 5-leg at 0.25u, 1 x 3-leg at 0.5u, 2 x 2-leg at 1u, built long-first. The weeks before week 7 are the replay of this exact card (the picker and the mix weights never saw the week they bet)."},
  mark3: {label: "Mark 3", role: "Tracked", rc: "", keys: ["S", "SS", "C3"], seeded: "Mark 3",
          what: "Boosted 2-leg parlays from the schedule picker's top unders (up to 4 a week). Same projections as Mark 2; the picker also weighs rest, short weeks, division games, primetime and turf. Its straight bets and 3-leg parlays are tracked, not bet.",
          hidden: [["candidate schedule", "Mark 3 parlays, one seed"]]},
  mark2: {label: "Mark 2", role: "Tracked", rc: "", keys: ["P"], seeded: "Mark 2",
          what: "The previous bet (2026-10-15 to 2026-10-16): the same parlay rule on the picker without schedule inputs.",
          hidden: [["parlays top 8", "Mark 2 parlays, one seed"]]},
  mark1: {label: "Mark 1", role: "Tracked", rc: "", keys: ["A", "B"], what: "The old rules on Mark 2's picks. A: top 2 straight + one parlay. B: top 8 straight + 4 parlays.",
          hidden: [["parlays top 2", "A's parlay (#1 + #2)"], ["singles top 2", "A's straight bets (top 2)"], ["singles top 8", "B's straight bets (top 8)"], ["parlays top 8", "B's parlays (same as Mark 2's)"]]}
};
const ORDER = ["overview", "mark4", "mark3", "mark2", "mark1"];
function seededTotal(name) {
  const S = DATA.seeded && DATA.seeded[name]; if (!S) return null;
  return Object.entries(S).filter(([g]) => g.startsWith("seeds")).map(([g, v]) => v.parlays);
}

function stats(k, weeks) {
  const ws = weeks.filter(w => w[k]);
  const s = ws.flatMap(w => w[k].singles).filter(x => x.profit != null), p = ws.flatMap(w => w[k].parlays).filter(x => x.profit != null);
  const su = s.reduce((a, x) => a + x.profit, 0), pu = p.reduce((a, x) => a + x.profit, 0);
  const sw = s.filter(x => x.won === 1).length, pw = p.filter(x => x.profit > 0).length;
  const up = ws.filter(w => w[k].singles.reduce((a, x) => a + x.profit, 0) + w[k].parlays.reduce((a, x) => a + x.profit, 0) > 0).length;
  const stk = xs => xs.reduce((a, x) => a + (x.stake ?? 1), 0);
  return {n: s.length, sw, su, pn: p.length, pw, pu, all: su + pu, staked: stk(s) + stk(p), up, weeks: ws.length};
}
function hiddenTotal(key) {
  const r = DATA.hidden && DATA.hidden[key]; if (!r) return null;
  const v = Object.values(r); const u = v.reduce((a, x) => a + x.units, 0), b = v.reduce((a, x) => a + x.bets, 0);
  return {u, b, pos: v.filter(x => x.units > 0).length, n: v.length};
}
const KLABEL = {S: "parlays", SS: "straight bets", C3: "3-leg", P: "parlays", A: "A", B: "B"};
function liveLine(k) {
  const Ls = (DATA.live || []).filter(L => L[k]); if (!Ls.length) return null;
  const t = stats(k, Ls), open = Ls.reduce((a, L) => a + L[k].singles.length + L[k].parlays.length, 0) - t.n - t.pn;
  const parts = [];
  if (t.n) parts.push(`${t.sw}–${t.n - t.sw} straight`);
  if (t.pn) parts.push(`${t.pw}–${t.pn - t.pw} parlays`);
  if (open) parts.push(`${open} pending`);
  return {text: parts.join(", ") || "none", units: t.all, settled: t.staked, weeks: Ls.map(L => L.week)};
}

// ---------- overview
function overview() {
  const C = DATA.compare;
  const rows = C ? [["Parlays · hidden seasons 2021–2026", C.hidden.parlays], ["Parlays · week-by-week replay 2024–2026", C.replay.parlays],
                    ["Straight bets · hidden seasons", C.hidden.singles], ["Straight bets · replay", C.replay.singles]] : [];
  const s3 = seededTotal("Mark 3") || [], s2 = seededTotal("Mark 2") || [];
  const pr = DATA.seeded && DATA.seeded["Mark 3"] && DATA.seeded["Mark 3"].p_better_than_mark2 ? Object.values(DATA.seeded["Mark 3"].p_better_than_mark2).map(x => Math.round(100 * x.parlays)) : [];
  const verdict = `<section class="verdict">
    <div class="eyebrow">The answer so far</div>
    <h2>Mark 4 is the bet from week 7. Mark 3 before it, still tracked.</h2>
    <p>Mark 4 (Mark 4i) beats Mark 3 on the same weeks: on the real 5/3/2/2 card, 2022 to 2026 week 6, it made +128.8 units on 131.8 bet, up every season, and came out ahead of Mark 3 in every one of 4,000 week-by-week resamples. Mark 4l (the same plus promotion-overreaction unders) made less, +101.3 on 151.8, at every strength of that add-on tried, so it was dropped on 2026-10-22 before week 7's first game. Its replay is in the Mark 4 tab; its live record starts with week 7. What follows is how Mark 3 became the bet before that.</p>
    <p>With the picker averaged over 5 random seeds, Mark 3's parlays returned ${s3.map(v => fmtPct(v)).join(" and ")} on seasons it never saw (two separate seed groups), against ${s2.map(v => fmtPct(v)).join(" and ")} for Mark 2: ${pr.join("% and ")}% likely to be truly better. On the week-by-week replay (2024–2026, fewer weeks) the two came out level${C ? ` (${fmtPct(C.replay.parlays.mark3.roi)} vs ${fmtPct(C.replay.parlays.mark2.roi)})` : ""}. Mark 3 was not worse anywhere and its straight bets did better on both, so it is the bet since 2026-10-16; live weeks keep checking it.</p>
  </section>`;
  const meter = p => `<div class="meter"><span>${Math.round(100 * p)}%</span><i style="--w:${Math.round(100 * p)}%"></i></div>`;
  const h2h = `<section class="panel"><div><h2>Mark 2 vs Mark 3, head to head</h2><p class="lede">Same weeks, same rule, two pickers, each averaged over 5 random seeds. "Chance Mark 3 is better" resamples the weeks thousands of times and counts how often Mark 3 comes out ahead; 50% would mean no difference.</p></div>
    <div class="scroll"><table><thead><tr><th>Test</th><th class="num">Mark 2</th><th class="num">Mark 3</th><th class="num">Chance Mark 3 is better</th></tr></thead><tbody>` +
    rows.map(([t, c]) => `<tr><td>${t}</td><td class="num ${cls(c.mark2.roi)}">${fmtPct(c.mark2.roi)}<br><small>${c.mark2.bets} bets</small></td><td class="num ${cls(c.mark3.roi)}">${fmtPct(c.mark3.roi)}<br><small>${c.mark3.bets} bets</small></td><td class="num">${meter(c.p_mark3_better)}</td></tr>`).join("") +
    `</tbody></table></div></section>`;
  const card = id => {
    const M = MARKS[id];
    const h = M.hidden.length ? hiddenTotal(M.hidden[0][0]) : null, sd = M.seeded ? seededTotal(M.seeded) : null;
    const keys = M.keys.filter(has);
    const rep = keys.map(k => [k, stats(k, DATA.weeks)]);
    const lives = M.keys.map(k => [k, liveLine(k)]).filter(([, v]) => v);
    return `<article class="mark ${M.rc === "bet" ? "is-bet" : ""}">
      <div class="mark-head"><b>${M.label}</b><span class="role ${M.rc}">${M.role}</span></div>
      <p>${M.what}</p>
      <div class="kv">
        ${sd ? `<span>Hidden seasons, 5-seed average</span><b class="${cls(sd[0])}">${sd.map(fmtPct).join(" / ")}</b>` : h ? `<span>Hidden seasons 2021–2026</span><b class="${cls(h.u)}">${ratio(h.u, h.b)}</b>` : ""}
        ${rep.map(([k, t]) => `<span>Replay 2024–2026${{C3: " (3-leg variant)", SS: " (straight bets)", S: " (parlays)", A: " (A)", B: " (B)"}[k] ?? ""}</span><b class="${cls(t.all)}">${ratio(t.all, t.staked)}</b>`).join("")}
        ${lives.map(([k, v]) => `<span>Live (week${v.weeks.length > 1 ? "s" : ""} ${v.weeks.join(", ")})${lives.length > 1 ? ` (${KLABEL[k]})` : ""}</span><b class="${cls(v.units)}">${v.text}${v.settled ? ` · ${fmtU(v.units)}u` : ""}</b>`).join("") || `<span>Live</span><b>–</b>`}
      </div>
      <button type="button" class="open" data-go="${id}">Open ${M.label}</button>
    </article>`;
  };
  const liveRows = ["mark4", "mark3", "mark2", "mark1"].flatMap(id => MARKS[id].keys.map(k => [MARKS[id].label, k, liveLine(k)])).filter(r => r[2]);
  const livePanel = DATA.live ? `<section class="panel" style="margin-top:22px"><div><h2>Live record</h2><p class="lede">Every card made before its games, from NFL week 6 (live week 1, ${DATA.live[0].made_at.slice(0, 10)}). Mark 4 is live from week 7. Return per unit staked; a pending bet counts once it is settled.</p></div>
    <div class="scroll"><table><thead><tr><th>Setup</th><th>Weeks</th><th>Result</th><th class="num">Units</th><th class="num">Return per unit</th></tr></thead><tbody>` +
    liveRows.map(([m, k, v]) => `<tr><td>${DATA.strategies[k]}${k === "M4" ? ` <span class="role bet">Bet</span>` : ""}</td><td>${v.weeks.join(", ")}</td><td>${v.text}</td><td class="num ${cls(v.units)}">${v.settled ? fmtU(v.units) : "–"}</td><td class="num ${cls(v.units)}">${v.settled ? fmtPct(v.units / v.settled) : "–"}</td></tr>`).join("") +
    `</tbody></table></div></section>` : "";
  return verdict + livePanel + `<section class="marks" style="margin-top:22px">${["mark4", "mark3", "mark2", "mark1"].map(card).join("")}</section>` + `<div style="margin-top:22px">${h2h}</div>` +
    `<section class="panel" style="margin-top:22px"><h2>How to read this</h2><ul class="notes">
      <li><b>Hidden seasons:</b> each season 2021–2026 was hidden in turn; the picker learned from the other five only. The fairest test we have.</li>
      <li><b>Seeds:</b> the picker has a random element; one seed moved returns by up to 15 points either way. Mark 2 and Mark 3 are compared averaged over 5 seeds, twice with different seeds, so a result counts only if it repeats.</li>
      <li><b>Replay:</b> 2024 week 4 to 2026 week 5, the picker retrained every week on earlier weeks only, the way it runs live.</li>
      <li><b>Blind 2021–2023:</b> Mark 1 and Mark 2's rules were fixed before those seasons were added, so they never shaped any rule.</li>
      <li><b>Live:</b> from 2026 week 6. Mark 3 replaced Mark 2 on 2026-10-16 after the seed-averaged tests; Mark 4 replaced Mark 3 as the bet from week 7. Every live week stays on the page.</li></ul></section>`;
}

// ---------- one Mark
function strat(k) {
  const t = stats(k, DATA.weeks);
  const seasons = [2024, 2025, 2026].map(y => [y, stats(k, DATA.weeks.filter(w => w.season === y))]);
  return `<section class="panel"><div><h3>${DATA.strategies[k]}</h3><p class="lede">Replay, 2024 week 4 to 2026 week 5.</p></div>
  <div class="tiles">
    <div class="tile"><span>Return per unit</span><b class="${cls(t.all)}">${ratio(t.all, t.staked)}</b><small>${fmtU(t.all)} units on ${t.staked}</small></div>
    ${t.n ? `<div class="tile"><span>Straight bets</span><b>${t.sw}–${t.n - t.sw}</b><small>${(100 * t.sw / t.n).toFixed(1)}% hit · <span class="${cls(t.su)}">${ratio(t.su, t.n)}</span></small></div>` : `<div class="tile"><span>Straight bets</span><b>None</b><small>parlays only</small></div>`}
    <div class="tile"><span>Parlays (boosted)</span><b>${t.pw}–${t.pn - t.pw}</b><small>${t.pn ? (100 * t.pw / t.pn).toFixed(0) : 0}% hit · <span class="${cls(t.pu)}">${ratio(t.pu, t.pn)}</span></small></div>
    <div class="tile"><span>Weeks in profit</span><b>${t.up} of ${t.weeks}</b><small>weeks with a bet</small></div>
  </div>
  <div class="scroll"><table><thead><tr><th>Season</th><th class="num">Weeks</th><th>Straight</th><th class="num">Straight return</th><th>Parlays</th><th class="num">Parlay return</th><th class="num">All bets</th></tr></thead><tbody>` +
  seasons.map(([y, s]) => `<tr><td>${y}</td><td class="num">${s.weeks}</td><td>${s.n ? `${s.sw}–${s.n - s.sw}` : "–"}</td><td class="num ${cls(s.su)}">${ratio(s.su, s.n)}</td><td>${s.pw}–${s.pn - s.pw}</td><td class="num ${cls(s.pu)}">${ratio(s.pu, s.pn)}</td><td class="num ${cls(s.all)}">${fmtU(s.all)}u (${ratio(s.all, s.staked)})</td></tr>`).join("") +
  `</tbody></table></div></section>`;
}
function seededTable(M) {
  const S = M.seeded && DATA.seeded && DATA.seeded[M.seeded]; if (!S) return "";
  const gs = Object.keys(S).filter(g => g.startsWith("seeds")); const ys = Object.keys(S[gs[0]].seasons);
  const c = v => v == null ? `<td class="num">–</td>` : `<td class="num ${cls(v)}">${fmtPct(v)}</td>`;
  return `<section class="panel"><div><h3>Hidden seasons, picker averaged over 5 seeds</h3><p class="lede">Each season hidden in turn; the picker learned from the other five only, averaged over 5 random seeds. Run twice with different seeds. Return per unit on the parlays (and straight bets, tracked).</p></div>
    <div class="scroll"><table><thead><tr><th>Seed group</th>${ys.map(y => `<th class="num">${y}</th>`).join("")}<th class="num">Parlays, all six</th><th class="num">Straight bets</th></tr></thead><tbody>` +
    gs.map(g => `<tr><td>${g}</td>${ys.map(y => c(S[g].seasons[y])).join("")}${c(S[g].parlays)}${c(S[g].singles)}</tr>`).join("") + `</tbody></table></div></section>`;
}
function hiddenTable(M) {
  const H = DATA.hidden; const rows = M.hidden.filter(([k]) => H && H[k]); if (!rows.length) return "";
  const ys = Object.keys(H[rows[0][0]]);
  const cell = x => x && x.bets ? `<td class="num ${cls(x.units)}">${fmtPct(x.units / x.bets)}<br><small>${x.bets} bets</small></td>` : `<td class="num">–</td>`;
  return `<section class="panel"><div><h3>Hidden seasons, 2021–2026</h3><p class="lede">Each season hidden in turn; the picker learned from the other five only. Return per unit staked.</p></div>
    <div class="scroll"><table><thead><tr><th>Rule</th>${ys.map(y => `<th class="num">${y}</th>`).join("")}<th class="num">All six</th></tr></thead><tbody>` +
    rows.map(([k, label]) => { const r = H[k], t = hiddenTotal(k); return `<tr><td>${label}</td>${ys.map(y => cell(r[y])).join("")}${cell({units: t.u, bets: t.b})}</tr>`; }).join("") +
    `</tbody></table></div></section>`;
}
function blindTable(M) {
  if (!DATA.blind) return "";
  const keys = M.keys.filter(k => DATA.blind.every(w => w[k])); if (!keys.length) return "";
  const bs = DATA.blind_seasons;
  const row = (k, label, ws) => { const t = stats(k, ws); return `<tr><td>${label}</td><td>${k === "P" ? "Mark 2" : k}</td><td>${t.n ? `${t.sw}–${t.n - t.sw}` : "–"}</td><td class="num ${cls(t.su)}">${ratio(t.su, t.n)}</td><td>${t.pw}–${t.pn - t.pw}</td><td class="num ${cls(t.pu)}">${ratio(t.pu, t.pn)}</td><td class="num ${cls(t.all)}">${fmtU(t.all)}u (${ratio(t.all, t.staked)})</td></tr>`; };
  const body = [...bs.map(y => keys.map(k => row(k, String(y), DATA.blind.filter(w => w.season === y))).join("")), keys.map(k => row(k, "<b>All blind</b>", DATA.blind)).join("")].join("");
  return `<section class="panel"><div><h3>Blind seasons: ${bs.join(", ")}</h3><p class="lede">The rules were fixed before these seasons were added; the picker learned only from later seasons.</p></div>
    <div class="scroll"><table><thead><tr><th>Season</th><th>Setup</th><th>Straight</th><th class="num">Straight return</th><th>Parlays</th><th class="num">Parlay return</th><th class="num">All bets</th></tr></thead><tbody>${body}</tbody></table></div></section>`;
}
function chart(k) {
  const W = 520, H = 220, m = {l: 44, r: 64, t: 10, b: 26};
  let cs = 0, cp = 0;
  const pts = DATA.weeks.filter(w => w[k]).map((w, i) => { cs += w[k].singles.reduce((a, x) => a + x.profit, 0); cp += w[k].parlays.reduce((a, x) => a + x.profit, 0); return {i, w, s: cs, p: cp}; });
  if (pts.length < 2) return {html: "", wire: () => {}};
  const ys = pts.flatMap(d => [d.s, d.p, 0]);
  let lo = Math.min(...ys), hi = Math.max(...ys); const step = (hi - lo) > 30 ? 10 : 5; lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step; if (hi === lo) hi = lo + step;
  const x = i => m.l + (W - m.l - m.r) * i / (pts.length - 1), y = v => m.t + (H - m.t - m.b) * (hi - v) / (hi - lo);
  const ticks = []; for (let v = lo; v <= hi; v += step) ticks.push(v);
  const line = key => pts.map((d, i) => `${i ? "L" : "M"}${x(d.i).toFixed(1)},${y(d[key]).toFixed(1)}`).join("");
  const starts = pts.filter((d, i) => i === 0 || d.w.season !== pts[i - 1].w.season);
  const last = pts[pts.length - 1], id = "c" + k, single = pts.some(d => d.s !== 0);
  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Running units for ${DATA.strategies[k]}" id="${id}">
    ${ticks.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)" stroke-width="1"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)" font-family="var(--mono)">${v}</text>`).join("")}
    <line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--faint)" stroke-width="1"/>
    ${starts.map(d => `<text x="${x(d.i)}" y="${H - 6}" font-size="11" fill="var(--muted)" font-family="var(--body)">${d.w.season}</text>`).join("")}
    <path d="${line("p")}" fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round"/>
    ${single ? `<path d="${line("s")}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>` : ""}
    <circle cx="${x(last.i)}" cy="${y(last.p)}" r="4" fill="var(--series-2)" stroke="var(--panel)" stroke-width="2"/>
    <text x="${x(last.i) + 8}" y="${y(last.p) + 4}" font-size="11" fill="var(--ink)" font-family="var(--mono)">${fmtU(last.p)}</text>
    ${single ? `<circle cx="${x(last.i)}" cy="${y(last.s)}" r="4" fill="var(--series-1)" stroke="var(--panel)" stroke-width="2"/><text x="${x(last.i) + 8}" y="${y(last.s) + 4}" font-size="11" fill="var(--ink)" font-family="var(--mono)">${fmtU(last.s)}</text>` : ""}
    <line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--faint)" stroke-width="1" visibility="hidden"/>
    <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" class="hit"/>
  </svg>`;
  return {html: `<div class="chart"><h3>${DATA.strategies[k]}</h3><div class="legend">${single ? `<span><i style="background:var(--series-1)"></i>Straight bets</span>` : ""}<span><i style="background:var(--series-2)"></i>Parlays</span></div><div class="chartbox">${svg}<div class="tip" hidden></div></div></div>`,
    wire: () => {
      const el = document.getElementById(id); if (!el) return;
      const tip = el.parentElement.querySelector(".tip"), xh = el.querySelector(".xh"), hit = el.querySelector(".hit");
      hit.addEventListener("pointermove", ev => { const r = el.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width; const i = Math.max(0, Math.min(pts.length - 1, Math.round((px - m.l) / (W - m.l - m.r) * (pts.length - 1)))); const d = pts[i];
        xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("visibility", "visible");
        const ws = d.w[k].singles.reduce((a, q) => a + q.profit, 0), wp = d.w[k].parlays.reduce((a, q) => a + q.profit, 0);
        tip.innerHTML = `<div><b>${d.w.season} week ${d.w.week}</b></div>${single ? `<div>Straight: <b class="${cls(ws)}">${fmtU(ws)}</b> (running ${fmtU(d.s)})</div>` : ""}<div>Parlays: <b class="${cls(wp)}">${fmtU(wp)}</b> (running ${fmtU(d.p)})</div>`;
        tip.hidden = false; tip.style.left = Math.min(r.width - 180, Math.max(0, x(i) * r.width / W + 12)) + "px"; tip.style.top = "8px"; });
      hit.addEventListener("pointerleave", () => { tip.hidden = true; xh.setAttribute("visibility", "hidden"); });
    }};
}
function legRow(r) { return `<tr><td>${r.player} <small>${r.team ?? ""}</small></td><td>${r.side ?? "Under"} ${r.line} ${r.market.toLowerCase()}</td><td class="num">${fmtP(r.price)}</td><td class="num">${r.actual ?? "–"}</td><td>${chip(r.won)}</td><td class="num ${cls(r.profit ?? 0)}">${r.profit == null ? "–" : fmtU(r.profit)}</td></tr>`; }
function block(k, w) {
  const s = w[k].singles, p = w[k].parlays;
  return `<div><h4>${DATA.strategies[k]}</h4>` + (s.length ? `<div class="scroll"><table><thead><tr><th>Straight</th><th>Bet</th><th class="num">Price</th><th class="num">Actual</th><th>Result</th><th class="num">Units</th></tr></thead><tbody>${s.map(legRow).join("")}</tbody></table></div>` : "") +
    (p.length ? `<div class="scroll" style="margin-top:8px"><table><thead><tr><th>Parlay</th><th class="num">Price</th><th>Result</th><th class="num">Units</th></tr></thead><tbody>` +
      p.map(q => `<tr><td>${q.stake != null ? `<b>${q.legs.length}-leg @ ${q.stake}u:</b> ` : ""}${q.legs.map(l => `${l.player} ${l.side ? l.side[0].toLowerCase() : "u"}${l.line} ${l.market.toLowerCase()}`).join(" + ")}</td><td class="num">${fmtP(q.price)}</td><td>${q.profit == null ? chip(null) : chip(q.profit > 0 ? 1 : 0)}</td><td class="num ${cls(q.profit ?? 0)}">${q.profit == null ? "–" : fmtU(q.profit)}</td></tr>`).join("") + `</tbody></table></div>`
      : `<p class="lede" style="font-size:.86rem">No parlay this week.</p>`) + `</div>`;
}
function liveBlock(M) {
  return (DATA.live || []).slice().reverse().map(L => {
    const keys = M.keys.filter(k => L[k]); if (!keys.length) return "";
    const open = keys.some(k => L[k].singles.some(x => x.won == null) || L[k].parlays.some(x => x.profit == null));
    return `<section class="panel"><div><h3>Live week ${L.live_no} (${L.season} week ${L.week}): ${open ? "partly graded" : "final"}</h3><p class="lede">Card made ${((keys.includes("M4") && L.M4.made_at) || L.made_at).replace("T", " ").replace("+00:00", " UTC")}, before kickoff${L.live_no === 1 ? ". The first week of the live record: no rule or setting was chosen after seeing it" : ""}.</p></div>
    <div class="weekbody">${keys.map(k => block(k, L)).join("")}</div></section>`;
  }).join("");
}
const SEASON = {};
function ledger(id) {
  const M = MARKS[id];
  const bkeys = DATA.blind ? M.keys.filter(k => DATA.blind.every(w => w[k])) : [];
  const bs = bkeys.length ? DATA.blind_seasons : [];
  const ys = [...bs, 2024, 2025, 2026];
  const season = SEASON[id] ?? 2026;
  const src = bs.includes(season) ? DATA.blind : DATA.weeks;
  const ws = src.filter(w => w.season === season).slice().reverse();
  const tot = (w, k) => w[k].singles.reduce((a, x) => a + x.profit, 0) + w[k].parlays.reduce((a, x) => a + x.profit, 0);
  const res = (w, k) => { const x = w[k], parts = []; if (x.singles.length) parts.push(`${x.singles.filter(q => q.won === 1).length}/${x.singles.length} straight`); parts.push(`${x.parlays.filter(q => q.profit > 0).length}/${x.parlays.length} parlays`);
    const v = tot(w, k); return `<span class="res">${M.keys.length > 1 ? k + " " : ""}<span class="${cls(v)}">${fmtU(v)}u</span> · ${parts.join(", ")}</span>`; };
  return `<section class="panel"><div><h3>Every week</h3><p class="lede">Open a week to see every pick and parlay.</p></div>
    <div class="filters" role="group" aria-label="Season">${ys.map(y => `<button type="button" aria-pressed="${y === season}" data-season="${y}">${bs.includes(y) ? y + " (blind)" : y}</button>`).join("")}</div>
    <div>${ws.map(w => { const ks = M.keys.filter(k => w[k]); return `<details><summary><span class="wk">Week ${w.week}</span>${ks.map(k => res(w, k)).join("")}</summary><div class="weekbody">${ks.map(k => block(k, w)).join("")}</div></details>`; }).join("") || `<p class="lede">No weeks.</p>`}</div></section>`;
}
function markView(id) {
  const M = MARKS[id];
  const keys = M.keys.filter(has);
  const intro = `<section class="panel"><div class="mark-head"><h2>${M.label}</h2><span class="role ${M.rc}">${M.role}</span></div><p class="lede" style="margin:0">${M.what}</p>
    ${id === "mark2" ? `<p class="note">The bet from 2026-10-15 to 2026-10-16, now tracked. Its single-seed +50% on hidden seasons was a lucky seed; averaged over 5 seeds it is about +28%.</p>` : ""}
    ${id === "mark1" ? `<p class="note">Retired as the bet on 2026-10-15 and still tracked. Its straight bets lost money on hidden seasons, and one parlay a week did worse than Mark 2's.</p>` : ""}</section>`;
  const charts = keys.map(chart);
  return {html: intro + liveBlock(M) + seededTable(M) + hiddenTable(M) + blindTable(M) + keys.map(strat).join("") +
          (charts.some(c => c.html) ? `<section class="panel"><div><h3>Running units, replay</h3><p class="lede">Units won so far, week by week. Hover a week for its numbers.</p></div><div class="charts">${charts.map(c => c.html).join("")}</div></section>` : "") + ledger(id),
          wire: () => charts.forEach(c => c.wire())};
}

// ---------- tabs
let current = "overview";
try { const t = localStorage.getItem("tr-tab"); if (ORDER.includes(t)) current = t; } catch (e) {}
const h = location.hash.replace("#", ""); if (ORDER.includes(h)) current = h;
function render() {
  $("#tabs").innerHTML = ORDER.map(id => `<button type="button" role="tab" id="tab-${id}" aria-selected="${id === current}" data-tab="${id}">${id === "overview" ? "Overview" : MARKS[id].label}${id === "overview" ? "" : ` <span class="role ${MARKS[id].rc}">${MARKS[id].role}</span>`}</button>`).join("");
  const v = $("#view");
  if (current === "overview") { v.innerHTML = overview(); }
  else { const r = markView(current); v.innerHTML = `<div style="display:grid;gap:22px">${r.html}</div>`; r.wire(); }
}
function go(id) { current = id; try { localStorage.setItem("tr-tab", id); } catch (e) {} render(); window.scrollTo({top: 0}); }
document.addEventListener("click", e => {
  const t = e.target.closest("[data-tab]"); if (t) return go(t.dataset.tab);
  const o = e.target.closest("[data-go]"); if (o) return go(o.dataset.go);
  const s = e.target.closest("[data-season]"); if (s) { SEASON[current] = +s.dataset.season; render(); }
});
render();
</script>
"""

if __name__ == "__main__":
    main()

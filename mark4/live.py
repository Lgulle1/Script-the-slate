"""Mark 4l live card (2026-10-21): the week's FanDuel lines -> Mark 4l chances -> the 5/3/2/2 boosted card.

  Mark 3's picker (fixed models, no random hold-out) retrained on every graded line before the week
  -> Mark 4i: mixed with FanDuel's no-vig chance at the season's weights (records/lab/mark4i_weights.json, chosen from earlier seasons only)
  -> Mark 4l: promotion-overreaction unders (role-change engine on this week's rosters and injury report) at the hit rate of earlier seasons
  -> legs: FanDuel's own lines only, edge >= 0 on the better side, ranked by expected value (chance x decimal price - 1), one per player
  -> card: 1 x 5-leg @ 0.25u, 1 x 3-leg @ 0.5u, 2 x 2-leg @ 1u, all boosted (30% profit boost), built long-first, never two legs of one game

Rerun before each boost window with fresh lines (scripts/import_bettingpros.py) and a fresh injury report (run_weekly.py --refresh): legs whose game has
started are kept as placed; the rest are rebuilt.

  python -m mark4.live --season 2026 --week 7     # writes records/cards/<season>_wkNN_card_mark4.csv and prints the card
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import polars as pl

import config
from mark4 import forward as F
from mark4 import rolechange as RC
from mark4 import tournament as T

PLAN = [(5, 1, 0.25), (3, 1, 0.5), (2, 2, 1.0)]
LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
OUTD = config.ROOT / "records" / "cards"
MARKET_LABEL = {"rec": "Receptions", "rec_yds": "Receiving Yards", "rush_yds": "Rushing Yards", "rush_att": "Rushing Attempts", "pass_yds": "Passing Yards",
                "pass_att": "Passing Attempts", "pass_cmp": "Passing Completions"}


def american(dec):
    return np.where(dec >= 2, (dec - 1) * 100, -100 / (dec - 1))


def mark3_live(d, A, X3, season, week):
    tr = (((d["season"] < season) | ((d["season"] == season) & (d["week"] < week))).to_numpy()) & ~np.isnan(A.over_won.astype(float))
    te = ((d["season"] == season) & (d["week"] == week)).to_numpy()
    preds = [mk().fit(X3[tr], A.over_won[tr].astype(int)).predict_proba(X3[te])[:, 1] for mk in F.det_makers()]
    return te, np.mean(preds, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    a = ap.parse_args(argv)
    d, A, X3, *_ = F.frame()
    te, p3w = mark3_live(d, A, X3, a.season, a.week)
    w = json.loads((F.OUT / "mark4i_weights.json").read_text())[str(a.season)]
    fair = A.fair_o.astype(float)
    p4i = np.load(F.OUT / "forward_p_Mark_4i_0.npy")
    if len(p4i) != d.height:                       # the frame grew by the live week: earlier rows keep their forward chances
        base = np.full(d.height, np.nan)
        base[~te] = p4i[: (~te).sum()] if (~te).sum() == len(p4i) else np.nan
        p4i = base
    p4i = p4i.copy()
    p4i[te] = w["model_weight"] * p3w + (1 - w["model_weight"]) * fair[te]
    p4l = RC.mark4l(d, A, p4i)
    # FanDuel's own lines this week (the import may carry another book where FanDuel had not posted a market yet)
    lg = pl.read_csv(LOG, infer_schema_length=0).filter((pl.col("book") == "FanDuel") & pl.col("game_id").str.starts_with(f"{a.season}_{a.week:02d}_"))
    fd = lg.select("game_id", "player_id", "market", line=pl.col("line").cast(pl.Float64)).unique()
    x = d.with_row_index("i").filter(pl.Series(te)).join(fd, on=["game_id", "player_id", "market", "line"], how="semi")
    i = x["i"].to_numpy()
    p = p4l[i]
    eo, eu = p - A.be_o[i], (1 - p) - A.be_u[i]
    over = eo > eu
    rows = pl.DataFrame(dict(i=i, player=x["player_name"], team=x["team"], game_id=x["game_id"], gameday=x["gameday"].cast(pl.Utf8), market=x["market"],
                             line=x["line"], side=np.where(over, "Over", "Under"), chance=np.where(over, p, 1 - p), edge=np.where(over, eo, eu),
                             dec=np.where(over, A.dec_o[i], A.dec_u[i]), mark3=p3w[np.searchsorted(np.flatnonzero(te), i)], book=fair[i],
                             history=x["history"].to_numpy(), overreaction=(p4l[i] != p4i[i])))
    rows = rows.with_columns(ev=pl.col("chance") * pl.col("dec") - 1, price=pl.Series(american(rows["dec"].to_numpy())).round(0))
    elig = rows.filter((pl.col("edge") >= 0) & (pl.col("history") >= F.CF.MIN_HISTORY)).sort("ev", descending=True)
    legs = [(r["chance"], r["edge"], r["dec"], np.nan, r["game_id"], r["player"], r["market"], r) for r in elig.iter_rows(named=True)]
    legs = T.pool(legs, 40)
    used, card = set(), []
    for k, c, stake in PLAN:
        free = [r for j, r in enumerate(legs) if j not in used]
        for n, par in enumerate(T.build(free, k, c, "rank", "any")):
            for r in par:
                used.add(next(j for j, y in enumerate(legs) if y is r))
            dec = float(np.prod([r[2] for r in par]))
            chance = float(np.prod([r[0] for r in par]))
            boosted = 1 + (dec - 1) * (1 + T.BOOST)
            pid = f"{k}-leg" + (f" #{n + 1}" if c > 1 else "")
            for r in par:
                q = r[7]
                card.append(dict(parlay=pid, stake_units=stake, legs=k, parlay_price=round(float(american(np.array([dec]))[0])), boosted_price=round(float(american(np.array([boosted]))[0])),
                                 parlay_chance=round(chance, 4), parlay_ev_boosted=round(chance * boosted - 1, 3), player=q["player"], team=q["team"],
                                 market=MARKET_LABEL.get(q["market"], q["market"]), side=q["side"], line=q["line"], price=int(q["price"]),
                                 chance=round(q["chance"], 3), edge=round(q["edge"], 3), ev=round(q["ev"], 3), overreaction_under=bool(q["overreaction"]),
                                 game=q["game_id"], gameday=q["gameday"]))
    out = pl.DataFrame(card)
    path = OUTD / f"{a.season}_wk{a.week:02d}_card_mark4.csv"
    out.write_csv(path)
    path.with_name(path.stem + "_meta.json").write_text(json.dumps(dict(made_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), season=a.season, week=a.week,
                                                                        version="Mark 4l", plan=PLAN, weights=w), indent=1))
    elig.drop("i").write_csv(OUTD / f"{a.season}_wk{a.week:02d}_legs_mark4.csv")
    print(f"Mark 4l, {a.season} week {a.week}: {rows.height} FanDuel lines scored, {elig.height} with an edge (weights: Mark 3 {w['mark3_weight']}, model {w['model_weight']} vs the book), "
          f"{int(rows['overreaction'].sum())} promotion-overreaction lines")
    for pid, g in out.group_by("parlay", maintain_order=True):
        r0 = g.row(0, named=True)
        print(f"\n{pid[0]} @ {r0['stake_units']}u  price {r0['parlay_price']:+d} -> boosted {r0['boosted_price']:+d}  chance {r0['parlay_chance']:.1%}  expected value {r0['parlay_ev_boosted']:+.0%}")
        for r in g.iter_rows(named=True):
            print(f"   {r['player']:24s} {r['team']:4s} {r['market']:16s} {r['side']:5s} {r['line']:6.1f} @ {r['price']:+5d}  chance {r['chance']:.1%}  edge {r['edge']:+.1%}  {r['game']}"
                  + ("  [overreaction under]" if r["overreaction_under"] else ""))
    print("\nwrote", path)
    return out


if __name__ == "__main__":
    main()

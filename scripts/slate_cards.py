"""Weekly slate cards (yours, 2026-10-15): every under the tournament winner would take, ranked across all markets by the picker's chance, in tiers
(Top 10 / Next 10 / Rest), with suggested 2-leg boosted parlays from the Top 10 (different games), graded.

Each week the picker is retrained on every line before that week (2024, 2025, and earlier 2026 weeks): the procedure that will make the week-6 card.
Setup: gradient-boosted picker (depth 3, rate 0.1, leaf 150, layered chance), unders only, edge >= 4 points, markets pass_att, pass_yds, rush_att, rush_yds, rec, rec_yds.

  python scripts/slate_cards.py            # 2026 weeks 1-5, graded; records/cards/
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("winner_breakdown", HERE / "winner_breakdown.py")
WB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WB)
TN = WB.TN
OUTD = config.ROOT / "records" / "cards"
BOOST, STAKE = 0.30, 10.0
FAVOURITE = {"rush_att": "favourite", "pass_att": "favourite", "rec": "favourite", "rush_yds": "solid", "rec_yds": "shaky", "pass_yds": "shaky"}


def american(dec):
    return np.where(dec >= 2, np.round((dec - 1) * 100), np.round(-100 / (dec - 1)))


def card_for(d, A, X, season, week):
    tr = np.flatnonzero(((d["season"] < season) | ((d["season"] == season) & (d["week"] < week))).to_numpy())
    te = np.flatnonzero(((d["season"] == season) & (d["week"] == week)).to_numpy())
    clf = WB.make().fit(X[tr], A.over_won[tr])
    p_under = 1 - clf.predict_proba(X[te])[:, 1]
    edge = p_under - A.be_u[te]
    keep = (edge >= WB.THETA) & np.isin(A.mi[te], [TN.MARKETS.index(m) for m in WB.MARKETS_ON])
    x = d[te].with_columns(picker_under_chance=pl.Series(p_under), edge=pl.Series(edge), keep=pl.Series(keep)).filter(pl.col("keep"))
    x = x.with_columns(price=pl.Series(american(x["dec_u"].to_numpy())), won=(1 - pl.col("over_won")),
                       profit=pl.when(pl.col("over_won") == 0).then(pl.col("dec_u") - 1).otherwise(-1.0))
    x = x.sort("picker_under_chance", descending=True).with_row_index("rank", offset=1)
    x = x.with_columns(tier=pl.when(pl.col("rank") <= 10).then(pl.lit("Top 10")).when(pl.col("rank") <= 20).then(pl.lit("Next 10")).otherwise(pl.lit("Rest")),
                       market_rating=pl.col("market").replace_strict(FAVOURITE, default=""))
    return x, len(tr)


def parlays(top: pl.DataFrame):
    """Suggested 2-leg parlays: walk down the Top 10 by rank, pairing each unused leg with the next unused leg from a different game."""
    rows = top.sort("rank").to_dicts()
    used, out = set(), []
    for i, a in enumerate(rows):
        if a["rank"] in used:
            continue
        for b in rows[i + 1:]:
            if b["rank"] not in used and b["game_id"] != a["game_id"]:
                used |= {a["rank"], b["rank"]}
                dec = a["dec_u"] * b["dec_u"]
                hit = a["won"] == 1 and b["won"] == 1
                out.append(dict(leg1=f"#{a['rank']} {a['player_name']} {a['market']} u{a['line']:g}", leg2=f"#{b['rank']} {b['player_name']} {b['market']} u{b['line']:g}",
                                price=int(american(np.array([dec]))[0]), chance=a["picker_under_chance"] * b["picker_under_chance"], hit=hit,
                                profit=STAKE * (dec - 1) * (1 + BOOST) if hit else -STAKE, profit_no_boost=STAKE * (dec - 1) if hit else -STAKE))
                break
    return pl.DataFrame(out) if out else pl.DataFrame()


def all_pairs(top: pl.DataFrame):
    """Every 2-leg combination of the Top 10 in different games: average result per $10 (a steadier read than five parlays)."""
    r = top.to_dicts()
    res = [(STAKE * (a["dec_u"] * b["dec_u"] - 1) * (1 + BOOST) if a["won"] and b["won"] else -STAKE) for a, b in itertools.combinations(r, 2) if a["game_id"] != b["game_id"]]
    return len(res), float(np.mean(res)) if res else 0.0


def main(season=2026, weeks=range(1, 6)):
    OUTD.mkdir(parents=True, exist_ok=True)
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    summary, parl_all, cards = [], [], []
    for wk in weeks:
        x, ntr = card_for(d, A, X, season, wk)
        cards.append(x.with_columns(week=pl.lit(wk)))
        pp = parlays(x.filter(pl.col("tier") == "Top 10"))
        npairs, avg_pair = all_pairs(x.filter(pl.col("tier") == "Top 10"))
        if pp.height:
            parl_all.append(pp.with_columns(week=pl.lit(wk)))
        row = dict(week=wk, trained_on_lines=ntr, picks=x.height)
        for t in ("Top 10", "Next 10", "Rest"):
            z = x.filter(pl.col("tier") == t)
            row[f"{t} record"] = f"{int(z['won'].sum())}-{z.height - int(z['won'].sum())}" if z.height else "-"
            row[f"{t} units"] = round(float(z["profit"].sum()), 2) if z.height else 0.0
        row["parlays"] = f"{int(pp['hit'].sum())}/{pp.height}" if pp.height else "-"
        row["parlay $ (boost)"] = round(float(pp["profit"].sum()), 2) if pp.height else 0.0
        row["every Top-10 pair, avg $/10"] = round(avg_pair, 2)
        summary.append(row)
    S = pl.DataFrame(summary)
    C = pl.concat(cards)
    P = pl.concat(parl_all) if parl_all else pl.DataFrame()
    cols = ["week", "rank", "tier", "player_name", "team", "market", "market_rating", "line", "price", pl.col("mean").round(1).alias("model_prediction"),
            pl.col("picker_under_chance").round(3), pl.col("edge").round(3), "actual", pl.when(pl.col("won") == 1).then(pl.lit("WIN")).otherwise(pl.lit("LOSS")).alias("result"),
            pl.col("profit").round(3).alias("profit_units"), "game_id"]
    C.select(cols).write_csv(OUTD / f"{season}_weeks1-5_cards.csv")
    if P.height:
        P.write_csv(OUTD / f"{season}_weeks1-5_parlays.csv")
    S.write_csv(OUTD / f"{season}_weeks1-5_summary.csv")
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=60):
        print(S)
        for t in ("Top 10", "Next 10", "Rest"):
            z = C.filter(pl.col("tier") == t)
            print(f"{t}: {int(z['won'].sum())}-{z.height - int(z['won'].sum())} ({z['won'].mean():.1%}), {z['profit'].sum():+.2f} units, ROI {z['profit'].mean():+.1%}, avg price {z['price'].mean():.0f}")
        print("by market (Top 10 only):")
        print(C.filter(pl.col("tier") == "Top 10").group_by("market").agg(picks=pl.len(), hit=pl.col("won").mean(), units=pl.col("profit").sum()).sort("market"))
        if P.height:
            print(P)
            print(f"parlays: {int(P['hit'].sum())}/{P.height} hit, ${P['profit'].sum():+.2f} with boost, ${P['profit_no_boost'].sum():+.2f} without (staked ${STAKE * P.height:.0f})")
        for wk in weeks:
            print(f"\nWEEK {wk} TOP 10")
            print(C.filter((pl.col("week") == wk) & (pl.col("tier") == "Top 10")).select("rank", "player_name", "team", "market", "line", "price", pl.col("mean").round(1).alias("model"),
                                                                                         pl.col("picker_under_chance").round(3).alias("chance"), "actual", "won"))


if __name__ == "__main__":
    main()

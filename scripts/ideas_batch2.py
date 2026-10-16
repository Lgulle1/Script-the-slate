"""Untested ideas, batch 2 (yours, 2026-10-16).

A. Player variability (hidden seasons 2021-2026, same protocol as batch 1 / tournament 3). The model's chances use one spread for every player at a given projection;
   a boom-or-bust receiver and a steady one get the same odds of clearing a line. New picker inputs, from the player's own games in this market strictly before
   this game (any season): games, average and spread of the stat, spread relative to the average, spread of the model's misses, share of games 30%+ over the line.
   Judged on the locked rule (4 boosted 2-leg parlays from the top 8); also with the schedule inputs of candidate S.
B. FanDuel against the other books (2026 weeks 1-5 only: the all-books file covers only 2026). For each FanDuel line, the median line of the other regular books
   for the same player and market: is FanDuel's line above (under looks better), level, or below? Under hit rate and return by group, and the same split for
   the locked picker's legs. Descriptive only: one part-season, too small to retrain the picker on.

  python scripts/ideas_batch2.py --books /path/NFL_2026_Weeks1-6_All_Books_Player_Props.csv     # writes records/tuning/ideas_batch2.json
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


IB = _load("ideas_batch")
TN, CF, T2, ST = IB.TN, IB.CF, IB.T2, IB.ST
REGULAR = ["DraftKings", "Caesars", "Fanatics", "BetMGM", "theScore Bet", "Hard Rock", "BetRivers", "bet365"]      # other than FanDuel
BOOK_MARKET = {"pass_yds": "passing-yards", "rush_yds": "rushing-yards", "rec": "receptions", "rec_yds": "receiving-yards"}


def variability_feats(d):
    """Per row: the player's record in this market strictly before this game (one entry per game)."""
    x = d.select("player_id", "market", "game_id", "gameday", "line", "actual", "mean").with_row_index("i").with_columns(
        day=pl.col("gameday").cast(pl.String).str.slice(0, 10).str.to_date())
    g = (x.filter(pl.col("actual").is_not_null()).group_by("player_id", "market", "game_id", "day")
         .agg(a=pl.col("actual").first(), res=(pl.col("actual") - pl.col("mean")).first(), boom=(pl.col("actual") >= 1.3 * pl.col("line")).cast(pl.Float64).mean())
         .sort("player_id", "market", "day"))
    k = ["player_id", "market"]
    prev = lambda c: pl.col(c).cum_sum().over(k) - pl.col(c)
    g = g.with_columns(a2=pl.col("a") ** 2, r2=pl.col("res") ** 2).with_columns(
        n=pl.int_range(pl.len()).over(k).cast(pl.Float64), sa=prev("a"), sa2=prev("a2"), sr=prev("res"), sr2=prev("r2"), sb=prev("boom"))
    ok = pl.col("n") >= 3
    g = g.with_columns(
        avg=pl.when(ok).then(pl.col("sa") / pl.col("n")),
        sd=pl.when(ok).then(((pl.col("sa2") / pl.col("n")) - (pl.col("sa") / pl.col("n")) ** 2).clip(0).sqrt()),
        res_sd=pl.when(ok).then(((pl.col("sr2") / pl.col("n")) - (pl.col("sr") / pl.col("n")) ** 2).clip(0).sqrt()),
        boom_rate=pl.when(ok).then(pl.col("sb") / pl.col("n")))
    g = g.with_columns(cv=pl.col("sd") / pl.col("avg").clip(0.5))
    y = x.join(g.select(*k, "game_id", "n", "avg", "sd", "cv", "res_sd", "boom_rate"), on=[*k, "game_id"], how="left").sort("i")
    return y.select("n", "avg", "sd", "cv", "res_sd", "boom_rate").to_numpy().astype(float)


def books_gap(d, books_csv: Path):
    b = pl.read_csv(books_csv, infer_schema_length=0, encoding="utf8-lossy")
    b = b.rename({c: c.lstrip("﻿") for c in b.columns})
    b = b.filter((pl.col("selection") == "Under") & pl.col("sportsbook").is_in(REGULAR) & pl.col("market").is_in(list(BOOK_MARKET.values()))
                 & (pl.col("is_main_line").str.to_lowercase() == "true"))
    b = b.with_columns(line=pl.col("line").cast(pl.Float64), week=pl.col("week").cast(pl.Int64), key=pl.col("player").str.to_lowercase())
    cons = b.group_by("key", "market", "week").agg(cons_line=pl.col("line").median(), n_books=pl.col("sportsbook").n_unique())
    inv = {v: k for k, v in BOOK_MARKET.items()}
    cons = cons.with_columns(market=pl.col("market").replace_strict(inv))
    x = d.with_columns(key=pl.col("player_name").str.to_lowercase(), week=pl.col("week").cast(pl.Int64)).join(cons, on=["key", "market", "week"], how="left")
    return x["cons_line"].to_numpy(), x["n_books"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", type=Path, required=True)
    a = ap.parse_args()
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X = TN.feats(d, layered=True)
    sched, _ = IB.schedule_feats(d)
    var = variability_feats(d)
    makers = CF.ensemble_makers()
    out = {"A_variability": {}, "B_books": {}}
    preds = {}
    for name, Xv in {"baseline": X, "variability": np.hstack([X, var]), "schedule": np.hstack([X, sched]), "schedule + variability": np.hstack([X, sched, var])}.items():
        p = IB.hidden_preds(d, A, Xv, makers)
        preds[name] = p
        W = IB.week_rows(d, A, p)
        r = IB.summary(IB.run_rule(W))
        r["singles_top8_roi"] = IB.singles(W)
        r3 = IB.summary(IB.run_rule(W, n=9, k=3))
        r["three_leg_top9"] = dict(roi=r3["roi"], stake=r3["stake"], seasons={s: v["roi"] for s, v in r3["seasons"].items()})
        out["A_variability"][name] = r
        print(f"{name:<24} parlays {r['roi']:+.1%} on {r['stake']:.0f}  3-leg {r3['roi']:+.1%}  singles top 8 {r['singles_top8_roi']:+.1%}  " +
              " ".join(f"{s}:{v['roi']:+.0%}" for s, v in r["seasons"].items()), flush=True)
    b = out["A_variability"]["baseline"]["seasons"]
    for name, r in out["A_variability"].items():
        r["seasons_beating_baseline"] = sum(1 for s, v in r["seasons"].items() if v["roi"] > b[s]["roi"])

    # B: FanDuel vs the other books, 2026 weeks 1-5
    cons, nb = books_gap(d, a.books)
    x = d.select("season", "week", "market", "line", "history", "game_id", "player_id").with_columns(
        cons=pl.Series(cons), n_books=pl.Series(nb), won=pl.Series(1 - A.over_won), dec=pl.Series(A.dec_u), chance=pl.Series(1 - preds["baseline"]),
        edge=pl.Series((1 - preds["baseline"]) - A.be_u))
    x = x.filter((pl.col("season") == 2026) & pl.col("market").is_in(IB.FOUR) & pl.col("won").is_not_nan() & pl.col("cons").is_not_null())
    x = x.with_columns(gap=pl.col("line") - pl.col("cons"), profit=pl.when(pl.col("won") == 1).then(pl.col("dec") - 1).otherwise(-1.0))
    x = x.with_columns(group=pl.when(pl.col("gap") > 0).then(pl.lit("FanDuel line above the others")).when(pl.col("gap") < 0).then(pl.lit("FanDuel line below the others")).otherwise(pl.lit("level")))
    tab = x.group_by("group").agg(lines=pl.len(), under_hit=pl.col("won").mean(), under_roi=pl.col("profit").mean()).sort("group")
    out["B_books"]["all_lines"] = tab.to_dicts()
    q = x.filter((pl.col("edge") >= 0.04) & (pl.col("history") >= 3))
    legs = []
    for _, g in q.sort("chance", descending=True).group_by("week", maintain_order=True):
        legs += ST.top(g.to_dicts(), 8, ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], chance="chance", player="player_id")
    L = pl.DataFrame(legs)
    out["B_books"]["locked_legs"] = L.group_by("group").agg(legs=pl.len(), hit=pl.col("won").mean(), roi=pl.col("profit").mean()).sort("group").to_dicts()
    out["B_books"]["coverage"] = dict(lines_2026_graded_with_consensus=x.height)
    print(json.dumps(out["B_books"], indent=1, default=str))
    (config.ROOT / "records" / "tuning" / "ideas_batch2.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

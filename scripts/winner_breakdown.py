"""Break down the tournament winner: gradient-boosted picker (depth 3, rate 0.1, leaf 150, with the layered chance), unders only, edge >= 4 points, six markets.

Rebuilds it exactly as in scripts/tournament.py (training = 2024 + 2025 weeks 1-9, out-of-fold on training; fit on all training for 2025 weeks 10-18 and 2026), then reports
which inputs it leans on, how it decides, its results by market / season / price, and every bet (records/tuning/winner_bets.csv).

  python scripts/winner_breakdown.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tournament", HERE / "tournament.py")
TN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TN)
OUTD = config.ROOT / "records" / "tuning"
MARKETS_ON = ["pass_att", "pass_yds", "rush_att", "rush_yds", "rec", "rec_yds"]
THETA = 0.04
FEATURE_NAMES = ["book's chance of the over (vig removed)", "model's chance of the over (pool 600)", "model's chance with the tuned layers", "model's chance (pool 300)",
                 "model's chance (pool 1200)", "model vs line gap (%)", "the line itself", "price of the over", "price of the under", "game total", "player's team spread",
                 "home or away", "week of season", "dome", *[f"market = {m}" for m in TN.MARKETS]]


def make():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=200, min_samples_leaf=150, l2_regularization=1.0, random_state=7)


def main():
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    y = A.over_won
    tr = np.flatnonzero(A.si == 0)
    blocks = (d["season"] * 100 + d["week"]).to_numpy()
    p = TN.oof_and_fit(make, X, y, tr, blocks)
    over, edge, prof, won = TN.settle(A, p)
    bet = (edge >= THETA) & ~over & np.isin(A.mi, [TN.MARKETS.index(m) for m in MARKETS_ON])
    seg = d["seg"].to_numpy()
    # sanity: same totals as the tournament
    for s in TN.SEGS:
        k = bet & (seg == s)
        print(f"{s}: {k.sum()} bets, hit {won[k].mean():.1%}, ROI {prof[k].mean():+.1%}, units {prof[k].sum():+.1f}")

    # what it leans on: permutation importance on the training rows (out-of-fold model would be costly; use the model fit on training, scored on test A+B)
    from sklearn.inspection import permutation_importance
    clf = make().fit(X[tr], y[tr])
    te = np.flatnonzero(A.si > 0)
    imp = permutation_importance(clf, X[te], y[te], scoring="neg_log_loss", n_repeats=10, random_state=7)
    order = np.argsort(-imp.importances_mean)
    print("\nINPUTS BY IMPORTANCE (how much worse its test predictions get when the input is scrambled; log-loss x 1000):")
    for i in order:
        print(f"   {FEATURE_NAMES[i]:42s} {imp.importances_mean[i] * 1000:+7.2f}  (+/- {imp.importances_std[i] * 1000:.2f})")

    # how it decides: the under chance it gives vs the book's under chance, by model-vs-line gap
    t = d.with_columns(p_under=pl.Series(1 - p), book_under=1 - pl.col("fair_o"), bet=pl.Series(bet), won=pl.Series(won), prof=pl.Series(prof), edge=pl.Series(edge),
                       price=pl.Series(np.where(A.dec_u >= 2, np.round((A.dec_u - 1) * 100), np.round(-100 / (A.dec_u - 1)))))
    t = t.with_columns(gap_band=pl.col("gap").cut([-0.15, -0.05, 0.0, 0.05, 0.15], labels=["model 15%+ below line", "5-15% below", "0-5% below", "0-5% above", "5-15% above", "15%+ above"]))
    print("\nHOW IT DECIDES (all lines): its chance of the UNDER vs the book's, by where our model sits against the line")
    print(t.group_by("gap_band").agg(lines=pl.len(), its_under_chance=pl.col("p_under").mean(), book_under_chance=pl.col("book_under").mean(),
                                     share_it_bets=pl.col("bet").mean(), under_actually_hit=(1 - pl.col("over_won")).mean()).sort("gap_band"))
    b = t.filter(pl.col("bet"))
    print("\nBY MARKET (all its bets):")
    print(b.group_by("market", "seg").agg(bets=pl.len(), hit=pl.col("won").mean(), roi=pl.col("prof").mean(), units=pl.col("prof").sum()).sort("market", "seg"))
    print("\nBY SEASON:")
    print(b.group_by("season").agg(bets=pl.len(), hit=pl.col("won").mean(), roi=pl.col("prof").mean(), units=pl.col("prof").sum()).sort("season"))
    print("\nBY THE UNDER PRICE IT TOOK:")
    b = b.with_columns(price_band=pl.col("price").cut([-130, -115, -105], labels=["-130 or worse", "-129 to -116", "-115 to -106", "-105 or better"]))
    print(b.group_by("price_band").agg(bets=pl.len(), hit=pl.col("won").mean(), roi=pl.col("prof").mean()).sort("price_band"))
    print("\nBY EDGE SIZE:")
    b = b.with_columns(edge_band=pl.col("edge").cut([0.06, 0.08, 0.12], labels=["4-6", "6-8", "8-12", "12+"]))
    print(b.group_by("edge_band").agg(bets=pl.len(), hit=pl.col("won").mean(), roi=pl.col("prof").mean()).sort("edge_band"))
    print("\nBETS PER WEEK:", round(b.height / b.select("season", "week").n_unique(), 1))
    out = b.select("seg", "season", "week", "gameday", "player_name", "team", "market", "line", pl.lit("UNDER").alias("bet"), "price", pl.col("mean").round(2).alias("model_prediction"),
                   pl.col("book_under").round(3).alias("book_under_chance"), pl.col("p_under").round(3).alias("picker_under_chance"), pl.col("edge").round(3), "actual",
                   pl.when(pl.col("won") == 1).then(pl.lit("WIN")).otherwise(pl.lit("LOSS")).alias("result"), pl.col("prof").round(3).alias("profit_units")).sort("season", "week", "player_name")
    out.write_csv(OUTD / "winner_bets.csv")
    print("\nSAMPLE (test weeks):")
    with pl.Config(tbl_rows=12, tbl_width_chars=200):
        print(out.filter(pl.col("seg") != "train").sample(12, seed=3).sort("season", "week"))


if __name__ == "__main__":
    main()

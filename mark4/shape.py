"""Mark 4m: the shape correction (2026-10-20). FanDuel sets lines near a player's AVERAGE; players land near their MEDIAN (a few big games pull the
average up). When the line sits above his recent median the under beats the price in nearly every season (receptions 6/6, rushing 4/5, ...).

For each line: ratio = line / median of his last MED_GAMES results in the market (pre-game). Lines are grouped by market family x ratio bucket; each
group's book error (over hit - FanDuel's no-vig chance) is measured on EARLIER seasons only (recency-weighted, shrunk toward 0 with SHRINK lines) and
added to Mark 4l's chance (scaled by LAMBDA, chosen forward by log loss from {0, 0.5, 1, 1.5}).

  python -m mark4.shape        # writes records/lab/forward_p_Mark_4m_0.npy and prints the comparison
"""
from __future__ import annotations

import numpy as np
import polars as pl

from mark4 import forward as F

MED_GAMES, MIN_GAMES = 12, 6
EDGES = [0.85, 0.95, 1.05, 1.15, 1.3]
SHRINK = 400.0
LAMBDAS = (0.0, 0.5, 1.0, 1.5)
SEASONS = (2022, 2023, 2024, 2025, 2026)


def groups(d: pl.DataFrame) -> np.ndarray:
    x = d.with_row_index("i").select("i", "player_id", "market", "gameday", "line", "actual").sort("player_id", "market", "gameday")
    x = x.with_columns(med=pl.col("actual").shift(1).rolling_median(MED_GAMES, min_samples=MIN_GAMES).over("player_id", "market"))
    x = x.with_columns(ratio=pl.col("line") / pl.col("med").clip(0.5, None)).sort("i")
    fam = (pl.when(pl.col("market").is_in(["rec"])).then(pl.lit("rec")).when(pl.col("market") == "rec_yds").then(pl.lit("rec_yds"))
           .when(pl.col("market").str.starts_with("rush")).then(pl.lit("rush")).otherwise(pl.lit("pass")))
    x = x.with_columns(b=pl.col("ratio").cut(EDGES, labels=[str(i) for i in range(len(EDGES) + 1)]).cast(pl.Utf8))
    return x.select(pl.when(pl.col("b").is_not_null()).then(fam + "|" + pl.col("b")).otherwise(None)).to_series().to_numpy()


def forward_bias(g, y, fair, season):
    out = np.zeros(len(y))
    for S in SEASONS:
        past = (season < S) & ~np.isnan(y) & (g != None)  # noqa: E711
        w = 0.5 ** (S - season[past])
        res = (y - fair)[past]
        gp = g[past]
        table = {}
        for k in set(gp):
            m = gp == k
            table[k] = float((w[m] * res[m]).sum() / (w[m].sum() + SHRINK))
        cur = season == S
        out[cur] = [table.get(k, 0.0) if k is not None else 0.0 for k in g[cur]]
    return out


def main():
    d, A, *_ = F.frame()
    y, fair, season = A.over_won.astype(float), A.fair_o.astype(float), d["season"].to_numpy()
    g = groups(d)
    bias = forward_bias(g, y, fair, season)
    p4l = np.load(F.OUT / "forward_p_Mark_4l_0.npy")
    ok = ~np.isnan(y) & ~np.isnan(fair) & ~np.isnan(p4l)
    cl = lambda q: np.clip(q, 1e-4, 1 - 1e-4)
    ll = lambda q: -(y * np.log(cl(q)) + (1 - y) * np.log(1 - cl(q)))
    # lambda chosen forward
    p = p4l.copy()
    for S in SEASONS:
        past = ok & (season < S)
        if past.sum() < 2000:
            lam = 1.0
        else:
            w = 0.5 ** (S - season[past])
            lam = max(LAMBDAS, key=lambda L: (w * (ll(p4l) - ll(p4l + L * bias))[past]).sum())
        cur = season == S
        p[cur] = np.clip(p4l[cur] + lam * bias[cur], 0.01, 0.99)
        print(f"   {S}: shape correction weight {lam}")
    np.save(F.OUT / "forward_p_Mark_4m_0.npy", p)
    for name, q in (("Mark 4l", p4l), ("Mark 4m", p), ("book + shape only", fair + bias)):
        gs = {int(s): round(1000 * float((ll(fair) - ll(q))[ok & (season == s)].mean()), 2) for s in SEASONS}
        print(f"   {name:18s} log-loss gain vs the book per 1,000 lines {gs}")
    return p


if __name__ == "__main__":
    main()

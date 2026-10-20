"""Comps v5: comps of the BOOK's behaviour (2026-10-20). Earlier comps compared players' stats; FanDuel already prices that. These compare LINES:
for each line, the most similar earlier lines (how the line moved and why, the player's last games vs his lines, his role change, the opponent's
record vs the line, the game script, the line's size and price shape) and how often FanDuel was wrong there.

Context of a line (all pre-game):
  line_form    log(line / his recency-weighted actual over his last games in the market)    the line vs his real form
  line_jump    log(line / his previous line in the market)                                 how far the book moved it
  last_vs_line log(his last actual / his previous line)                                    did he just smash or miss
  player_tend  recency-weighted (over - book) of his earlier lines;   def_tend: same for the opponent's defense in the market
  role         vacancy in his role, game N of it, engine's predicted share jump, teammates back, QB change (mark4.rolechange)
  game         team implied total, spread;   price   book's over chance, hold, line size within market, Thursday
Two forward learners, each 6-week block fitted only on lines graded before it:
  knn   the K most similar earlier lines (standardised, features weighted by the gradient-boosting importance from the previous block) -> shrunk
        mean of (over - book chance);   hgb  a small gradient-boosting model of (over - book chance) on the same context
Output: book chance + predicted error = Mark 4 "context" chance; then mixed with Mark 4l, weight chosen forward.

  python -m mark4.context
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from mark4 import forward as F
from mark4 import rolechange as RC

BLOCKS = (1, 7, 13)
SEASONS = (2022, 2023, 2024, 2025, 2026)
K, SHRINK = 300, 300.0
FEATS = ["line_form", "line_jump", "last_vs_line", "player_tend", "def_tend", "vac", "vac_nth", "role_jump", "back", "qb_change",
         "team_pts", "team_spread", "fair", "hold", "line_pct", "thursday", "mk_rec", "mk_recyds", "mk_rush", "mk_pass"]


def frame():
    d, A, *_ = F.frame()
    x = d.with_row_index("i").select("i", "season", "week", "gameday", "game_id", "player_id", "market", "line", "actual", "team", "home_team",
                                     "total_line", "team_spread")
    x = x.with_columns(won=pl.Series(A.over_won.astype(float)), fair=pl.Series(A.fair_o.astype(float)), hold=pl.Series(A.be_o + A.be_u - 1))
    R = RC.role_block(d)
    x = x.with_columns(**{c: pl.Series(R[:, j]) for j, c in enumerate(RC.ROLE_COLS) if c in ("vac", "vac_nth", "role_jump", "back", "qb_change")})
    x = x.sort("player_id", "market", "gameday")
    g = ["player_id", "market"]
    x = x.with_columns(prev_line=pl.col("line").shift(1).over(g), prev_actual=pl.col("actual").shift(1).over(g),
                       form=pl.col("actual").shift(1).ewm_mean(half_life=4, ignore_nulls=True).over(g),
                       res=pl.col("won") - pl.col("fair"))
    x = x.with_columns(player_tend=pl.col("res").shift(1).ewm_mean(half_life=8, ignore_nulls=True).over(g))
    opp = pl.when(pl.col("team") == pl.col("home_team")).then(pl.lit("away")).otherwise(pl.lit("home"))
    import duckdb, config
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, home_team, away_team, weekday FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    x = x.join(sc.select("game_id", "away_team", "weekday"), on="game_id", how="left")
    x = x.with_columns(opp=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_team")).otherwise(pl.col("home_team"))).sort("opp", "market", "gameday")
    x = x.with_columns(def_tend=pl.col("res").shift(1).ewm_mean(half_life=12, ignore_nulls=True).over("opp", "market"))
    lg = lambda a, b: (pl.col(a).clip(0.5, None) / pl.col(b).clip(0.5, None)).log()
    x = x.with_columns(line_form=lg("line", "form"), line_jump=lg("line", "prev_line"), last_vs_line=lg("prev_actual", "prev_line"),
                       team_pts=pl.col("total_line") / 2 - pl.col("team_spread") / 2,
                       line_pct=pl.col("line").rank("average").over("market", "season") / pl.len().over("market", "season"),
                       thursday=(pl.col("weekday") == "Thursday").cast(pl.Float64),
                       mk_rec=(pl.col("market") == "rec").cast(pl.Float64), mk_recyds=(pl.col("market") == "rec_yds").cast(pl.Float64),
                       mk_rush=pl.col("market").str.starts_with("rush").cast(pl.Float64), mk_pass=pl.col("market").str.starts_with("pass").cast(pl.Float64))
    return x.sort("i"), d, A


def forward(x: pl.DataFrame):
    X = x.select(FEATS).cast(pl.Float64).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    y = x["res"].to_numpy()
    season, week = x["season"].to_numpy(), x["week"].to_numpy()
    ok = ~np.isnan(y)
    out = {"knn": np.full(len(y), np.nan), "hgb": np.full(len(y), np.nan)}
    imp = np.ones(len(FEATS))
    for S in SEASONS:
        for i, b0 in enumerate(BLOCKS):
            b1 = BLOCKS[i + 1] if i + 1 < len(BLOCKS) else 99
            te = (season == S) & (week >= b0) & (week < b1)
            tr = ok & ((season < S) | ((season == S) & (week < b0)))
            if not te.any() or tr.sum() < 3000:
                continue
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False)
            m.fit(X[tr], y[tr])
            out["hgb"][te] = m.predict(X[te])
            # kNN: impute medians, standardise on training rows, weight by |correlation-ish| importance from the gradient-boosting fit (permutation-free proxy)
            med = np.nanmedian(X[tr], axis=0)
            Z = np.where(np.isnan(X), med, X)
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
            Zs = (Z - mu) / sd
            gain = np.abs([np.corrcoef(Zs[tr][:, j], m.predict(X[tr]))[0, 1] if sd[j] > 1e-6 else 0 for j in range(len(FEATS))])
            w = np.nan_to_num(gain) + 0.05
            nn = NearestNeighbors(n_neighbors=K).fit(Zs[tr] * w)
            idx = nn.kneighbors(Zs[te] * w, return_distance=False)
            yt = y[tr]
            out["knn"][te] = yt[idx].sum(1) / (K + SHRINK)
    return out


def main():
    x, d, A = frame()
    out = forward(x)
    y, fair, season = x["won"].to_numpy(), x["fair"].to_numpy(), x["season"].to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(fair)
    cl = lambda q: np.clip(q, 1e-4, 1 - 1e-4)
    ll = lambda q: -(y * np.log(cl(q)) + (1 - y) * np.log(1 - cl(q)))
    p4l = np.load(F.OUT / "forward_p_Mark_4l_0.npy")
    cands = {"Mark 4l": p4l}
    for k, r in out.items():
        cands[f"book + {k} comps of the line"] = np.where(np.isnan(r), fair, fair + r)
    for k, r in out.items():
        cands[f"Mark 4l + {k} (shift by its predicted book error)"] = np.where(np.isnan(r), p4l, p4l + r)
    for name, q in cands.items():
        g = {int(s): round(1000 * float(np.nanmean((ll(fair) - ll(q))[ok & (season == s)])), 2) for s in SEASONS}
        print(f"   {name:52s} log-loss gain vs the book per 1,000 lines {g}")
    for k, r in out.items():
        np.save(F.OUT / f"context_{k}.npy", r)
    # straights: bet where the predicted book error is big
    for k, r in out.items():
        for t in (0.02, 0.04, 0.06):
            sel = ok & ~np.isnan(r) & (np.abs(r) >= t)
            over = r > 0
            w = np.where(over, y, 1 - y)
            ret = np.where(w == 1, np.where(over, A.dec_o, A.dec_u) - 1, -1.0)
            by = {int(s): (int((sel & (season == s)).sum()), round(float(ret[sel & (season == s)].mean()), 3)) for s in SEASONS if (sel & (season == s)).any()}
            print(f"   {k}: predicted book error >= {t:.0%}: {sel.sum():5d} bets, overs {over[sel].mean():.0%}, return {ret[sel].mean():+.1%}  {by}")
    return out


if __name__ == "__main__":
    main()

"""Does the lineup-adjusted target vector (4c.1.4) describe the coming game better than the healthy one?  (prints; writes nothing)

  python run_lineup_adjustment_check.py

The numbers behind docs/lineup_adjustment_memo.md. For the 2020-2024 team-games with 4a expectations and the recency_weighted window, builds the
lineup corrections of the lineup-corrected features four ways and compares them with the healthy (unadjusted) window value:
  now            the current construction: 4a expected minus 4a baseline shares
  window         4a expected shares (renormalised over the game's lineup players) minus the window's normal shares
  rebased        each player's window share x his 4a expected / baseline ratio, renormalised, minus the window's normal shares
  participation  the pool's 'who actually played' recipe with the 4a play probability: window shares x P(play), renormalised
Yardsticks (league SDs of the window values): (a) distance to the pool's 'actual' vector, (b) RMSE against the game's own realized value, on all games
and on the 10% with the largest lineup move, (c) the calibration slope of the realized change on the predicted change; then (d) the 'now' and
'participation' corrections with the player values shrunk by K = 5 / 50 / 200 opportunities. Inputs are capped at 2024; 2025 is never loaded.
"""
import numpy as np
import polars as pl

import config
from models import comps as C
from models import comps_spec as cs

VARIANTS = ("healthy", "now", "window", "rebased", "participation")


def main():
    inp = C.load_inputs()
    V = [v for v in C.VARIANTS if C.vname(v) == "recency_weighted"]
    team_keys = [k for u in C.TEAM_UNITS for k in C.feature_keys(u)]
    gidx = {(g, t): i for i, (g, t) in enumerate(zip(inp.games["game_id"].to_list(), inp.games["team"].to_list()))}
    tw = C.team_windows(inp.games, inp.lineups, inp.team, team_keys, V)
    q = C.build_queries(inp)
    pw = C.build_player_windows(inp, q, C.player_keys(inp.player), V)
    pws = C.build_player_windows(inp, q, list(C.SHARE_KEYS.values()), V, ledger=C.share_ledger(inp))
    lm = pw.queries.select("game_id", "team", "player_id").join(C.lineup_queries(inp).with_columns(_l=pl.lit(True)), on=["game_id", "team", "player_id"],
                                                                 how="left", maintain_order="left")["_l"].fill_null(False).to_numpy()
    pl_, ps_ = C.subset_windows(pw, lm), C.subset_windows(pws, lm)
    shares = C.share_tables(inp, pl_.queries)
    gid = C._gid(shares)
    ng = int(gid.max()) + 1

    def renorm(x):
        tot = np.bincount(gid, weights=x, minlength=ng)
        return np.divide(x, tot[gid], out=np.zeros_like(x), where=tot[gid] > 0)

    sidx = {k: i for i, k in enumerate(ps_.keys)}
    norm = {k: renorm(np.nan_to_num(ps_.rate[0][:, sidx[key]].astype(np.float64), nan=0.0)) for k, key in C.SHARE_KEYS.items()}
    pout = shares.select("game_id", "team", "player_id").join(
        inp.detail.filter(pl.col("player_id") != "rest").select("game_id", "team", "player_id", "p_out", listed=pl.lit(True)),
        on=["game_id", "team", "player_id"], how="left", maintain_order="left")
    pplay = (1.0 - pout["p_out"].fill_null(1.0).to_numpy()) * pout["listed"].fill_null(False).to_numpy()
    tables = {"now": shares, "window": shares, "rebased": shares, "participation": shares}
    for k in ("carry", "target", "dropback"):
        e, b = shares[f"exp_{k}"].to_numpy(), shares[f"b_{k}"].to_numpy()
        reb = np.where(b > 0, norm[k] * np.divide(e, b, out=np.zeros_like(e), where=b > 0), e)
        for name, new in (("window", renorm(e)), ("rebased", renorm(reb)), ("participation", renorm(norm[k] * pplay))):
            tables[name] = tables[name].with_columns(pl.Series(f"exp_{k}", new), pl.Series(f"b_{k}", norm[k]))

    def deltas(name):
        return C.lineup_deltas(pl_, ps_, tables[name], tw, team_keys, gidx, inp.games.height)[V[0]]

    d = {name: deltas(name) for name in tables}
    rate, den = tw[V[0]]
    wk = (inp.games["season"].to_numpy() * 100 + inp.games["week"].to_numpy()).astype(np.int64)
    _, _, sd = C.league_z(rate, den, wk)
    kidx = {k: i for i, k in enumerate(team_keys)}
    have = ~np.isnan(d["now"]["adjusted"]).all(axis=1)
    real = inp.games.select("game_id", "team").join(inp.team, on=["game_id", "team"], how="left", maintain_order="left")
    ra, rb, rc, rd = [], [], [], []
    for j, key in enumerate(C.DELTA_KEYS):
        c = kidx[key]
        s = sd[:, c]
        clip = (lambda x: np.clip(x, 0, 1)) if key in C.BOUNDED_01 else (lambda x: x)
        cand = {"healthy": rate[:, c], **{n: clip(rate[:, c] + np.nan_to_num(d[n]["adjusted"][:, j])) for n in tables}}
        act = clip(rate[:, c] + np.nan_to_num(d["now"]["actual"][:, j]))
        nn, dd = real[f"{key}|n"].to_numpy().astype(float), real[f"{key}|d"].to_numpy().astype(float)
        y = np.divide(nn, dd, out=np.full(len(nn), np.nan), where=dd > 0)
        ok = have & np.isfinite(s) & (s > 0) & np.isfinite(rate[:, c])
        oky = ok & np.isfinite(y)
        ra.append(dict(feature=key, **{n: float(np.mean(np.abs(v[ok] - act[ok]) / s[ok])) for n, v in cand.items()}))
        rb.append(dict(feature=key, **{n: float(np.sqrt(np.mean(((v[oky] - y[oky]) / s[oky]) ** 2))) for n, v in cand.items()}))
        mv = np.maximum(np.abs(cand["now"] - rate[:, c]), np.abs(cand["participation"] - rate[:, c])) / s
        big = oky & (mv >= np.nanquantile(np.where(oky, mv, np.nan), 0.9))
        rc.append(dict(feature=key, **{n: float(np.sqrt(np.mean(((v[big] - y[big]) / s[big]) ** 2))) for n, v in cand.items()}))
        for n, v in cand.items():
            if n == "healthy":
                continue
            x_, y_ = (v[oky] - rate[oky, c]) / s[oky], (y[oky] - rate[oky, c]) / s[oky]
            m = np.abs(x_) > 1e-9
            rd.append(dict(feature=key, variant=n, slope=float((x_[m] * y_[m]).sum() / (x_[m] ** 2).sum()) if m.sum() > 10 else float("nan"),
                           corr=float(np.corrcoef(x_[m], y_[m])[0, 1]) if m.sum() > 10 else float("nan")))
    unit = pl.col("feature").str.split(".").list.first().alias("unit")
    by_unit = lambda rows: pl.DataFrame(rows).with_columns(unit).group_by("unit").agg(*[pl.col(n).mean().round(4) for n in VARIANTS]).sort("unit")
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print("(a) mean |vector - the pool's 'actual' vector|, league SDs:\n", by_unit(ra))
        print("(b) RMSE against the game's realized value:\n", by_unit(rb))
        print("    the 10% of games with the largest lineup move:\n", by_unit(rc))
        print("(c) calibration (median over features): slope and correlation of the realized change on the predicted change:\n",
              pl.DataFrame(rd).with_columns(unit).group_by("unit", "variant").agg(pl.col("slope").median().round(3), pl.col("corr").median().round(3))
              .sort("unit", "variant"))
    # (d) shrink the player values harder
    rows = []
    k0 = cs.SHRINK_K
    try:
        for K in (5, 50, 200):
            cs.SHRINK_K = K
            for name in ("now", "participation"):
                dk = deltas(name)["adjusted"]
                for j, key in enumerate(C.RUN_KEYS + C.PASS_KEYS):
                    c = kidx[key]
                    s = sd[:, c]
                    nn, dd = real[f"{key}|n"].to_numpy().astype(float), real[f"{key}|d"].to_numpy().astype(float)
                    y = np.divide(nn, dd, out=np.full(len(nn), np.nan), where=dd > 0)
                    oky = have & np.isfinite(s) & (s > 0) & np.isfinite(rate[:, c]) & np.isfinite(y)
                    x_, y_ = np.nan_to_num(dk[oky, j]) / s[oky], (y[oky] - rate[oky, c]) / s[oky]
                    rows.append(dict(K=K, variant=name, feature=key, rmse_healthy=float(np.sqrt(np.mean(y_ ** 2))),
                                     rmse_adjusted=float(np.sqrt(np.mean((x_ - y_) ** 2))), mean_abs_move=float(np.mean(np.abs(x_)))))
    finally:
        cs.SHRINK_K = k0
    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        print("(d) player values shrunk by K opportunities (run / pass offense):\n",
              pl.DataFrame(rows).with_columns(unit).group_by("unit", "variant", "K").agg(pl.col("rmse_healthy", "rmse_adjusted", "mean_abs_move").mean().round(4))
              .sort("unit", "variant", "K"))
    print(f"\nholdout {config.HOLDOUT_SEASON} untouched (inputs capped at {max(config.BACKTEST_SEASONS)})")


if __name__ == "__main__":
    main()

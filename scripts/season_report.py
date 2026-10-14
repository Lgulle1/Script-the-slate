"""Grade a replayed season against the real results (no sportsbook line is used for the player markets; the schedule's game lines are used for the game markets only).

Used for the one-time 2025 benchmark (docs/overnight_plan.md, "Improvement plan from weeks 1-5", workstream A) and, to debug, on the 2026 reconstruction store.
  python scripts/season_report.py --store records/holdout_2025/predictions --season 2025 --tag holdout_2025
Writes data/processed/season_report_<tag>_{markets,bias,bars,games}.parquet. Descriptive; no gate.
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import baselines as bl
from eval import calibrate as C
from live import results as RS
from live import store as ST

OUTD = config.PROCESSED_DIR
IS_GAME = pl.col("market").is_in(["spread", "total", "moneyline"])
LEVELS = (0.9, 0.8, 0.7, 0.6, 0.5)
DRAWS, SEED = 2000, 7


def boot(d: np.ndarray, cluster: np.ndarray) -> tuple:
    u, inv = np.unique(cluster, return_inverse=True)
    s, n = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    idx = np.random.default_rng(SEED).integers(0, len(u), size=(DRAWS, len(u)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def load(store: Path, season: int) -> pl.DataFrame:
    preds = ST.PredictionStore(store).load().sort("version").group_by(ST.KEY, maintain_order=True).last()
    act, plog, tlog = RS.played(season)
    j = preds.join(act, on=["game_id", "player_id", "market"], how="left", nulls_equal=True).filter(pl.col("game_id").is_in(act["game_id"].unique().to_list()))
    j = j.with_columns(did_not_play=pl.col("actual").is_null() & ~IS_GAME, actual=pl.when(pl.col("actual").is_null() & ~IS_GAME).then(0.0).otherwise(pl.col("actual")))
    j = j.with_columns(week=pl.col("game_id").str.split("_").list.get(1).cast(pl.Int64))
    return j.sort("game_id", "player_id", "market", nulls_last=True), plog


def trail4(j: pl.DataFrame, plog_all: pl.DataFrame, season: int) -> pl.DataFrame:
    parts = []
    for wk in sorted(j["week"].unique().to_list()):
        w = j.filter(pl.col("week") == wk)
        cutoff = w["kickoff_time"].min().date()
        hist = plog_all.filter(pl.col("gameday") < cutoff).sort("player_id", "gameday")
        out = [hist.group_by("player_id", maintain_order=True).agg(v=pl.col(col).cast(pl.Float64).tail(4).mean()).select("player_id", market=pl.lit(m), trail4="v") for m, col in bl.PLAYER_MARKETS.items()]
        parts.append(w.join(pl.concat(out), on=["player_id", "market"], how="left"))
    return pl.concat(parts)


def markets_table(j: pl.DataFrame, plog: pl.DataFrame) -> pl.DataFrame:
    played = j.filter(~pl.col("did_not_play") & pl.col("actual").is_not_null() & ~IS_GAME & pl.col("trail4").is_not_null())
    rows = []
    bm = RS.baseline_mae(j.filter(~IS_GAME), plog).group_by("market").agg(n_best=pl.col("n_compared").sum(), model_best=(pl.col("model_mae") * pl.col("n_compared")).sum() / pl.col("n_compared").sum(),
                                                                             best_baseline=(pl.col("baseline_mae") * pl.col("n_compared")).sum() / pl.col("n_compared").sum())
    for m, t in sorted(played.partition_by("market", as_dict=True).items()):
        m = m[0]
        e, b = np.abs(t["mean"].to_numpy() - t["actual"].to_numpy()), np.abs(t["trail4"].to_numpy() - t["actual"].to_numpy())
        g, lo, hi = boot(b - e, (t["week"].to_numpy()))
        has = t.filter(pl.col("q10").is_not_null())
        cov = float(np.mean((has["actual"].to_numpy() >= has["q10"].to_numpy()) & (has["actual"].to_numpy() <= has["q90"].to_numpy()))) if has.height else None
        rows.append(dict(market=m, n=t.height, mae=float(e.mean()), mae_trail4=float(b.mean()), gain_vs_trail4=g, lo=lo, hi=hi, skill_pct=float(100 * g / b.mean()), cov_10_90=cov))
    return pl.DataFrame(rows).join(bm, on="market", how="left").with_columns(skill_vs_best_pct=100 * (1 - pl.col("model_best") / pl.col("best_baseline"))).sort("market")


def bias_table(j: pl.DataFrame) -> pl.DataFrame:
    p = j.filter(~pl.col("did_not_play") & ~IS_GAME & pl.col("actual").is_not_null())
    rows = []
    for m, t in sorted(p.partition_by("market", as_dict=True).items()):
        a, f = t["actual"].to_numpy(), t["mean"].to_numpy()
        rows.append(dict(source="this season", market=m[0], n=t.height, share_above=float(np.mean(a > f)), mean_actual_minus_pred=float(np.mean(a - f))))
    dev = pl.read_parquet(OUTD / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet").filter((pl.col("method") == "vol_x_eff") & pl.col("market").is_in(list(bl.PLAYER_MARKETS))).drop_nulls(["prediction", "actual"])
    for m, t in sorted(dev.partition_by("market", as_dict=True).items()):
        a, f = t["actual"].to_numpy(), t["prediction"].to_numpy()
        rows.append(dict(source="2020-2024 walk-forward", market=m[0], n=t.height, share_above=float(np.mean(a > f)), mean_actual_minus_pred=float(np.mean(a - f))))
    return pl.DataFrame(rows).sort("market", "source")


def sample_for(cal, market: str, p: float):
    v = cal.markets.get(market)
    if v is None or len(v["pred"]) < cal.cfg.min_pool[market]:
        return None
    s = v["act"][C._window(v["pred"], np.array([p]), cal.cfg.k[market])[0]].astype(float)
    c = v["c"] if v["use_width"] else 1.0
    med = float(np.quantile(s, 0.5))
    return med + c * (s - med)


def bars_table(j: pl.DataFrame) -> pl.DataFrame:
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    p = j.filter(~pl.col("did_not_play") & ~IS_GAME & pl.col("actual").is_not_null() & pl.col("market").is_in(["pass_yds", "rush_yds", "rec", "rec_yds"]))
    rows = []
    for r in p.iter_rows(named=True):
        s = sample_for(cal, r["market"], r["mean"])
        if s is None:
            continue
        for l in LEVELS:
            f = float(np.floor(np.quantile(s, 1 - l, method="lower")))
            rows.append(dict(market=r["market"], level=l, week=r["week"], bar=f, exact=float((s >= f).mean()), hit=float(r["actual"] >= f)))
    d = pl.DataFrame(rows)
    out = []
    for key, t in sorted(d.partition_by("market", "level", as_dict=True).items()):
        g, lo, hi = boot(t["hit"].to_numpy(), t["week"].to_numpy())
        out.append(dict(market=key[0], level=key[1], n=t.height, stated_exact=float(t["exact"].mean()), hit=g, lo=lo, hi=hi, share_bar_zero=float((t["bar"] <= 0).mean())))
    allr = []
    for l in LEVELS:
        t = d.filter(pl.col("level") == l)
        g, lo, hi = boot(t["hit"].to_numpy(), t["week"].to_numpy())
        allr.append(dict(market="ALL", level=l, n=t.height, stated_exact=float(t["exact"].mean()), hit=g, lo=lo, hi=hi, share_bar_zero=float((t["bar"] <= 0).mean())))
    return pl.DataFrame(out + allr).sort("market", "level")


def games_table(j: pl.DataFrame, season: int) -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    s = con.execute("SELECT game_id, spread_line, total_line, home_moneyline FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) WHERE season = ?", [season]).pl()
    con.close()
    g = j.filter(IS_GAME & pl.col("actual").is_not_null()).select("game_id", "market", "mean", "actual", "week")
    rows = []
    sp = g.filter(pl.col("market") == "spread").join(s, on="game_id")
    if sp.height:
        e, b = np.abs(sp["mean"].to_numpy() - sp["actual"].to_numpy()), np.abs(sp["spread_line"].to_numpy() - sp["actual"].to_numpy())
        gg, lo, hi = boot(b - e, sp["week"].to_numpy())
        rows.append(dict(market="spread", n=sp.height, model_mae=float(e.mean()), line_mae=float(b.mean()), line_minus_model=gg, lo=lo, hi=hi))
    tt = g.filter(pl.col("market") == "total").join(s, on="game_id")
    e, b = np.abs(tt["mean"].to_numpy() - tt["actual"].to_numpy()), np.abs(tt["total_line"].to_numpy() - tt["actual"].to_numpy())
    gg, lo, hi = boot(b - e, tt["week"].to_numpy())
    rows.append(dict(market="total", n=tt.height, model_mae=float(e.mean()), line_mae=float(b.mean()), line_minus_model=gg, lo=lo, hi=hi))
    ml = g.filter(pl.col("market") == "moneyline").join(s, on="game_id")
    home_won = ml["actual"].to_numpy() > 0.5
    rows.append(dict(market="winner picked (model)", n=ml.height, model_mae=float(np.mean((ml["mean"].to_numpy() > 0.5) == home_won)), line_mae=float(np.mean((ml["spread_line"].to_numpy() > 0) == home_won)), line_minus_model=None, lo=None, hi=None))
    return pl.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    j, plog = load(a.store, a.season)
    from live import history as H
    plog_all = H.read_played_logs(a.season)[0]
    j = trail4(j, plog_all, a.season)
    tabs = dict(markets=markets_table(j, plog), bias=bias_table(j), bars=bars_table(j), games=games_table(j, a.season))
    for k, t in tabs.items():
        t.write_parquet(OUTD / f"season_report_{a.tag}_{k}.parquet")
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3, tbl_width_chars=220):
        for k, t in tabs.items():
            print(f"-- {k}"); print(t)
    return tabs


if __name__ == "__main__":
    main()

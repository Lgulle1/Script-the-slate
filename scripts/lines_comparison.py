"""Lines comparison, game markets (docs/overnight_plan.md, "Lines comparison, game markets"): an after-the-fact evaluation, never a model input.

Development seasons 2020-2024 only (asserted). The line is read from the schedule table to SCORE the committed walk-forward forecasts; nothing here feeds a model, a record or a weight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import compare
from live import layer_sim

LAST = config.HOLDOUT_SEASON - 1
PRED = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
THRESHOLDS = (1.0, 2.0, 3.0)
BREAKEVEN = 110 / 210                      # hit rate that breaks even at the standard -110 price
N_CELLS = 12                               # 3 margin forecasts x 3 thresholds + 3 total thresholds (plan)
LEVEL = 1 - 0.05 / N_CELLS


def load() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sch = con.execute(f"""SELECT game_id, season, week, spread_line, total_line, home_moneyline, away_moneyline, result, total FROM
                          (SELECT DISTINCT ON (game_id) * FROM schedules WHERE game_type = 'REG' AND season BETWEEN 2020 AND {LAST} ORDER BY game_id, pulled_at DESC)""").pl()
    con.close()
    assert sch["season"].max() <= LAST
    # sign check (plan): a positive spread_line must favour the home side, i.e. correlate positively with the home margin
    c = np.corrcoef(sch.drop_nulls(["spread_line", "result"])["spread_line"], sch.drop_nulls(["spread_line", "result"])["result"])[0, 1]
    assert c > 0.3, f"spread_line sign convention not as planned (corr {c:.2f})"
    g = pl.read_parquet(PRED).filter((pl.col("method") == "vol_x_eff") & (pl.col("kind") == "game") & (pl.col("season") <= LAST))
    for m, nm in (("spread", "p3_spread"), ("total", "p3_total"), ("moneyline", "p3_home_win")):
        sch = sch.join(g.filter(pl.col("market") == m).select(game_id="entity", **{nm: "prediction"}), on="game_id", how="left")
    ge = pl.read_parquet(config.PROCESSED_DIR / "game_expectations.parquet").select("game_id", "margin_mean", "margin_sd")
    ge = ge.with_columns(sim_spread=pl.struct("game_id", "margin_mean", "margin_sd").map_elements(
        lambda r: layer_sim.spread_sim_mean(r["game_id"], r["margin_mean"], r["margin_sd"]) if r["margin_mean"] is not None and r["margin_sd"] is not None else None, return_dtype=pl.Float64))
    sch = sch.join(ge.rename({"margin_mean": "g4b2_margin"}).select("game_id", "g4b2_margin", "sim_spread"), on="game_id", how="left")
    return sch.with_columns(cluster=pl.col("season") * 100 + pl.col("week"))


def implied(ml):
    return pl.when(ml < 0).then(-ml / (-ml + 100.0)).otherwise(100.0 / (ml + 100.0))


def s1(d: pl.DataFrame) -> list:
    rows = []
    for fc, actual, line in (("p3_spread", "result", "spread_line"), ("g4b2_margin", "result", "spread_line"), ("sim_spread", "result", "spread_line"), ("p3_total", "total", "total_line")):
        x = d.drop_nulls([fc, actual, line])
        diff = ((x[line] - x[actual]).abs() - (x[fc] - x[actual]).abs()).to_numpy()                 # positive = the model beat the line
        lo, hi = compare.cluster_bootstrap(diff, x["cluster"].to_numpy())
        won = int(sum(diff[(x["season"] == s).to_numpy()].mean() > 0 for s in x["season"].unique()))
        rows.append(dict(score="S1_mae", forecast=fc, n=x.height, model_mae=float((x[fc] - x[actual]).abs().mean()), line_mae=float((x[line] - x[actual]).abs().mean()),
                         gain=float(diff.mean()), ci_lo=lo, ci_hi=hi, seasons_won=won, verdict=compare.verdict(float(diff.mean()), lo, hi, won)))
    return rows


def s2(d: pl.DataFrame) -> list:
    rows = []
    for fc, actual, line in (("p3_spread", "result", "spread_line"), ("g4b2_margin", "result", "spread_line"), ("sim_spread", "result", "spread_line"), ("p3_total", "total", "total_line")):
        x = d.drop_nulls([fc, actual, line])
        gap = (x[fc] - x[line]).to_numpy()                                                          # model above the line: home / over side
        beat = np.sign(x[actual].to_numpy() - x[line].to_numpy())                                   # +1 home/over covered, -1 other side, 0 push
        for t in THRESHOLDS:
            take = np.abs(gap) >= t
            side = np.sign(gap)
            ok = take & (beat != 0)
            hit = (beat[ok] == side[ok]).astype(float)
            if len(hit) < 50:
                rows.append(dict(score="S2_hit", forecast=fc, threshold=t, n=int(len(hit)), verdict="too few")); continue
            cl, ss = x["cluster"].to_numpy()[ok], x["season"].to_numpy()[ok]
            lo, hi = compare.cluster_bootstrap(hit, cl, level=LEVEL)
            per = {int(s): float(hit[ss == s].mean()) for s in np.unique(ss)}
            above = int(sum(v > BREAKEVEN for v in per.values()))
            rows.append(dict(score="S2_hit", forecast=fc, threshold=t, n=int(len(hit)), hit_rate=float(hit.mean()), ci_lo=lo, ci_hi=hi, seasons_above_breakeven=above,
                             per_season=str({k: round(v, 3) for k, v in per.items()}), verdict="EDGE" if (lo > BREAKEVEN and above >= 2) else "no edge"))
    return rows


def s3(d: pl.DataFrame) -> list:
    x = d.drop_nulls(["p3_home_win", "home_moneyline", "away_moneyline", "result"]).filter(pl.col("result") != 0)
    x = x.with_columns(ph=implied(pl.col("home_moneyline").cast(pl.Float64)), pa=implied(pl.col("away_moneyline").cast(pl.Float64)))
    line_p = (x["ph"] / (x["ph"] + x["pa"])).to_numpy()
    y = (x["result"] > 0).cast(pl.Float64).to_numpy()
    mp = np.clip(x["p3_home_win"].to_numpy(), 1e-6, 1 - 1e-6)
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    diff = ll(np.clip(line_p, 1e-6, 1 - 1e-6)) - ll(mp)                                             # positive = the model beat the line
    lo, hi = compare.cluster_bootstrap(diff, x["cluster"].to_numpy())
    won = int(sum(diff[(x["season"] == s).to_numpy()].mean() > 0 for s in x["season"].unique()))
    return [dict(score="S3_logloss", forecast="p3_home_win", n=x.height, model_mae=float(ll(mp).mean()), line_mae=float(ll(np.clip(line_p, 1e-6, 1 - 1e-6)).mean()), gain=float(diff.mean()),
                 ci_lo=lo, ci_hi=hi, seasons_won=won, verdict=compare.verdict(float(diff.mean()), lo, hi, won))]


def main():
    d = load()
    out = pl.concat([pl.DataFrame(r, infer_schema_length=None) for r in (s1(d) + s2(d) + s3(d))], how="diagonal_relaxed")
    out.write_parquet(config.PROCESSED_DIR / "lines_comparison_games.parquet")
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
    print("games", d.height, "S2 level", round(LEVEL, 5), "break-even", round(BREAKEVEN, 4))
    print(out)


if __name__ == "__main__":
    main()

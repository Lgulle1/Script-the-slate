"""Workstream C: does feeding the book's game total and spread to the Phase 3 models improve them?  (lines_ablation_results_<pool version>.parquet)

  python run_lines_ablation.py [--tag TAG]

The gate was fixed in docs/overnight_plan.md ("Improvement plan from weeks 1-5", C) before this was built. Walk-forward 2020-2024, 2025 never loaded, WITH vs WITHOUT (the committed Phase 3 run).
Per market the gain with the 95% season-week cluster interval; layer_weight is 1.0 only on CLEARS (config.LAYER_WEIGHT_RULE), else 0.0 and flagged. A null control (the schedule's lines shuffled among the games of the same
season and week, home perspective kept) is reported, not gated. Game markets are reference only: with the book's total and spread as inputs they largely copy the book. This produces the "informed" stack only; the pure path never reads a line.
"""
import argparse
import json

import lightgbm
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from features import line_features as LF
from features import volume_features as vf
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint
from run_comps_ablation import _paired, gain_stats
from eval import compare

PREDICTIONS_PATH = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
WITHOUT = "vol_x_eff"
NULL_SEED = 20261014
PLAYER = list(bl.PLAYER_MARKETS)
REFERENCE = list(bl.GAME_MARKETS)


def results_name(tag=None) -> str:
    return f"lines_ablation_results_{config.ELIGIBLE_PLAYER_RULE['version']}" + (f"_{tag}" if tag else "")


def shuffled_schedule(sched: pl.DataFrame, games: pl.DataFrame, seed: int = NULL_SEED) -> pl.DataFrame:
    """spread_line and total_line permuted among the games of the same (season, week), kept together; no information, same distribution."""
    g = games.select("game_id", "season", "week").unique().join(sched, on="game_id", how="inner").sort("season", "week", "game_id")
    grp = g.select(pl.struct("season", "week").rank("dense")).to_series().to_numpy()
    rng = np.random.default_rng(seed)
    perm = np.arange(g.height)
    for k in np.unique(grp):
        idx = np.flatnonzero(grp == k)
        perm[idx] = idx[rng.permutation(len(idx))]
    vals = g.select("spread_line", "total_line")[perm]
    return g.select("game_id", "home_team", "away_team").hstack(vals)


def run_with(data, feats: pl.DataFrame, name: str) -> pl.DataFrame:
    vt = LF.attach(vf.load_feature_tables(data), feats)
    et = LF.attach(efficiency.load_efficiency_tables(data), feats)
    pair = combine.combined_predictors(volume.volume_predictors(vt), efficiency.efficiency_predictors(et))
    return bt.walk_forward(name, data, *pair)


def table(without: pl.DataFrame, withs: dict) -> pl.DataFrame:
    rows = []
    for market in PLAYER + REFERENCE:
        row = dict(market=market, metric="brier" if market == "moneyline" else "mae", reference_only=market in REFERENCE)
        for name, w in withs.items():
            st = gain_stats(market, _paired(market, without, w))
            row.update({f"{k}_{name}": v for k, v in st.items()})
        row["layer_weight"] = 0.0 if market in REFERENCE else compare.layer_weight(row["verdict_lines"])
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col(pl.Null).cast(pl.Float64))


def run(tag=None, save=True):
    data = bt.load_backtest_data()
    sched = LF.load_schedule_lines()
    games = data.team_log.select("game_id", "season", "week").unique()
    assert games["season"].is_in(config.BACKTEST_SEASONS).all()
    without = pl.read_parquet(PREDICTIONS_PATH).filter(pl.col("method") == WITHOUT)
    withs = {"lines": run_with(data, LF.features(sched), f"{WITHOUT}_lines"),
             "null": run_with(data, LF.features(shuffled_schedule(sched, games)), f"{WITHOUT}_lines_null")}
    t = table(without, withs)
    fp = fingerprint(data)
    counts = dict(games=games.height, with_lines=int(sched.drop_nulls(["spread_line", "total_line"]).height))
    if save:
        config.ensure_data_dirs()
        arrow = t.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(), b"lightgbm": lightgbm.__version__.encode(),
            b"bootstrap": json.dumps(dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"line_columns": json.dumps(LF.LN_COLUMNS).encode(), b"counts": json.dumps(counts).encode(), b"layer_weight_rule": config.LAYER_WEIGHT_RULE.encode()})
        pq.write_table(arrow, config.ROOT / f"{results_name(tag)}.parquet")
        pl.concat([w.with_columns(variant=pl.lit(k)) for k, w in withs.items()]).sort("variant", "market", "season", "week", "entity").write_parquet(
            config.PROCESSED_DIR / f"lines_ablation_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}" + (f"_{tag}" if tag else "") + ".parquet")
    return t, fp, counts


def print_summary(t, fp, counts):
    print(f"\nLines ablation (informed stack), {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward  fingerprint {fp}  games {counts['games']}, with lines {counts['with_lines']}\n")
    r = lambda c, k=4: pl.col(c).round(k)
    with pl.Config(tbl_rows=20, tbl_cols=40, tbl_width_chars=300, tbl_hide_dataframe_shape=True):
        print(t.select("market", "reference_only", n="n_lines", gain=r("gain_lines"), pct=r("gain_pct_lines", 2), ci=pl.format("[{}, {}]", r("ci_lo_lines"), r("ci_hi_lines")), won="seasons_won_lines", verdict="verdict_lines",
                       null_gain=r("gain_null"), null_verdict="verdict_null", layer_weight="layer_weight"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    table_, fp_, counts_ = run(a.tag)
    print_summary(table_, fp_, counts_)

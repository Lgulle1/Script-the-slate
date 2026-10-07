"""Run all five baselines through the walk-forward harness for all 11 markets over 2020-2024,
grade them, and save the results.

  python run_baseline_backtest.py

Writes baseline_results.parquet (graded long table, tracked in git; the data fingerprint is in
its metadata) and data/processed/baseline_predictions.parquet (every prediction/actual, git-ignored).
Running it twice on the same data gives identical numbers. The 2025 holdout is never loaded.
"""
import hashlib
import json

import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from eval import grade

RESULTS_PATH = config.ROOT / "baseline_results.parquet"
PREDICTIONS_PATH = config.PROCESSED_DIR / "baseline_predictions.parquet"
PLAYER_MARKET_ORDER = list(bl.PLAYER_MARKETS)
MARKET_ORDER = PLAYER_MARKET_ORDER + list(bl.GAME_MARKETS)


def fingerprint(data: bt.BacktestData) -> str:
    """Hash of the exact input rows, so two runs can be shown to have used identical data."""
    h = hashlib.sha256()
    for df in (data.player_log.sort("player_id", "game_id"), data.team_log.sort("team", "game_id")):
        h.update(df.hash_rows(seed=0).to_numpy().tobytes())
    return h.hexdigest()[:16]


def run(save=True):
    data = bt.load_backtest_data()
    preds = []
    for method in bl.METHODS:
        player, game = bt.baseline_predictors(method)
        preds.append(bt.walk_forward(method, data, player, game))
    preds = pl.concat(preds).sort("method", "market", "season", "week", "entity")
    results = grade.grade_backtest(preds)
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        preds.write_parquet(PREDICTIONS_PATH)
        table = results.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(),
            b"skill_reference": config.SKILL_REFERENCE_METHOD.encode()})
        pq.write_table(table, RESULTS_PATH)
    return results, preds, fp


def _pivot(results, metric, key, fmt):
    t = (results.filter((pl.col("metric") == metric) & (pl.col("key") == key))
         .pivot(on="method", index="market", values="value"))
    t = t.with_columns(pl.col("market").replace_strict({m: i for i, m in enumerate(MARKET_ORDER)}).alias("_o")).sort("_o").drop("_o")
    return t.with_columns([pl.col(c).map_elements(lambda v: "-" if v is None else format(v, fmt), return_dtype=pl.String)
                           for c in t.columns if c != "market"]).select("market", *bl.METHODS)


def print_summary(results, fp):
    n = results.filter((pl.col("metric") == "mae") & (pl.col("method") == "last3"))["n"].to_list()
    print(f"\nBaseline backtest {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]}  "
          f"(holdout {config.HOLDOUT_SEASON} untouched)  data fingerprint {fp}")
    print("All methods graded on identical rows per market; skill is vs", config.SKILL_REFERENCE_METHOD)
    with pl.Config(tbl_rows=30, tbl_cols=10, fmt_str_lengths=20, tbl_hide_dataframe_shape=True):
        print("\nMean absolute error (point markets)");           print(_pivot(results, "mae", "", ".2f"))
        print("\nBrier score, mean over ladder rungs (lower is better)"); print(_pivot(results, "brier", "mean", ".4f"))
        print("\nLog loss, mean over rungs");                      print(_pivot(results, "logloss", "mean", ".4f"))
        print(f"\nBrier skill vs {config.SKILL_REFERENCE_METHOD}");  print(_pivot(results, "skill_brier", "mean", "+.3f"))
        print("\nWhole-curve (PIT) max deviation from flat, 0 = perfectly flat"); print(_pivot(results, "pit_max_dev", "", ".3f"))
        print("\nCoverage (share of target rows a method could predict)"); print(_pivot(results, "coverage", "", ".3f"))
    cal = results.filter((pl.col("method") == "recency") & (pl.col("market") == "rush_yds") & pl.col("metric").is_in(["calib_pred", "calib_actual"]))
    print("\nCalibration example (recency, rush_yds; all rungs pooled)")
    print(cal.pivot(on="metric", index=["key", "n"], values="value").sort("key"))


if __name__ == "__main__":
    res, _, fp = run()
    print_summary(res, fp)
    print(f"\nsaved {RESULTS_PATH.name} ({res.height} rows) and {PREDICTIONS_PATH.relative_to(config.ROOT)}")

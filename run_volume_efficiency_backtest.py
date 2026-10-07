"""Phase 3 go/no-go: volume x efficiency models vs the best phase-2 baseline, per market.

  python run_volume_efficiency_backtest.py

Runs the five baselines and the combined volume (3.1) x efficiency (3.2) model through the same
walk-forward harness for all 11 markets over 2020-2024 (2025 is never loaded), compares each market's
model with its best baseline on identical rows, bootstraps a 95% CI on the gain, and writes
volume_efficiency_results.parquet (long format; see `results_table`). Re-running on the same data and
library versions gives identical numbers.
"""
import json
from datetime import date

import lightgbm
import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from eval import compare, grade
from features import lineups as lu
from features import volume_features as vf
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint

RESULTS_PATH = config.ROOT / "volume_efficiency_results.parquet"
PREDICTIONS_PATH = config.PROCESSED_DIR / "volume_efficiency_predictions.parquet"
MODEL = "vol_x_eff"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
SCHEMA = {"table": pl.String, "market": pl.String, "method": pl.String, "season": pl.Int32, "metric": pl.String,
          "key": pl.String, "value": pl.Float64, "n": pl.Int64, "label": pl.String}


def run_models(data):
    """Combined volume x efficiency predictions through the harness."""
    vt = vf.load_feature_tables(data)
    et = efficiency.load_efficiency_tables(data)
    pair = combine.combined_predictors(volume.volume_predictors(vt), efficiency.efficiency_predictors(et))
    return bt.walk_forward(MODEL, data, *pair)


def run_baselines(data):
    out = []
    for method in bl.METHODS:
        out.append(bt.walk_forward(method, data, *bt.baseline_predictors(method)))
    return pl.concat(out)


def results_table(comparisons: dict, graded: pl.DataFrame) -> pl.DataFrame:
    """Long results: table in {summary, by_season, baseline_loss, grade}."""
    rows = []

    def add(table, market, metric, value, n=None, season=None, method=MODEL, key="", label=None):
        rows.append(dict(table=table, market=market, method=method, season=season, metric=metric, key=key,
                         value=None if value is None else float(value), n=n, label=label))

    for market, c in comparisons.items():
        s = c["summary"]
        for metric in ("n", "model_loss", "best_loss", "gain", "gain_pct", "ci_lo", "ci_hi", "ci_lo_iid", "ci_hi_iid",
                       "seasons_won", "seasons_total", "model_bias", "best_bias"):
            add("summary", market, metric, s[metric], n=s["n"])
        add("summary", market, "best_baseline", None, n=s["n"], label=s["best_baseline"])
        add("summary", market, "loss_metric", None, n=s["n"], label=s["metric"])
        add("summary", market, "verdict", None, n=s["n"], label=s["verdict"])
        for b, v in s["baseline_losses"].items():
            add("baseline_loss", market, "loss", v, n=s["n"], method=b)
        for b in c["by_season"]:
            for metric in ("model_loss", "best_loss", "gain", "gain_pct", "ci_lo", "ci_hi"):
                add("by_season", market, metric, b[metric], n=b["n"], season=b["season"])
    df = pl.DataFrame(rows, schema=SCHEMA)
    g = graded.with_columns(table=pl.lit("grade"), season=pl.lit(None, dtype=pl.Int32), label=pl.lit(None, dtype=pl.String)
                            ).select(list(SCHEMA)).cast(SCHEMA)
    return pl.concat([df, g]).sort("table", "market", "method", "season", "metric", "key", nulls_last=True)


def run(save=True):
    data = bt.load_backtest_data()
    base, model = run_baselines(data), run_models(data)
    comparisons = {m: compare.compare_market(m, base, model) for m in MARKET_ORDER}
    graded = grade.grade_backtest(pl.concat([base, model]))
    results = results_table(comparisons, graded)
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        pl.concat([base, model]).sort("method", "market", "season", "week", "entity").write_parquet(PREDICTIONS_PATH)
        table = results.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(),
            b"lightgbm": lightgbm.__version__.encode(), b"bootstrap": json.dumps(
                dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"lgbm_params": json.dumps(volume.PARAMS).encode()})
        pq.write_table(table, RESULTS_PATH)
    return comparisons, results, fp


def print_summary(comparisons, fp):
    print(f"\nVolume x efficiency vs best baseline, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward "
          f"(holdout {config.HOLDOUT_SEASON} untouched)  fingerprint {fp}")
    print("Gain = best-baseline loss - model loss (positive = model better); MAE, moneyline = Brier. CI = 95% bootstrap over "
          f"season-weeks ({config.BOOTSTRAP_RESAMPLES:,} resamples). Same rows for model and baselines.\n")
    rows = []
    for m in MARKET_ORDER:
        s = comparisons[m]["summary"]
        unit = "" if m != "moneyline" else "(Brier) "
        rows.append(dict(market=m, n=s["n"], best_baseline=s["best_baseline"], baseline=round(s["best_loss"], 4),
                         model=round(s["model_loss"], 4), gain=round(s["gain"], 4), gain_pct=f"{s['gain_pct']:+.1f}%",
                         ci95=f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]", seasons=f"{s['seasons_won']}/{s['seasons_total']}",
                         verdict=s["verdict"]))
    with pl.Config(tbl_rows=20, tbl_cols=12, fmt_str_lengths=24, tbl_hide_dataframe_shape=True, tbl_width_chars=200):
        print(pl.DataFrame(rows))
        print("\nGain by season (positive = model better):")
        grid = [{"market": m, **{str(b["season"]): round(b["gain"], 4) for b in comparisons[m]["by_season"]}} for m in MARKET_ORDER]
        print(pl.DataFrame(grid))
    flagged = [m for m in MARKET_ORDER if comparisons[m]["summary"]["verdict"] != "CLEARS"]
    print("\nNo overall average is reported on purpose: each market stands on its own.")
    print(f"Clears the gate: {[m for m in MARKET_ORDER if m not in flagged]}")
    print(f"EDGE (interval touches/crosses zero, or <{config.MIN_SEASONS_WON} seasons) -- flag, do not ship quietly: "
          f"{[m for m in flagged if comparisons[m]['summary']['verdict'] == 'EDGE']}")
    print(f"NO (no gain / worse): {[m for m in flagged if comparisons[m]['summary']['verdict'] == 'NO']}")
    print(f"(11 markets were tested; at 95% a couple of lucky intervals are expected by chance alone.)")


if __name__ == "__main__":
    comps, res, fp = run()
    print_summary(comps, fp)
    print(f"\nsaved {RESULTS_PATH.name} ({res.height} rows) and {PREDICTIONS_PATH.relative_to(config.ROOT)}")

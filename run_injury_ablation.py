"""4a.4: does the injury layer improve the Phase 3 models?  (injury_ablation_results_<pool version>.parquet)

  python run_injury_ablation.py [--recompute]

Runs the combined volume x efficiency model twice through the walk-forward harness over 2020-2024 (2025 never loaded): WITHOUT the
injury features (the Phase 3 model) and WITH them (expected_snap_share, p_out, P(early exit), the redistributed carry / target /
dropback / snap shares and their change from baseline, and the team's QB / skill / OL absence totals, own and opponent's -- see
features/injury_features.py). Same LightGBM hyperparameters, same cutoffs. The WITHOUT predictions come from
data/processed/volume_efficiency_predictions_<pool version>.parquet (the reproducible Phase 3 run); --recompute reruns them.

One row per market, judged on identical rows (both predictions present). Gain = loss without - loss with (positive = the injury
features help); loss is absolute error, Brier for moneyline. 95% interval: the 3.3 season-week cluster bootstrap (10,000 resamples,
fixed seed). Per-season gains and mean bias (prediction - actual) for both runs are in the same row. No average across markets.

layer_weight is fixed in advance (not tuned): 1.0 if gain > 0 and the layer wins at least 2 seasons, otherwise 0.0 -- the layer then
stays at zero weight for that market and the market is flagged, never dropped.
"""
import argparse
import json

import lightgbm
import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from eval import compare
from features import injury_features as inf
from features import volume_features as vf
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint

RESULTS_PATH = config.result_path("injury_ablation_results")
PREDICTIONS_PATH = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
WITHOUT, WITH = "vol_x_eff", "vol_x_eff_injury"
MIN_SEASONS_FOR_WEIGHT = 2


def run_with_injury(data, feats):
    vt = vf.load_feature_tables(data, injury=feats)
    et = efficiency.load_efficiency_tables(data, injury=feats)
    pair = combine.combined_predictors(volume.volume_predictors(vt), efficiency.efficiency_predictors(et))
    return bt.walk_forward(WITH, data, *pair)


def ablation_rows(without: pl.DataFrame, with_: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for market in MARKET_ORDER:
        a = without.filter(pl.col("market") == market).select(*compare.KEY, "actual", without=pl.col("prediction"))
        b = with_.filter(pl.col("market") == market).select(*compare.KEY, with_=pl.col("prediction"))
        t = a.join(b, on=compare.KEY, how="inner").drop_nulls(["without", "with_"]).sort(compare.KEY)
        y, p0, p1 = t["actual"].to_numpy(), t["without"].to_numpy(), t["with_"].to_numpy()
        seasons = t["season"].to_numpy()
        cluster = seasons * 100 + t["week"].to_numpy()
        d = compare._loss(market, p0, y) - compare._loss(market, p1, y)
        lo, hi = compare.cluster_bootstrap(d, cluster)
        row = dict(market=market, n=t.height, metric="brier" if market == "moneyline" else "mae",
                   loss_without=float(compare._loss(market, p0, y).mean()), loss_with=float(compare._loss(market, p1, y).mean()),
                   gain=float(d.mean()), gain_pct=float(100 * d.mean() / compare._loss(market, p0, y).mean()), ci_lo=lo, ci_hi=hi)
        wins = 0
        for s in sorted(set(seasons.tolist())):
            g = float(d[seasons == s].mean())
            row[f"gain_{s}"] = g
            wins += g > 0
        row["seasons_won"], row["seasons_total"] = wins, len(set(seasons.tolist()))
        row["verdict"] = compare.verdict(row["gain"], lo, hi, wins)
        row["bias_without"] = None if market == "moneyline" else float((p0 - y).mean())
        row["bias_with"] = None if market == "moneyline" else float((p1 - y).mean())
        row["layer_weight"] = 1.0 if (row["gain"] > 0 and wins >= MIN_SEASONS_FOR_WEIGHT) else 0.0
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def run(recompute=False, save=True):
    data = bt.load_backtest_data()
    feats = inf.build_injury_features(data)
    if recompute or not PREDICTIONS_PATH.exists():
        import run_volume_efficiency_backtest as rv
        without = rv.run_models(data)
    else:
        without = pl.read_parquet(PREDICTIONS_PATH).filter(pl.col("method") == WITHOUT)
    with_ = run_with_injury(data, feats)
    table = ablation_rows(without, with_)
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        arrow = table.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(), b"lightgbm": lightgbm.__version__.encode(),
            b"bootstrap": json.dumps(dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"injury_columns": json.dumps(dict(players=inf.PLAYER_COLUMNS, teams=inf.TEAM_COLUMNS)).encode(),
            b"layer_weight_rule": f"1.0 if gain > 0 and seasons_won >= {MIN_SEASONS_FOR_WEIGHT} else 0.0".encode()})
        pq.write_table(arrow, RESULTS_PATH)
        with_.sort("market", "season", "week", "entity").write_parquet(
            config.PROCESSED_DIR / f"injury_ablation_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet")
    return table, fp


def print_summary(t: pl.DataFrame, fp: str):
    print(f"\nInjury ablation, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward  fingerprint {fp}")
    print("Gain = loss WITHOUT injury features - loss WITH (positive = the injury layer helps); MAE, moneyline = Brier; "
          f"CI = 95% season-week bootstrap ({config.BOOTSTRAP_RESAMPLES:,} resamples).\n")
    r = lambda c, k=4: pl.col(c).round(k)
    seasons = [c for c in t.columns if c.startswith("gain_20")]
    with pl.Config(tbl_rows=20, tbl_cols=24, tbl_width_chars=250, tbl_hide_dataframe_shape=True, fmt_str_lengths=14):
        print(t.select("market", "n", without=r("loss_without"), with_inj=r("loss_with"), gain=r("gain"), gain_pct=r("gain_pct", 2),
                       ci=pl.format("[{}, {}]", r("ci_lo"), r("ci_hi")), won=pl.format("{}/{}", "seasons_won", "seasons_total"),
                       verdict="verdict", weight="layer_weight"))
        print("\nGain by season:")
        print(t.select("market", *[r(c) for c in seasons]))
        print("\nMean bias (prediction - actual):")
        print(t.select("market", r("bias_without"), r("bias_with")))
    off = t.filter(pl.col("layer_weight") == 0.0)["market"].to_list()
    print(f"\nlayer_weight 0 (no lift: flagged, kept in the build): {off}")
    print("No overall average is reported on purpose: each market stands on its own.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    table, fp = run(a.recompute)
    print_summary(table, fp)
    print(f"\nsaved {RESULTS_PATH.name} ({table.height} rows)")

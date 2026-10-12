"""4e: does the weather layer improve the Phase 3 models on weather degraded to forecast quality?  (weather_ablation_results_<pool version>.parquet)

  python run_weather_ablation.py [--tag TAG] [--results-dir DIR] [--pred-dir DIR]

The gate was fixed in docs/overnight_plan.md ("4e gate") before this was built. The backtest has only OBSERVED weather, so the model runs on weather
degraded to forecast quality three ways -- A1 bins, A2 noise (+/-4 mph, +/-4 F, precipitation flipped 0.2) and A2x2 (double) -- and on exact observed
weather for reference only (features/weather_features.py). Walk-forward over 2020-2024, 2025 never loaded, WITH vs WITHOUT (the committed Phase 3 run).
Per market and variant, the gain with the 3.3 season-week interval on the SEVERE rows (outdoor games with observed wind >= 15 mph or temperature
<= 32 F) and on the other rows. layer_weight is 1.0 only if the market CLEARS on the severe rows under A1 AND A2 AND A2x2, and is not significantly
worse (interval upper bound >= 0) on the other rows under any of them. The exact-weather result is reported, never gated; a null control (the A2
weather columns shuffled among the games of the same week) is reported, not gated.
"""
import argparse
import json
from pathlib import Path

import lightgbm
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from features import volume_features as vf
from features import weather_features as wf
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint
from run_comps_ablation import _paired, gain_stats

PREDICTIONS_PATH = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
WITHOUT = "vol_x_eff"
GATED = ("A1", "A2", "A2x2")
NULL_SEED = 20261013


def results_name(tag: str | None = None) -> str:
    return f"weather_ablation_results_{config.ELIGIBLE_PLAYER_RULE['version']}" + (f"_{tag}" if tag else "")


def predictions_name(tag: str | None = None) -> str:
    return f"weather_ablation_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}" + (f"_{tag}" if tag else "") + ".parquet"


def null_features(feats: pl.DataFrame, games: pl.DataFrame, seed: int = NULL_SEED) -> pl.DataFrame:
    """The weather columns shuffled among the games of the same season and week (a fixed permutation): the same values, no information.
    games: game_id, season, week."""
    f = games.select("game_id", "season", "week").join(feats, on="game_id", how="inner").sort("season", "week", "game_id")
    grp = f.select(pl.struct("season", "week").rank("dense")).to_series().to_numpy()
    rng = np.random.default_rng(seed)
    perm = np.arange(f.height)
    for gid in np.unique(grp):
        idx = np.flatnonzero(grp == gid)
        perm[idx] = idx[rng.permutation(len(idx))]
    return f.select("game_id").hstack(f.select(wf.WX_COLUMNS)[perm])


def assign_severity(pred: pl.DataFrame, data, severe: list) -> pl.DataFrame:
    """Each prediction row (kind, season, week, entity) with severe = its game is in `severe`: a player row through the player's game that week."""
    sev = set(severe)
    gid = data.player_log.select("player_id", "season", "week", "game_id").unique(["player_id", "season", "week"])
    pl_rows = pred.filter(pl.col("kind") != "game").join(gid, left_on=["entity", "season", "week"], right_on=["player_id", "season", "week"], how="left")
    gm_rows = pred.filter(pl.col("kind") == "game").with_columns(game_id=pl.col("entity"))
    return pl.concat([pl_rows, gm_rows], how="diagonal_relaxed").with_columns(severe=pl.col("game_id").is_in(list(sev)))


def layer_decision(stats: dict) -> tuple:
    """(layer_weight, reason). stats: variant -> {'severe': gain_stats, 'other': gain_stats} for the gated variants."""
    for v in GATED:
        if stats[v]["severe"]["verdict"] != "CLEARS":
            return 0.0, f"not CLEARS on severe-weather rows under {v} ({stats[v]['severe']['verdict']})"
    for v in GATED:
        hi = stats[v]["other"].get("ci_hi")
        if hi is not None and hi < 0:
            return 0.0, f"significantly worse on the other rows under {v}"
    return 1.0, "CLEARS under A1, A2 and A2x2, not worse elsewhere"


def run_with(data, feats: pl.DataFrame, name: str) -> pl.DataFrame:
    vt = vf.load_feature_tables(data, weather=feats)
    et = efficiency.load_efficiency_tables(data, weather=feats)
    pair = combine.combined_predictors(volume.volume_predictors(vt), efficiency.efficiency_predictors(et))
    return bt.walk_forward(name, data, *pair)


def ablation_rows(without: pl.DataFrame, withs: dict, data, severe: list) -> pl.DataFrame:
    """withs: variant name -> predictions (variants GATED + 'exact' + 'null_A2')."""
    sev_all = assign_severity(without.select("kind", "season", "week", "entity", "market").unique(), data, severe)
    rows = []
    for market in MARKET_ORDER:
        m = sev_all.filter(pl.col("market") == market).select("season", "week", "entity", "severe")
        stats, row = {}, dict(market=market, metric="brier" if market == "moneyline" else "mae")
        for name, w in withs.items():
            t = _paired(market, without, w).join(m, on=["season", "week", "entity"], how="left")
            sv, ot = gain_stats(market, t.filter(pl.col("severe"))), gain_stats(market, t.filter(~pl.col("severe")))
            stats[name] = {"severe": sv, "other": ot}
            for k, s in (("sev", sv), ("oth", ot)):
                row.update({f"n_{k}_{name}": s["n"], f"gain_{k}_{name}": s["gain"], f"ci_lo_{k}_{name}": s["ci_lo"], f"ci_hi_{k}_{name}": s["ci_hi"],
                            f"seasons_won_{k}_{name}": s["seasons_won"], f"verdict_{k}_{name}": s["verdict"]})
        row["layer_weight"], row["decision"] = layer_decision(stats)
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col(pl.Null).cast(pl.Float64))


def run(tag=None, results_dir=None, pred_dir=None, save=True):
    data = bt.load_backtest_data()
    wx = wf.load_observed()
    games = data.team_log.select("game_id", "season", "week").unique()
    assert games["season"].is_in(config.BACKTEST_SEASONS).all()
    without = pl.read_parquet(PREDICTIONS_PATH).filter(pl.col("method") == WITHOUT)
    feats = {mode: wf.features(wx, mode, games=games) for mode in wf.MODES}
    withs = {mode: run_with(data, feats[mode], f"{WITHOUT}_wx_{mode}") for mode in wf.MODES}
    withs["null_A2"] = run_with(data, null_features(feats["A2"], games), f"{WITHOUT}_wx_null")
    severe = [g for g in wf.severe_games(wx) if g in set(games["game_id"].to_list())]
    table = ablation_rows(without, withs, data, severe)
    counts = dict(games=games.height, observed_rows=wx.height, indoor=int((wx["indoor"] == True).sum()), severe_games=len(severe))  # noqa: E712
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        arrow = table.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(), b"lightgbm": lightgbm.__version__.encode(),
            b"bootstrap": json.dumps(dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"weather_columns": json.dumps(wf.WX_COLUMNS).encode(), b"counts": json.dumps(counts).encode(),
            b"noise": json.dumps(dict(wind_sd=wf.A2_WIND_SD, temp_sd=wf.A2_TEMP_SD, precip_flip=wf.A2_PRECIP_FLIP, seed=wf.A2_SEED)).encode(),
            b"layer_weight_rule": b"1.0 only if CLEARS on severe rows under A1, A2 and A2x2 and not worse on the other rows; " + config.LAYER_WEIGHT_RULE.encode()})
        pq.write_table(arrow, (results_dir or config.ROOT) / f"{results_name(tag)}.parquet")
        pl.concat([w.with_columns(variant=pl.lit(k)) for k, w in withs.items()]).sort("variant", "market", "season", "week", "entity").write_parquet(
            (pred_dir or config.PROCESSED_DIR) / predictions_name(tag))
    return table, fp, counts


def print_summary(t: pl.DataFrame, fp: str, counts: dict):
    print(f"\n4e weather ablation, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward  fingerprint {fp}")
    print(f"games {counts['games']}, observed weather rows {counts['observed_rows']} (indoor {counts['indoor']}), severe-weather games {counts['severe_games']}\n")
    r = lambda c, k=4: pl.col(c).round(k)
    with pl.Config(tbl_rows=20, tbl_cols=40, tbl_width_chars=300, tbl_hide_dataframe_shape=True, fmt_str_lengths=44):
        for v in (*GATED, "exact", "null_A2"):
            print(f"-- {v}: severe rows")
            print(t.select("market", n=f"n_sev_{v}", gain=r(f"gain_sev_{v}"), ci=pl.format("[{}, {}]", r(f"ci_lo_sev_{v}"), r(f"ci_hi_sev_{v}")), verdict=f"verdict_sev_{v}",
                           other_gain=r(f"gain_oth_{v}"), other_hi=r(f"ci_hi_oth_{v}")))
        print(t.select("market", "layer_weight", "decision"))
    off = t.filter(pl.col("layer_weight") == 0.0)["market"].to_list()
    print(f"\nlayer_weight 0 (flagged, markets kept): {off}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--pred-dir", type=Path, default=None)
    a = ap.parse_args()
    table, fp, counts = run(a.tag, a.results_dir, a.pred_dir)
    print_summary(table, fp, counts)
    print(f"\nsaved {results_name(a.tag)}.parquet ({table.height} rows)")

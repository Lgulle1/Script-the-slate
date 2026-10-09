"""4c.6: does the comparable layer improve the Phase 3 models?  (comps_ablation_results_<pool version>.parquet)

  python run_comps_ablation.py [--window recency_weighted] [--recompute] [--results-dir DIR] [--pred-dir DIR]

Writes comps_ablation_results_<window>_<pool version>.parquet (repo root by default; the window is in the name, so the run with the windows chosen
per unit and market never overwrites this one) and the WITH predictions to data/processed.

Runs the combined volume x efficiency model through the walk-forward harness over 2020-2024 (2025 never loaded) WITH the comparable features of
4c.4 / 4c.5 (features/comps_features.py: per search the volume and efficiency shifts, n_eff, no-match flags, best similarity, the efficiency-side
n_eff / flag and the observed / derived / estimated shares and completeness) and WITHOUT them (the Phase 3 model). Same LightGBM hyperparameters,
same cutoffs. The WITHOUT predictions come from data/processed/volume_efficiency_predictions_<pool version>.parquet (the reproducible Phase 3 run);
--recompute reruns them. It also runs the model with ONE search's columns at a time (S1 only, ..., S5 only) to show which searches carry any lift.

One row per market, judged on identical rows (both predictions present). Gain = loss without - loss with (positive = the comparables help); loss is
absolute error, Brier for moneyline. 95% interval: the 3.3 season-week cluster bootstrap (10,000 resamples, fixed seed). Per-season gains, the
no-match rate and the mean n_eff (matched targets) of every search for the market, and the per-search gains are in the same row. No average across
markets. 2016-2019 games fill the comparable pool but are never scored.

Gate (build plan 4c, fixed in advance): the comparable features stay in a market's model only on CLEARS -- gain > 0, an interval that excludes
zero, and a win in at least config.MIN_SEASONS_WON seasons separately. layer_weight = 1.0 on CLEARS, otherwise 0.0: the features are left out of
that market's model for now and the market is flagged, never dropped. The run that decides the gate is the one on the windows chosen per unit and
market (--window selected, 4c.6.1); the recency_weighted run is a reference.

Null control (reported, not part of the gate): the same model with the comparable columns shuffled among the targets of the same market and week.
LightGBM samples 80% of the columns per tree, so adding any columns moves the predictions a little; the null gain shows how large a "gain" columns
without information produce.
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
from eval import compare
from features import comps_features as cf
from features import volume_features as vf
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint

def results_path(window: str, results_dir=None):
    p = config.result_path(f"comps_ablation_results_{window.replace(':', '_')}")
    return p if results_dir is None else results_dir / p.name


PREDICTIONS_PATH = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
WITHOUT, WITH = "vol_x_eff", "vol_x_eff_comps"
VARIANTS = {"all": cf.SEARCHES, **{s: (s,) for s in cf.SEARCHES}}
NULL_SEED = 20240901


def null_features(feats: pl.DataFrame, seed: int = NULL_SEED) -> pl.DataFrame:
    """The comparable columns shuffled among the rows of the same market, season and week (a fixed permutation): same values, no information."""
    keys = ["market", "season", "week", "game_id", "team", "player_id"]
    f = feats.sort(keys, nulls_last=True)
    cols = [c for c in f.columns if c not in keys and c != "opponent"]
    grp = f.select(pl.struct("market", "season", "week").rank("dense")).to_series().to_numpy()
    rng = np.random.default_rng(seed)
    perm = np.arange(f.height)
    for gid in np.unique(grp):
        idx = np.flatnonzero(grp == gid)
        perm[idx] = idx[rng.permutation(len(idx))]
    return f.select(keys + ["opponent"]).hstack(f.select(cols)[perm])


def run_with(data, feats: pl.DataFrame, searches: tuple, name: str) -> pl.DataFrame:
    vt = vf.load_feature_tables(data, comps=feats, comp_searches=searches)
    et = efficiency.load_efficiency_tables(data, comps=feats, comp_searches=searches)
    pair = combine.combined_predictors(volume.volume_predictors(vt), efficiency.efficiency_predictors(et))
    return bt.walk_forward(name, data, *pair)


def _paired(market: str, without: pl.DataFrame, with_: pl.DataFrame) -> pl.DataFrame:
    a = without.filter(pl.col("market") == market).select(*compare.KEY, "actual", without=pl.col("prediction"))
    b = with_.filter(pl.col("market") == market).select(*compare.KEY, with_=pl.col("prediction"))
    return a.join(b, on=compare.KEY, how="inner").drop_nulls(["without", "with_"]).sort(compare.KEY)


def gain_stats(market: str, t: pl.DataFrame) -> dict:
    """Gain of WITH over WITHOUT on the paired rows: mean, 95% cluster-bootstrap interval, per-season gains, seasons won, verdict."""
    if t.height == 0:
        return dict(n=0, loss_without=None, loss_with=None, gain=None, gain_pct=None, ci_lo=None, ci_hi=None, seasons_won=0, seasons_total=0, verdict="NO")
    y, p0, p1 = t["actual"].to_numpy(), t["without"].to_numpy(), t["with_"].to_numpy()
    seasons = t["season"].to_numpy()
    d = compare._loss(market, p0, y) - compare._loss(market, p1, y)
    lo, hi = compare.cluster_bootstrap(d, seasons * 100 + t["week"].to_numpy())
    base = float(compare._loss(market, p0, y).mean())
    out = dict(n=t.height, loss_without=base, loss_with=float(compare._loss(market, p1, y).mean()), gain=float(d.mean()),
               gain_pct=float(100 * d.mean() / base) if base else float("nan"), ci_lo=lo, ci_hi=hi)
    wins = 0
    for s in sorted(set(seasons.tolist())):
        g = float(d[seasons == s].mean())
        out[f"gain_{s}"] = g
        wins += g > 0
    out["seasons_won"], out["seasons_total"] = wins, len(set(seasons.tolist()))
    out["verdict"] = compare.verdict(out["gain"], lo, hi, wins)
    return out


def search_stats(feats: pl.DataFrame, market: str) -> dict:
    """The market's no-match rate and mean n_eff (over matched targets) per search, 2020-2024 targets."""
    f = feats.filter(pl.col("market") == market)
    out = {}
    for s in cf.SEARCHES:
        out[f"nomatch_rate_{s}"] = float(f[f"nomatch_{s}"].mean()) if f.height else None
        m = f.filter(~pl.col(f"nomatch_{s}"))
        out[f"mean_n_eff_{s}"] = float(m[f"n_eff_{s}"].mean()) if m.height else None
    return out


def ablation_rows(without: pl.DataFrame, with_by: dict, feats: pl.DataFrame, null: pl.DataFrame | None = None) -> pl.DataFrame:
    rows = []
    for market in MARKET_ORDER:
        st = gain_stats(market, _paired(market, without, with_by["all"]))
        row = dict(market=market, metric="brier" if market == "moneyline" else "mae", **st)
        row["layer_weight"] = compare.layer_weight(st["verdict"])          # the one rule for every layer (config.LAYER_WEIGHT_RULE)
        row.update(search_stats(feats, market))
        for s in cf.SEARCHES:                                # which searches carry the lift: the model with that search's columns only
            g = gain_stats(market, _paired(market, without, with_by[s]))
            row.update({f"gain_only_{s}": g["gain"], f"ci_lo_only_{s}": g["ci_lo"], f"ci_hi_only_{s}": g["ci_hi"], f"verdict_only_{s}": g["verdict"]})
        if null is not None:                                  # shuffled comparable columns: the size of a "gain" without information
            g = gain_stats(market, _paired(market, without, null))
            row.update(gain_null=g["gain"], ci_lo_null=g["ci_lo"], ci_hi_null=g["ci_hi"], verdict_null=g["verdict"])
        rows.append(row)
    t = pl.DataFrame(rows, infer_schema_length=None)
    return t.with_columns(pl.col(pl.Null).cast(pl.Float64))         # a statistic missing for every market (a search that never matches) stays numeric


def run(window: str = "recency_weighted", recompute: bool = False, save: bool = True, results_dir=None, pred_dir=None):
    data = bt.load_backtest_data()
    feats = cf.load(window)
    assert feats["season"].is_in(config.BACKTEST_SEASONS).all(), "comparable features outside the backtest seasons"
    if recompute or not PREDICTIONS_PATH.exists():
        import run_volume_efficiency_backtest as rv
        without = rv.run_models(data)
    else:
        without = pl.read_parquet(PREDICTIONS_PATH).filter(pl.col("method") == WITHOUT)
    with_by = {v: run_with(data, feats, s, f"{WITH}_{v}") for v, s in VARIANTS.items()}
    null = run_with(data, null_features(feats), cf.SEARCHES, f"{WITH}_null")
    table = ablation_rows(without, with_by, feats, null)
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        arrow = table.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(), b"lightgbm": lightgbm.__version__.encode(),
            b"comparable_window": window.encode(), b"comparable_columns": json.dumps(cf.columns()).encode(),
            b"window_choice": ((config.ROOT / "comp_windows.json").read_bytes() if window == "selected" else b"{}"),
            b"null_seed": str(NULL_SEED).encode(),
            b"bootstrap": json.dumps(dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"layer_weight_rule": b"1.0 on CLEARS (gain > 0, CI excludes 0, wins >= MIN_SEASONS_WON seasons) else 0.0"})
        pq.write_table(arrow, results_path(window, results_dir))
        pl.concat(list(with_by.values()) + [null]).sort("method", "market", "season", "week", "entity").write_parquet(
            (pred_dir or config.PROCESSED_DIR) / f"comps_ablation_predictions_{window.replace(':', '_')}_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet")
    return table, fp


def print_summary(t: pl.DataFrame, fp: str, window: str):
    print(f"\nComparables ablation ({window} window), {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward  fingerprint {fp}")
    print("Gain = loss WITHOUT comparable features - loss WITH (positive = they help); MAE, moneyline = Brier; "
          f"CI = 95% season-week bootstrap ({config.BOOTSTRAP_RESAMPLES:,} resamples).\n")
    r = lambda c, k=4: pl.col(c).round(k)
    seasons = [c for c in t.columns if c.startswith("gain_20")]
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=250, tbl_hide_dataframe_shape=True, fmt_str_lengths=14):
        print(t.select("market", "n", without=r("loss_without"), with_=r("loss_with"), gain=r("gain"), gain_pct=r("gain_pct", 2),
                       ci=pl.format("[{}, {}]", r("ci_lo"), r("ci_hi")), won=pl.format("{}/{}", "seasons_won", "seasons_total"),
                       verdict="verdict", weight="layer_weight"))
        print("\nGain by season:")
        print(t.select("market", *[r(c) for c in seasons]))
        print("\nGain with one search's columns only:")
        print(t.select("market", *[r(f"gain_only_{s}") for s in cf.SEARCHES], *[f"verdict_only_{s}" for s in cf.SEARCHES]))
        if "gain_null" in t.columns:
            print("\nNull control (comparable columns shuffled within market and week):")
            print(t.select("market", r("gain"), r("gain_null"), ci_null=pl.format("[{}, {}]", r("ci_lo_null"), r("ci_hi_null")), verdict_null="verdict_null"))
        print("\nNo-match rate and mean n_eff (matched targets) per search:")
        print(t.select("market", *[r(f"nomatch_rate_{s}", 3) for s in cf.SEARCHES], *[r(f"mean_n_eff_{s}", 2) for s in cf.SEARCHES]))
    off = t.filter(pl.col("layer_weight") == 0.0)["market"].to_list()
    print(f"\nlayer_weight 0 (not CLEARS: comparable features left out of these markets for now, flagged, markets kept): {off}")
    print("No overall average is reported on purpose: each market stands on its own.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="recency_weighted")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--pred-dir", type=Path, default=None)
    a = ap.parse_args()
    table, fp = run(a.window, a.recompute, results_dir=a.results_dir, pred_dir=a.pred_dir)
    print_summary(table, fp, a.window)
    print(f"\nsaved {results_path(a.window, a.results_dir).name} ({table.height} rows)")

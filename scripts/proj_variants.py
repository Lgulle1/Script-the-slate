"""Projection-model variants (yours, 2026-10-15): retrain the projection model itself (the volume x efficiency LightGBM models that predict each player's
attempts, catches and yards) under different settings, walk-forward over 2020-2024 exactly like the base model: every week is predicted by models refit on
games before that week only. Each variant writes data/processed/proj_variants/<name>_wf.parquet in the same layout as vol_x_eff_2024_walkforward.parquet,
so the line study, the picker and the betting tournament can be rerun on it (scripts/tournament4.py).

  python scripts/proj_variants.py                 # every variant not yet built, 4 at a time
  python scripts/proj_variants.py --only base l2
"""
import argparse
import sys
import time
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import backtest as bt
from features import volume_features as vf
from models import combine, efficiency, volume

OUTD = config.PROCESSED_DIR / "proj_variants"
P0 = volume.PARAMS

# name -> dict(params=..., window_years=None | n, half_life=None | games, layer=None | injury / weather / coach / comps)
VARIANTS = {
    "base": dict(),
    "l2_mean": dict(params=dict(P0, objective="l2")),
    "huber": dict(params=dict(P0, objective="huber", alpha=1.0)),
    "bigger_trees": dict(params=dict(P0, n_estimators=400, num_leaves=16, learning_rate=0.03)),
    "smaller_trees": dict(params=dict(P0, n_estimators=80, num_leaves=4)),
    "more_trees_slow": dict(params=dict(P0, n_estimators=300, learning_rate=0.025)),
    "min_leaf_100": dict(params=dict(P0, min_child_samples=100)),
    "min_leaf_10": dict(params=dict(P0, min_child_samples=10)),
    "window_2y": dict(window_years=2),
    "window_3y": dict(window_years=3),
    "half_life_3": dict(half_life=3),
    "half_life_12": dict(half_life=12),
    "injury": dict(layer="injury"),
    "weather": dict(layer="weather"),
    "coach": dict(layer="coach"),
    "comps": dict(layer="comps"),
}


def windowed(pair, years):
    """Wrap (player, game) predictors so models only learn from the last `years` years before the cutoff."""
    def wrap(f):
        if f is None:
            return None

        def g(history, targets, cutoff):
            start = cutoff - timedelta(days=365 * years)
            h = bt.History(history.player_log.filter(pl.col("gameday") >= start), history.team_log.filter(pl.col("gameday") >= start))
            return f(h, targets, cutoff)
        return g
    return tuple(wrap(f) for f in pair)


def layer_inputs(layer, data):
    if layer is None:
        return {}
    if layer == "injury":
        from features import injury_features as inf
        return dict(injury=inf.build_injury_features(data))
    if layer == "weather":
        from features import weather_features as wx
        games = data.team_log.select("game_id", "season", "week").unique()
        return dict(weather=wx.features(wx.load_observed(), "A1", games=games))
    if layer == "coach":
        from features import coach_prior as cp
        feats, _ = cp.build()
        return dict(coach=feats.filter(pl.col("season").is_in(config.BACKTEST_SEASONS)))
    if layer == "comps":
        from features import comps_features as cf
        return dict(comps=cf.load("recency_weighted"), comp_searches=cf.SEARCHES)
    raise ValueError(layer)


def run(name):
    try:
        return _run(name)
    except Exception as e:
        import traceback
        print(f"{name}: FAILED {e!r}\n{traceback.format_exc()}", flush=True)
        return None


def _run(name):
    v = VARIANTS[name]
    t0 = time.time()
    if v.get("half_life"):
        vf.HALF_LIFE = float(v["half_life"])
    data = bt.load_backtest_data()
    extra = layer_inputs(v.get("layer"), data)
    vt = vf.load_feature_tables(data, **extra)
    et = efficiency.load_efficiency_tables(data, **extra)
    params = v.get("params", P0)
    pair = combine.combined_predictors(volume.volume_predictors(vt, params=params), efficiency.efficiency_predictors(et, params=params))
    if v.get("window_years"):
        pair = windowed(pair, v["window_years"])
    m = bt.walk_forward("vol_x_eff", data, *pair)
    m.sort("method", "market", "season", "week", "entity").write_parquet(OUTD / f"{name}_wf.parquet")
    print(f"{name}: {m.height} rows in {time.time() - t0:.0f}s", flush=True)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    names = a.only or [n for n in VARIANTS if a.force or not (OUTD / f"{n}_wf.parquet").exists()]
    with Pool(4, maxtasksperchild=1) as pool:
        for n in pool.imap_unordered(run, names):
            pass


if __name__ == "__main__":
    main()

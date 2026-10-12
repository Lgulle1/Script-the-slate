"""Per-prediction attribution (live/explain.py): every prediction is a baseline plus exact per-feature contributions."""
import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from features.volume_features import FeatureTables
from live import explain as EX
from live import predict as LP


def _model(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 5))
    y = 3 + 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(scale=0.1, size=400)
    cols = [f"f{i}" for i in range(5)]
    import pandas as pd
    m = lgb.LGBMRegressor(n_estimators=40, num_leaves=8, min_child_samples=10, random_state=0, n_jobs=1, verbose=-1).fit(pd.DataFrame(X, columns=cols), y)
    return (m, cols), pl.DataFrame({c: X[:5, i] for i, c in enumerate(cols)} | {"game_id": ["g"] * 5, "player_id": [f"p{i}" for i in range(5)]})


def test_contributions_plus_baseline_equal_the_model_prediction_exactly():
    mc, df = _model()
    long = EX.contributions(mc, df, quantity="targets", keys=["game_id", "player_id"], bounds=(0.0, np.inf))
    pred = mc[0].predict(df.select(mc[1]).to_pandas())
    got = long.group_by("player_id").agg(raw=pl.col("contribution").sum(), component_prediction=pl.col("component_prediction").first()).sort("player_id")
    assert np.abs(got["raw"].to_numpy() - pred).max() < 1e-9
    assert np.allclose(got["component_prediction"].to_numpy(), np.maximum(pred, 0.0))
    assert (long.filter(pl.col("feature") == EX.BASELINE)["value"].is_null()).all() and long["feature"].n_unique() == 6


def test_the_top_drivers_are_the_largest_absolute_contributions_and_the_baseline_is_not_a_driver():
    mc, df = _model()
    long = EX.contributions(mc, df, quantity="targets", keys=["game_id", "player_id"], bounds=(0.0, np.inf))
    top = EX.top_drivers(long, n=2)
    one = long.filter(pl.col("player_id") == "p0").filter(pl.col("feature") != EX.BASELINE).with_columns(a=pl.col("contribution").abs()).sort("a", descending=True)
    assert top.filter(pl.col("player_id") == "p0")["feature"].to_list() == one["feature"].head(2).to_list() and EX.BASELINE not in top["feature"].to_list()


def test_a_market_is_the_product_of_its_components_and_the_map_covers_every_player_market():
    from models import combine
    assert set(EX.MARKET_COMPONENTS) == set(combine.PLAYER_MARKETS)
    assert EX.MARKET_COMPONENTS["rec_yds"] == ("targets", "catch_pct", "yds_per_rec") and EX.MARKET_COMPONENTS["pass_att"] == ("pass_att",)


def test_the_margin_decomposition_adds_up_to_the_modelled_margin():
    coef = dict(c=-0.065, x_rating=0.7, x_home=1.4, x_rest=0.03, x_qb=0.43)
    home = dict(team="LA", league_pts=23.0, x_rating=5.4, x_home=0.5, x_rest=0.0, x_qb=0.02)
    away = dict(team="BUF", league_pts=23.0, x_rating=2.3, x_home=-0.5, x_rest=0.0, x_qb=0.30)
    t = EX.margin_decomposition(home, away, coef)
    from models import game_state as gs
    margin, total = gs.game_prediction(home, away, coef)
    assert abs(t.filter(pl.col("term") == "margin")["points"][0] - margin) < 1e-12 and abs(t.filter(pl.col("term") == "total")["points"][0] - total) < 1e-12
    parts = t.filter(~pl.col("term").is_in(["margin", "total"]) & (pl.col("side") == "margin"))["points"].sum()
    assert abs(parts - margin) < 1e-12

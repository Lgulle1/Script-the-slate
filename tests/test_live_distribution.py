"""Attaching the predictive distribution to a run's predictions (live/distribution.py)."""
import json

import numpy as np
import polars as pl

from eval import calibrate as C
from live import distribution as D
from tests.test_calibrate import _cfg, _world


def _cal():
    return C.build(_world(n_weeks=14, per_week=200), _cfg(), markets=["targets"])


def _preds():
    return pl.DataFrame(dict(game_id=["g"] * 3, player_id=["a", "b", None], market=["targets", "targets", "moneyline"], mean=[6.0, 11.0, 0.55]), schema_overrides={"player_id": pl.String})


def test_the_distribution_is_attached_without_touching_the_mean_or_the_row_order():
    out = D.attach(_preds(), _cal())
    assert out["mean"].to_list() == [6.0, 11.0, 0.55] and out["player_id"].to_list() == ["a", "b", None]
    q = out.select("q05", "q10", "q25", "q50", "q75", "q90", "q95").row(0)
    assert list(q) == sorted(q) and out["median"][0] == out["q50"][0]
    lad = json.loads(out["ladder"][0])
    assert set(lad) == {"2", "4", "6", "8", "10"} and lad["4"] >= lad["8"]


def test_a_market_without_a_distribution_keeps_null_quantiles_and_a_null_ladder():
    out = D.attach(_preds(), _cal())
    assert out.row(2, named=True)["q50"] is None and out.row(2, named=True)["ladder"] is None
    plain = D.attach(_preds(), None)
    assert plain["q05"].null_count() == 3 and plain["ladder"].null_count() == 3


def test_a_higher_prediction_gets_a_higher_median():
    out = D.attach(_preds(), _cal())
    assert out["q50"][1] > out["q50"][0]

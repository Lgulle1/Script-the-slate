import math

import numpy as np
import polars as pl
import pytest

import config
from eval import grade as g


def test_mae_brier_logloss_known_values():
    assert g.mean_absolute_error([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert g.brier_score([0.9, 0.2], [1, 0]) == pytest.approx((0.01 + 0.04) / 2)
    assert g.log_loss([0.8, 0.5], [1, 0]) == pytest.approx(-(math.log(0.8) + math.log(0.5)) / 2)
    assert g.log_loss([1.0], [0]) < 20  # clipped, finite


def test_skill_vs_baseline():
    assert g.skill_vs_baseline(0.20, 0.25) == pytest.approx(0.2)
    assert g.skill_vs_baseline(0.25, 0.25) == 0
    assert g.skill_vs_baseline(0.30, 0.25) < 0


def test_calibration_bands_and_edges():
    t = g.calibration_table([0.05, 0.15, 0.5, 0.95, 1.0], [0, 1, 1, 1, 1])
    assert t["band"].to_list() == ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    assert t["n"].to_list() == [2, 0, 1, 0, 2]
    assert t["actual_frequency"].to_list() == [0.5, None, 1.0, None, 1.0]


def test_calibrated_forecaster_is_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    o = (rng.uniform(0, 1, p.size) < p).astype(float)
    t = g.calibration_table(p, o)
    assert (t["actual_frequency"] - t["mean_predicted"]).abs().max() < 0.02


def test_pit_flatness_detects_flat_and_not_flat():
    rng = np.random.default_rng(1)
    flat = g.pit_flatness(rng.uniform(size=20_000))
    assert flat["max_dev"] < 0.02 and flat["ks"] < 0.02
    skew = g.pit_flatness(rng.beta(0.4, 0.4, size=20_000))  # U-shaped = overconfident
    assert skew["max_dev"] > 0.05


def _synthetic(seed=0, n_weeks=20, per_week=60, sigma=5.0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        for i in range(per_week):
            p = rng.uniform(40, 80)
            rows.append(dict(season=2024, week=w, entity=f"e{w}_{i}", prediction=p,
                             actual=float(round(p + rng.normal(0, sigma)))))
    return pl.DataFrame(rows)


def test_predictive_distribution_is_flat_for_a_correctly_specified_baseline():
    d = _synthetic()
    s = g.predictive_scores(d, [49.5, 59.5, 69.5])
    assert s.height > 600
    assert g.pit_flatness(s["pit"].to_numpy())["max_dev"] < 0.04


def test_predictive_distribution_uses_only_earlier_weeks():
    d = _synthetic()
    base = g.predictive_scores(d, [59.5])
    # wreck every actual from week 10 onward: rows before week 10 must be unchanged
    wrecked = d.with_columns(actual=pl.when(pl.col("week") >= 10).then(pl.col("actual") + 500).otherwise(pl.col("actual")))
    w = g.predictive_scores(wrecked, [59.5])
    early = lambda df: df.filter(pl.col("week") <= 10).sort("week", "entity")  # noqa: E731  week 10 itself is predicted from weeks <10
    assert early(base)["p_over_59.5"].to_list() == early(w)["p_over_59.5"].to_list()
    # and week 10's PIT/probabilities depend on week 10's own actual only through the PIT, never the pool
    assert early(base)["pit"].to_list()[:10] == early(w)["pit"].to_list()[:10]


def test_no_distribution_before_enough_residuals_and_determinism():
    d = _synthetic(n_weeks=3, per_week=20)  # 60 rows < MIN_RESIDUALS
    assert g.predictive_scores(d, [59.5]).height == 0
    d = _synthetic()
    assert g.predictive_scores(d, [59.5]).equals(g.predictive_scores(d, [59.5]))


def test_probabilities_are_proper_and_monotone_in_rung():
    s = g.predictive_scores(_synthetic(), [49.5, 59.5, 69.5])
    assert s.select(pl.min_horizontal(pl.col("^p_over_.*$")).min()).item() > 0
    assert s.select(pl.max_horizontal(pl.col("^p_over_.*$")).max()).item() < 1
    assert (s["p_over_49.5"] >= s["p_over_59.5"]).all() and (s["p_over_59.5"] >= s["p_over_69.5"]).all()


def test_moneyline_scores_clip():
    m = g.moneyline_scores(pl.DataFrame({"prediction": [0.0, 0.5, None], "actual": [1.0, 0.0, 1.0]}))
    assert m.height == 2 and m["p_over_win"].min() > 0

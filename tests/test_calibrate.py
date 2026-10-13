"""Predictive distribution, width fix and isotonic correction (eval/calibrate.py): rules of docs/overnight_plan.md, "Prediction schedule, calibration and grading"."""
import numpy as np
import polars as pl
import pytest

from eval import calibrate as C


def _world(n_weeks=14, per_week=120, seed=0, noise=3.0, market="targets"):
    """Synthetic walk-forward rows: actual = prediction + noise (a calibrated world)."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        p = rng.uniform(2, 14, per_week)
        a = p + rng.normal(0, noise, per_week)
        rows.append(pl.DataFrame(dict(market=market, season=2021 + (w - 1) // 7, week=(w - 1) % 7 + 1, entity=[f"e{w}_{i}" for i in range(per_week)], prediction=p, actual=a)))
    return pl.concat(rows)


def _cfg(**kw):
    return C.Config(k={"targets": 100}, min_pool={"targets": 300}, rungs={"targets": (2, 4, 6, 8, 10)}, **kw)


def test_a_rows_distribution_uses_only_strictly_earlier_weeks():
    rows = _world()
    cfg = _cfg()
    base = C.walk_forward(rows, "targets", cfg)
    changed = rows.with_columns(actual=pl.when((pl.col("season") == 2022) & (pl.col("week") >= 4)).then(pl.col("actual") + 50).otherwise(pl.col("actual")))
    again = C.walk_forward(changed, "targets", cfg)
    early = lambda d: d.filter((pl.col("season") == 2021) | ((pl.col("season") == 2022) & (pl.col("week") <= 4))).sort("entity")
    assert early(base).select(["q05", "q50", "q95", "p_ge_4"]).equals(early(again).select(["q05", "q50", "q95", "p_ge_4"]))   # week 4 itself and everything before are unchanged
    late = base.filter((pl.col("season") == 2022) & (pl.col("week") == 5)).sort("entity")["q95"].to_numpy()
    assert not np.allclose(late, again.filter((pl.col("season") == 2022) & (pl.col("week") == 5)).sort("entity")["q95"].to_numpy())


def test_burn_in_gives_no_distribution_not_a_made_up_one():
    rows = _world(n_weeks=3, per_week=50)
    out = C.walk_forward(rows, "targets", _cfg())
    assert out.height == 0 or out["season"].min() >= 2021           # the first weeks have a pool below min_pool: no rows
    assert C.walk_forward(rows.filter(pl.col("week") == 1), "targets", _cfg()).height == 0


def test_quantiles_are_ordered_the_ladder_is_decreasing_and_a_calibrated_world_covers_about_as_stated():
    out = C.walk_forward(_world(n_weeks=14, per_week=200), "targets", _cfg())
    q = out.select("q05", "q10", "q25", "q50", "q75", "q90", "q95").to_numpy()
    assert (np.diff(q, axis=1) >= -1e-9).all()
    ladder = out.select([f"p_ge_{t}" for t in (2, 4, 6, 8, 10)]).to_numpy()
    assert (np.diff(ladder, axis=1) <= 1e-9).all()
    cov = C.coverage(out, c=1.0)
    assert abs(cov[0.5] - 0.5) < 0.08 and abs(cov[0.8] - 0.8) < 0.08 and abs(cov[0.9] - 0.9) < 0.08


def test_the_width_fix_narrows_intervals_that_are_too_wide_and_widens_ones_that_are_too_narrow():
    out = C.walk_forward(_world(n_weeks=14, per_week=200), "targets", _cfg())
    wide = out.with_columns([(pl.col("q50") + 1.35 * (pl.col(c) - pl.col("q50"))).alias(c) for c in ("q05", "q10", "q25", "q75", "q90", "q95")])
    narrow = out.with_columns([(pl.col("q50") + 0.75 * (pl.col(c) - pl.col("q50"))).alias(c) for c in ("q05", "q10", "q25", "q75", "q90", "q95")])
    assert C.fit_width(wide) < 0.85 and C.fit_width(narrow) > 1.15
    assert abs(C.fit_width(out) - 1.0) < 0.15


def test_the_width_fix_is_applied_only_if_it_helps_in_at_least_two_of_three_folds():
    assert C.use_fix([0.10, 0.08, 0.20], [0.12, 0.09, 0.15]) is True        # helps in folds 1 and 2
    assert C.use_fix(with_fix=[0.05, 0.04, 0.09], without_fix=[0.12, 0.10, 0.08]) is True
    assert C.use_fix(with_fix=[0.05, 0.11, 0.09], without_fix=[0.12, 0.10, 0.08]) is False


def test_isotonic_correction_makes_a_distorted_probability_reliable_and_the_gate_rule_matches_the_width_rule():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p ** 2).astype(float)                  # the true probability is p squared: p is badly calibrated
    iso = C.fit_isotonic(p[:3000], y[:3000])
    assert C.brier(iso(p[3000:]), y[3000:]) < C.brier(p[3000:], y[3000:])


def test_the_calibrator_artifact_round_trips_has_a_stable_hash_and_never_returns_a_mean(tmp_path):
    rows = _world(n_weeks=14, per_week=200)
    cal = C.build(rows, _cfg(), markets=["targets"])
    h = cal.hash
    cal.save(tmp_path / "cal")
    again = C.Calibrator.load(tmp_path / "cal")
    assert again.hash == h
    d1, d2 = cal.distribution("targets", 6.0), again.distribution("targets", 6.0)
    assert d1 == d2 and set(d1) >= {"median", "q05", "q95", "ladder"} and "mean" not in d1
    assert cal.distribution("targets", 6.0)["ladder"]["4"] >= cal.distribution("targets", 6.0)["ladder"]["8"]
    assert C.build(rows, _cfg(), markets=["targets"]).hash == h


def test_a_market_without_enough_rows_has_no_distribution():
    cal = C.build(_world(n_weeks=2, per_week=40), _cfg(), markets=["targets"])
    assert cal.distribution("targets", 6.0) is None


def test_rows_from_the_holdout_season_are_refused():
    rows = _world().with_columns(season=pl.lit(2025))
    with pytest.raises(ValueError):
        C.build(rows, _cfg(), markets=["targets"])

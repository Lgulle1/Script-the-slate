"""The fingerprint diagnostic (models/comps_unit_signal.py)."""
import numpy as np
import polars as pl
import pytest

import config
from models import comps_spec as cs
from models import comps_unit_signal as US
from tests.test_comps import _pool


def test_headlines_are_the_first_spec_feature_of_each_team_unit():
    assert US.headline("run_defense") == "run_defense.rush_epa_allowed" and US.headline("pass_rush") == "pass_rush.sack_rate"
    assert US.headline("rb_rotation") == "rb_rotation.rb1_carry_share" and set(US.UNITS) == set(cs.OFFENSE_UNITS + cs.DEFENSE_UNITS)


def test_knn_takes_the_k_most_similar_with_an_outcome_and_weighs_by_similarity():
    sim = np.array([0.9, np.nan, 0.8, 0.95, 0.1, 0.8])
    y = np.array([1.0, 5.0, np.nan, 3.0, 9.0, 2.0])
    pred, cand, w = US.knn(sim, y, k=3)
    assert cand.tolist() == [3, 0, 5] and pred == pytest.approx((0.95 * 3 + 0.9 * 1 + 0.8 * 2) / (0.95 + 0.9 + 0.8))
    assert np.isnan(US.knn(np.full(3, np.nan), y[:3])[0])


def test_single_game_outcomes_divide_the_ledger_and_leave_an_empty_denominator_missing():
    cols = {}
    for u in US.UNITS:
        cols[f"{US.headline(u)}|n"] = [2.0, 1.0]
        cols[f"{US.headline(u)}|d"] = [4.0, 0.0]
    wide = pl.DataFrame({"game_id": ["g1", "g2"], "team": ["A", "A"], **cols})
    o = US.single_game_outcomes(wide)
    assert o["run_offense"][("g1", "A")] == 0.5 and np.isnan(o["pass_rush"][("g2", "A")])


def test_the_verdict_uses_the_bonferroni_level_and_needs_the_lower_bound_above_zero():
    rng = np.random.default_rng(0)
    rows = []
    for s in (2021, 2022, 2023, 2024):
        for wk in range(1, 18):
            for t in range(6):
                for u in US.UNITS:
                    good = u == "run_defense"
                    rows.append(dict(unit=u, season=s, week=wk, game_id=f"{s}{wk}{t}", team=f"T{t}", outcome=0.0, a=0.0, k=0.0,
                                     loss_a=1.0 + (0.3 if good else float(rng.normal(0, 0.3))), loss_k=1.0, loss_r=1.2, n_neighbours=20))
    v = US.verdicts(pl.DataFrame(rows))
    assert v.filter(pl.col("unit") == "run_defense")["passes"][0] and v["interval_level"][0] == pytest.approx(1 - 0.05 / 9)
    assert v.filter(pl.col("unit") != "run_defense")["passes"].sum() <= 1


def test_unit_rows_on_a_synthetic_league_are_deterministic_and_walk_forward():
    pool, inp, _ = _pool()
    rng = np.random.default_rng(2)
    outcomes = {u: {(r["game_id"], r["team"]): float(rng.normal()) for r in inp.games.iter_rows(named=True)} for u in ("run_offense", "run_defense")}
    a = US.unit_rows(pool, outcomes, units=("run_offense", "run_defense"), k=5, n_draws=2)
    b = US.unit_rows(pool, outcomes, units=("run_offense", "run_defense"), k=5, n_draws=2)
    assert a.equals(b) and a.height > 0 and set(a["season"].unique()) <= set(config.BACKTEST_SEASONS)
    # a later game's outcome never moves an earlier prediction
    late = {u: dict(d) for u, d in outcomes.items()}
    for (g, t) in list(late["run_offense"]):
        if g.startswith("2020_10"):
            late["run_offense"][(g, t)] = 99.0
    c = US.unit_rows(pool, late, units=("run_offense",), k=5, n_draws=2)
    early = lambda x: x.filter((pl.col("unit") == "run_offense") & (pl.col("week") < 10)).select("game_id", "team", "a", "k", "loss_r")
    assert early(a).equals(early(c))

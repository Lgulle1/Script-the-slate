"""The team-level sanity test (models/comps_team_test.py): the offense's own average, the matchup adjustment, the random control, the verdict."""
import numpy as np
import polars as pl
import pytest

import config
from models import comps_spec as cs
from models import comps_team_test as TT
from tests.test_comps import _pool


def test_recent_average_uses_only_earlier_weeks_of_the_same_team_with_recency_weights():
    key = np.array([201901, 201902, 201903, 201903, 201904])
    clock = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
    team = np.array(["A", "A", "A", "B", "A"])
    y = np.array([10.0, 20.0, 30.0, 99.0, 40.0])
    out = TT.recent_average(key, clock, team, y)
    assert np.isnan(out[0]) and out[1] == pytest.approx(10.0) and np.isnan(out[3])      # no earlier game of its own team
    h = config.RECENCY_HALF_LIFE_GAMES
    w = np.array([0.5 ** (2 / h), 0.5 ** (1 / h)])
    assert out[2] == pytest.approx((w * [10.0, 20.0]).sum() / w.sum())
    y2 = y.copy(); y2[4] = -500.0
    assert np.array_equal(TT.recent_average(key, clock, team, y2)[:4], out[:4], equal_nan=True)   # a later game changes nothing earlier


def test_adjustment_is_the_shrunk_weighted_mean_deviation():
    w, d = np.array([1.0, 1.0, 2.0]), np.array([3.0, -1.0, 2.0])
    n = 16 / 6
    assert TT.adjustment(w, d) == pytest.approx((3 - 1 + 4) / 4 * n / (n + cs.SHRINK_K))


def test_the_verdict_needs_both_gains_with_intervals_above_zero():
    rng = np.random.default_rng(0)
    rows = []
    for s in (2021, 2022, 2023):
        for wk in range(1, 18):
            for t in range(4):
                y = float(rng.normal(100, 30))
                for market in TT.MARKETS:
                    good = market == "team_rush_yds"               # (b) close to y in one market, no better than (a) in the other
                    rows.append(dict(market=market, season=s, week=wk, game_id=f"{s}{wk}{t}", team=f"T{t}", actual=y, a=y + 20, b=y + (5 if good else 20),
                                     loss_a=20.0, loss_b=5.0 if good else 20.0, loss_b_random=19.0 if good else 20.0, matched=True, n_eff=3.0))
    v = TT.verdicts(pl.DataFrame(rows))
    p = dict(zip((v["market"] + "|" + v["rows"]).to_list(), v["passes"].to_list()))
    assert p["team_rush_yds|matched"] is True and p["team_pass_att|matched"] is False and p["team_rush_yds|all"] is False


def test_test_rows_on_a_synthetic_league_are_deterministic_and_fall_back_to_a_without_a_match():
    pool, inp, _ = _pool()
    rng = np.random.default_rng(4)
    out = inp.games.select("game_id", "team").with_columns(rush_yds=pl.Series(rng.normal(110, 30, inp.games.height)),
                                                           pass_att=pl.Series(rng.normal(34, 6, inp.games.height)))
    a = TT.test_rows(pool, out, threshold=0.3, n_draws=3)
    b = TT.test_rows(pool, out, threshold=0.3, n_draws=3)
    assert a.equals(b) and a.height > 0 and set(a["season"].unique()) <= set(config.BACKTEST_SEASONS)
    um = a.filter(~pl.col("matched"))
    assert (um["b"] == um["a"]).all() and (um["loss_b_random"] == um["loss_a"]).all()
    if a["matched"].any():
        m = a.filter(pl.col("matched"))
        assert (m["n_eff"] >= cs.MIN_NEFF - 1e-9).all()

"""4c.6 ablation report logic (run_comps_ablation.py): paired gains, the CLEARS gate and the per-search statistics, on synthetic predictions."""
import numpy as np
import polars as pl
import pytest

import config
import run_comps_ablation as R
from features import comps_features as cf


def _preds(method, market, shift, seasons=(2021, 2022, 2023), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in range(1, 9):
            for e in range(10):
                y = float(rng.normal(50, 10))
                rows.append(dict(method=method, market=market, season=s, week=w, entity=f"E{e}", actual=y, prediction=y + 5.0 - shift))
    return pl.DataFrame(rows)


def test_a_clear_gain_in_every_season_clears_and_gets_full_weight():
    t = R._paired("rush_yds", _preds("a", "rush_yds", 0.0), _preds("b", "rush_yds", 2.0))
    g = R.gain_stats("rush_yds", t)
    assert g["gain"] == pytest.approx(2.0) and g["ci_lo"] > 0 and g["seasons_won"] == 3 and g["verdict"] == "CLEARS"


def test_no_gain_is_no_and_an_empty_market_does_not_crash():
    t = R._paired("rush_yds", _preds("a", "rush_yds", 0.0), _preds("b", "rush_yds", 0.0))
    assert R.gain_stats("rush_yds", t)["verdict"] == "NO"
    assert R.gain_stats("rush_yds", t.head(0))["verdict"] == "NO"


def test_one_winning_season_is_not_enough():
    good = _preds("b", "rush_yds", 2.0, seasons=(2021,))
    flat = _preds("b", "rush_yds", 0.0, seasons=(2022, 2023), seed=1)
    without = pl.concat([_preds("a", "rush_yds", 0.0, seasons=(2021,)), _preds("a", "rush_yds", 0.0, seasons=(2022, 2023), seed=1)])
    g = R.gain_stats("rush_yds", R._paired("rush_yds", without, pl.concat([good, flat])))
    assert g["seasons_won"] == 1 and g["verdict"] != "CLEARS" and config.MIN_SEASONS_WON >= 2


def test_search_stats_read_the_markets_feature_rows():
    rows = []
    for i in range(4):
        r = dict(market="rush_att", season=2022)
        for s in cf.SEARCHES:
            r.update({f"nomatch_{s}": (i > 0) if s == "S1" else True, f"n_eff_{s}": 3.0 if (s == "S1" and i == 0) else 0.5})
        rows.append(r)
    st = R.search_stats(pl.DataFrame(rows), "rush_att")
    assert st["nomatch_rate_S1"] == pytest.approx(0.75) and st["mean_n_eff_S1"] == pytest.approx(3.0)
    assert st["nomatch_rate_S2"] == 1.0 and st["mean_n_eff_S2"] is None

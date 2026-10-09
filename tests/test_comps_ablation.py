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


def test_null_features_shuffle_within_market_and_week_and_keep_the_values():
    rows = []
    for m in ("rush_att", "targets"):
        for w in (1, 2):
            for i in range(6):
                rows.append(dict(market=m, season=2022, week=w, game_id=f"G{w}", team="A", opponent="B", player_id=f"P{i}", shift_vol_S1=float(100 * w + i)))
    f = pl.DataFrame(rows)
    n = R.null_features(f)
    assert n.height == f.height and n.columns == f.select(["market", "season", "week", "game_id", "team", "player_id", "opponent", "shift_vol_S1"]).columns
    for (m, w), g in n.group_by("market", "week"):
        assert sorted(g["shift_vol_S1"].to_list()) == [float(100 * w + i) for i in range(6)]     # the same values, within the same market-week
    assert not n.sort("market", "week", "player_id")["shift_vol_S1"].equals(f.sort("market", "week", "player_id")["shift_vol_S1"])
    assert R.null_features(f).equals(n)                                                          # a fixed permutation


def test_the_report_prints_when_a_search_never_matches(capsys):
    """S2 / S4 never match at the plan's threshold: their mean n_eff is null for every market and must not break the report."""
    rows = []
    for i in range(4):
        r = dict(market="rush_yds", season=2022)
        for s in cf.SEARCHES:
            r.update({f"nomatch_{s}": True, f"n_eff_{s}": 0.0})
        rows.append(r)
    feats = pl.DataFrame(rows)
    w0 = pl.concat([_preds("vol_x_eff", m, 0.0) for m in R.MARKET_ORDER])
    by = {v: pl.concat([_preds(v, m, 0.5) for m in R.MARKET_ORDER]) for v in R.VARIANTS}
    t = R.ablation_rows(w0, by, feats, null=pl.concat([_preds("null", m, 0.0) for m in R.MARKET_ORDER]))
    assert t["mean_n_eff_S2"].dtype == pl.Float64
    R.print_summary(t, "fp", "recency_weighted")
    assert "Null control" in capsys.readouterr().out

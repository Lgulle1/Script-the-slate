"""scripts/price_check.py: break-even, the side choice and the profit arithmetic."""
import importlib.util
from pathlib import Path

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("pc", Path(__file__).resolve().parents[1] / "scripts" / "price_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def test_break_even_of_american_odds():
    be = pc.breakeven([-110, 100, -400, 150, -233])
    assert np.allclose(be, [110 / 210, 0.5, 0.8, 0.4, 233 / 333])


def test_a_won_bet_pays_its_odds_and_a_lost_one_loses_the_stake():
    e = pl.DataFrame(dict(game_id=["2026_01_A_B"] * 2, player_id=["p1", "p2"], market=["rec", "rec"], line=[3.5, 3.5], side=["over", "under"], price=[-200.0, 150.0], edge=[0.1, 0.1], model_chance=[0.7, 0.6]))
    act = pl.DataFrame(dict(game_id=["2026_01_A_B"] * 2, player_id=["p1", "p2"], market=["rec", "rec"], actual=[5.0, 5.0]))

    class RS:
        @staticmethod
        def played(season):
            return act, None, None
    import sys, types
    mod = types.ModuleType("live.results"); mod.played = RS.played
    sys.modules["live.results"] = mod
    import live
    live.results = mod
    g = pc.grade(e, 2026).sort("player_id")
    assert g["won"].to_list() == [1.0, 0.0] and np.allclose(g["profit"].to_list(), [0.5, -1.0])


def test_the_summary_reports_thresholds_and_a_return_interval():
    rng = np.random.default_rng(0)
    n = 400
    g = pl.DataFrame(dict(book="b", game_id=[f"2026_0{1 + i % 5}_A_B{i % 40}" for i in range(n)], player_id=[f"p{i}" for i in range(n)], market="rec", line=1.5, edge=rng.uniform(0, 0.15, n), model_chance=rng.uniform(0.3, 0.9, n),
                          price=-110.0, won=(rng.random(n) < 0.5).astype(float), week=[1 + i % 5 for i in range(n)])).with_columns(profit=pl.when(pl.col("won") > 0).then(0.909).otherwise(-1.0))
    s = pc.summary(g)
    assert set(s["threshold"].unique().to_list()) <= set(pc.THRESHOLDS) and (s["lo"] <= s["roi"]).all() and (s["roi"] <= s["hi"]).all()

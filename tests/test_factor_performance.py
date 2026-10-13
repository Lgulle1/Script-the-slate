"""Model health factor table (eval/factor_performance.py): plan "Phase 7 app and Phase 6.1 / 6.2"."""
import polars as pl
import pytest

import config
from eval import factor_performance as FP

needs = pytest.mark.skipif(not (config.ROOT / "layer_weights.csv").exists(), reason="no layer_weights.csv")


@needs
def test_one_row_per_layer_and_market_with_status_and_the_red_flag_rule():
    t = FP.build()
    w = pl.read_csv(config.ROOT / "layer_weights.csv")
    assert t.height == w.height and set(t.columns) >= {"layer", "market", "verdict", "weight", "gain_pct", "ci_lo", "ci_hi", "seasons_won", "last_season_gain", "status", "red_flag"}
    chk = t.with_columns(s2=pl.when(pl.col("weight") > 0).then(pl.lit("on")).otherwise(pl.lit("off")), r2=(pl.col("status") == "on") & (pl.col("last_season_gain") <= 0))
    assert (chk["status"] == chk["s2"]).all() and (chk["red_flag"].fill_null(False) == chk["r2"].fill_null(False)).all()


@needs
def test_the_kept_layers_carry_their_gain_columns_and_off_layers_do_not_need_them():
    t = FP.build()
    on = t.filter(pl.col("status") == "on")
    assert on.height >= 6 and on["gain_pct"].null_count() == 0 and (on["seasons_won"] >= 2).all()
    assert t.filter((pl.col("layer") == "4c_comparables"))["status"].unique().to_list() == ["off"]


def test_a_layer_that_is_on_but_lost_in_the_latest_season_is_flagged_red():
    w = pl.DataFrame(dict(layer=["x"], market=["m"], verdict=["CLEARS"], weight=[1.0], source=["s"]))
    g = pl.DataFrame(dict(layer=["x"], market=["m"], gain_pct=[1.0], ci_lo=[0.1], ci_hi=[2.0], seasons_won=[3], last_season_gain=[-0.01]))
    t = FP.combine(w, g)
    assert t["red_flag"].to_list() == [True] and t["status"].to_list() == ["on"]

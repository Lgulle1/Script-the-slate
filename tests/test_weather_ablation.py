"""4e ablation runner: severity assignment, the three-variant AND, the null control."""
from types import SimpleNamespace

import polars as pl
import pytest

import run_weather_ablation as RW
from features import weather_features as WF


def test_player_rows_take_their_games_severity_and_game_rows_their_own():
    data = SimpleNamespace(player_log=pl.DataFrame({"player_id": ["p1", "p2"], "season": [2022, 2022], "week": [3, 3], "game_id": ["gA", "gB"]}))
    pred = pl.DataFrame({"kind": ["player", "player", "game", "game"], "season": [2022] * 4, "week": [3, 3, 3, 3], "entity": ["p1", "p2", "gA", "gC"]})
    out = RW.assign_severity(pred, data, ["gA"]).sort("entity")
    assert dict(zip(out["entity"], out["severe"])) == {"gA": True, "gC": False, "p1": True, "p2": False}


def _s(verdict, ci_hi=0.1):
    return {"severe": {"verdict": verdict}, "other": {"ci_hi": ci_hi}}


def test_the_layer_needs_clears_under_all_three_variants_and_no_loss_elsewhere():
    ok = {v: _s("CLEARS") for v in RW.GATED}
    assert RW.layer_decision(ok)[0] == 1.0
    assert RW.layer_decision({**ok, "A2x2": _s("EDGE")})[0] == 0.0                   # an AND: one weak variant is enough to fail
    assert RW.layer_decision({**ok, "A1": _s("NO")})[0] == 0.0
    assert RW.layer_decision({**ok, "A2": _s("CLEARS", ci_hi=-0.01)})[0] == 0.0     # significantly worse on the other rows
    assert RW.layer_decision({**ok, "exact": _s("NO")})[0] == 1.0                    # the exact-weather run is reference only
    assert RW.layer_decision({**ok, "A1": {"severe": {"verdict": "CLEARS"}, "other": {"ci_hi": None}}})[0] == 1.0


def test_the_null_control_shuffles_within_the_week_only_and_deterministically():
    games = pl.DataFrame({"game_id": [f"g{i}" for i in range(8)], "season": [2022] * 8, "week": [1] * 4 + [2] * 4})
    f = pl.DataFrame({"game_id": games["game_id"], **{c: [float(i) for i in range(8)] for c in WF.WX_COLUMNS}})
    n = RW.null_features(f, games)
    for w, ids in ((1, [f"g{i}" for i in range(4)]), (2, [f"g{i}" for i in range(4, 8)])):
        assert sorted(n.filter(pl.col("game_id").is_in(ids))["wx_wind"].to_list()) == sorted(f.filter(pl.col("game_id").is_in(ids))["wx_wind"].to_list())
    assert n.sort("game_id")["wx_wind"].to_list() != f.sort("game_id")["wx_wind"].to_list() and n.equals(RW.null_features(f, games))

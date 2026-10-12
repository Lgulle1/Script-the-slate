"""4d.1 ablation runner: window assignment, the pass rule, the null control."""
from types import SimpleNamespace

import polars as pl
import pytest

import run_coach_ablation as RC
from features import coach_prior as CP


def _win():
    return pl.DataFrame({"game_id": ["g1", "g1", "g2", "g2"], "team": ["AAA", "BBB", "CCC", "DDD"], "season": [2022] * 4, "week": [1, 1, 9, 9],
                         "early": [True, True, False, False], "post_change": [False, False, False, True],
                         "window": ["W1+W2", "W1+W2", "W0", "W1+W2"]})


def test_player_rows_take_their_teams_window_and_game_rows_are_w1w2_when_either_team_is():
    data = SimpleNamespace(player_log=pl.DataFrame({"player_id": ["p1", "p2"], "season": [2022, 2022], "week": [9, 9], "team": ["CCC", "DDD"]}))
    pred = pl.DataFrame({"kind": ["player", "player", "game", "game"], "season": [2022] * 4, "week": [9, 9, 9, 1], "entity": ["p1", "p2", "g2", "g1"]})
    out = RC.assign_windows(pred, data, _win()).sort("entity")
    assert dict(zip(out["entity"], out["window"])) == {"g1": "W1+W2", "g2": "W1+W2", "p1": "W0", "p2": "W1+W2"}      # g2 has DDD in a post-change window


def _s(verdict, ci_hi=0.1):
    return {"verdict": verdict, "ci_hi": ci_hi}


def test_the_pass_rule_needs_clears_on_w1w2_and_no_significant_loss_on_w0():
    assert RC.layer_decision({"W1+W2": _s("CLEARS"), "W0": _s("NO", 0.2)})[0] == 1.0
    assert RC.layer_decision({"W1+W2": _s("CLEARS"), "W0": _s("NO", -0.01)})[0] == 0.0         # significantly worse outside the window
    assert RC.layer_decision({"W1+W2": _s("EDGE"), "W0": _s("EDGE")})[0] == 0.0
    assert RC.layer_decision({"W1+W2": _s("CLEARS"), "W0": {"verdict": "NO", "ci_hi": None}})[0] == 1.0          # no W0 rows: nothing to be worse on


def test_the_null_control_keeps_the_values_but_breaks_the_team_link():
    f = pl.DataFrame({"game_id": [f"g{i}" for i in range(8)], "team": list("ABCDEFGH"), "season": [2022] * 8, "week": [1] * 4 + [2] * 4,
                      **{c: [float(i) for i in range(8)] for c in CP.COACH_COLUMNS}})
    n = RC.null_features(f)
    assert sorted(n["coach_state"].to_list()) == sorted(f["coach_state"].to_list())
    for w in (1, 2):                                                                    # shuffled only within a week
        assert sorted(n.filter(pl.col("week") == w)["coach_games"].to_list()) == sorted(f.filter(pl.col("week") == w)["coach_games"].to_list())
    assert n["coach_state"].to_list() != f["coach_state"].to_list()
    assert n.equals(RC.null_features(f))                                                # deterministic


def test_output_names_carry_the_tag_and_build_valid_paths(tmp_path):
    """A precedence slip once crashed a finished run while writing the predictions (Path / str + str)."""
    import run_weather_ablation as RWx
    assert RC.predictions_name(None).endswith("eligibility_v2.parquet") and RC.predictions_name("run1").endswith("eligibility_v2_run1.parquet")
    assert RWx.predictions_name("t").startswith("weather_ablation_predictions_") and RWx.predictions_name("t").endswith("_t.parquet")
    assert (tmp_path / RC.predictions_name("run1")).parent == tmp_path

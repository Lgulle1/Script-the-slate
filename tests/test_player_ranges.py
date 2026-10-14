"""eval/player_ranges.py: the player-specific range is used only where players really differ in spread."""
import numpy as np
import polars as pl

from eval import calibrate as C
from eval import player_ranges as PR


def _world(spread_by_player: bool, seed=0, n_players=120, seasons=(2020, 2021, 2022, 2023, 2024), weeks=range(1, 15)):
    rng = np.random.default_rng(seed)
    scale = rng.choice([2.0, 12.0], size=n_players) if spread_by_player else np.full(n_players, 7.0)
    rows = []
    for s in seasons:
        for w in weeks:
            for j in range(n_players):
                p = 40 + 5 * rng.normal()
                rows.append((s, w, f"p{j}", p, max(0.0, p + scale[j] * rng.normal())))
    return pl.DataFrame(rows, schema=["season", "week", "entity", "prediction", "actual"], orient="row").with_columns(market=pl.lit("rec_yds"))


CFG = C.Config(k={m: 300 for m in C.RUNGS}, min_pool={m: 400 for m in C.RUNGS})


def test_history_uses_only_strictly_earlier_rows():
    d = pl.DataFrame(dict(market="rec_yds", season=[2020] * 4, week=[1, 2, 3, 4], entity="a", prediction=[10.0] * 4, actual=[10.0, 14.0, 20.0, 10.0]))
    h = PR.with_history(d)
    assert h["n_prior"].to_list() == [0, 1, 2, 3]
    assert h["d_j"].to_list()[1:] == [0.0, 2.0, 14.0 / 3]


def test_it_helps_when_players_differ_in_spread_and_the_ratio_follows_the_players():
    wf = PR.walk_forward(_world(True), "rec_yds", CFG)
    ev = PR.evaluate(wf)
    assert ev["passes"] and ev["lo"] > 0
    late = wf.filter(pl.col("n_prior") >= 30)
    assert late.filter(pl.col("r") > 1.2).height > 0 and late.filter(pl.col("r") < 0.7).height > 0


def test_it_does_not_pass_when_every_player_has_the_same_spread():
    ev = PR.evaluate(PR.walk_forward(_world(False, seed=1), "rec_yds", CFG))
    assert not ev["passes"]


def test_ratio_is_clipped_and_one_without_history():
    wf = PR.walk_forward(_world(True, seed=2), "rec_yds", CFG)
    assert wf["r"].min() >= PR.R_LO - 1e-12 and wf["r"].max() <= PR.R_HI + 1e-12
    assert (wf.filter(pl.col("n_prior") < PR.MIN_PRIOR)["r"] == 1.0).all()

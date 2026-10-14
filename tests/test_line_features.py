"""features/line_features.py: the informed model's only line input; the pure path never imports it."""
import re
from pathlib import Path

import polars as pl
import pytest

import config
from features import line_features as LF

ROOT = Path(__file__).resolve().parents[1]
S = pl.DataFrame(dict(game_id=["g1", "g2"], home_team=["A", "C"], away_team=["B", "D"], spread_line=[3.0, None], total_line=[44.0, 40.5]))


def test_team_sides_total_and_implied_points():
    f = LF.features(S).filter(pl.col("game_id") == "g1")
    a, b = f.filter(pl.col("team") == "A").row(0, named=True), f.filter(pl.col("team") == "B").row(0, named=True)
    assert (a["ln_team_spread"], a["ln_team_pts"], a["ln_opp_pts"], a["ln_total"]) == (3.0, 23.5, 20.5, 44.0)
    assert (b["ln_team_spread"], b["ln_team_pts"], b["ln_opp_pts"]) == (-3.0, 20.5, 23.5)


def test_a_missing_line_is_null_not_zero():
    f = LF.features(S).filter(pl.col("game_id") == "g2")
    assert f["ln_team_spread"].null_count() == 2 and f["ln_team_pts"].null_count() == 2 and (f["ln_total"] == 40.5).all()


def test_the_holdout_season_is_refused():
    with pytest.raises(config.HoldoutError):
        LF.load_schedule_lines(last=2025)


def test_attach_joins_on_game_and_team_and_keeps_row_count():
    from features.volume_features import FeatureTables
    players = {"targets": pl.DataFrame(dict(player_id=["p1", "p2"], game_id=["g1", "g1"], team=["A", "B"], x=[1.0, 2.0]))}
    teams = pl.DataFrame(dict(game_id=["g1", "g1"], team=["A", "B"], y=[1.0, 2.0]))
    out = LF.attach(FeatureTables(players, teams), LF.features(S))
    assert out.players["targets"]["ln_team_pts"].to_list() == [23.5, 20.5] and out.teams.height == 2


def test_only_the_informed_path_reads_line_features():
    allowed = {"features/line_features.py", "run_lines_ablation.py", "tests/test_line_features.py"}
    users = []
    for p in list(ROOT.rglob("*.py")):
        rel = str(p.relative_to(ROOT))
        if rel in allowed or "egg-info" in rel or rel.startswith("."):
            continue
        if re.search(r"line_features", p.read_text()):
            users.append(rel)
    assert users == [], users

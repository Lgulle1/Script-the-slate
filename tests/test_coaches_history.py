"""4d.0 coaches_history (option 3): head coach from nflverse 2016-2024, 2026 from the seed, everything else UNKNOWN as null; 2025 never read."""
import polars as pl
import pytest

import config
from ingest import build_coaches_history as CH

GAMES = pl.DataFrame({
    "season": [2020] * 6, "week": [1, 2, 3, 1, 2, 3], "game_id": ["g1", "g2", "g3", "g1", "g2", "g3"],
    "team": ["DET", "DET", "DET", "OAK", "OAK", "OAK"],
    "head_coach": ["Matt Patricia", "Matt Patricia", "Darrell Bevell", "Jon Gruden", "Jon Gruden", "Jon Gruden"]})
SEED = pl.DataFrame({"team": ["Detroit Lions"] * 4, "role": ["head_coach", "offensive_coordinator", "offensive_play_caller", "defensive_coordinator"],
                     "name": ["Dan Campbell", "Drew Petzing", "Drew Petzing", "Kelvin Sheppard"], "source_url": ["u"] * 4, "confidence": ["high"] * 4})


def test_every_team_season_has_one_row_per_role_known_or_unknown():
    h = CH.history(GAMES, SEED, seasons=(2020, 2025, 2026), teams=("DET", "LV"))
    assert h.group_by("season", "team", "role").len()["len"].max() == 1
    assert h.height == 3 * 2 * len(CH.ROLES)
    assert set(h["role"].unique()) == set(CH.ROLES) == {"head_coach", "offensive_coordinator", "play_caller"}


def test_unknown_means_null_not_zero_or_empty():
    h = CH.history(GAMES, SEED, seasons=(2020, 2025, 2026), teams=("DET", "LV"))
    unk = h.filter(pl.col("status") == "UNKNOWN")
    assert unk.height > 0 and unk["name"].null_count() == unk.height                 # null, never "" / "0"
    assert (h.filter(pl.col("status") == "known")["name"].str.len_chars() > 0).all()
    assert CH.lookup(h, 2020, "DET", "play_caller") is None                             # a missing play caller carries no information
    assert CH.lookup(h, 2026, "DET", "play_caller") == "Drew Petzing"


def test_head_coach_is_the_coach_of_most_games_and_an_interim_change_is_kept():
    h = CH.history(GAMES, SEED, seasons=(2020,), teams=("DET", "LV"))
    det = h.filter((pl.col("team") == "DET") & (pl.col("role") == "head_coach")).row(0, named=True)
    assert det["name"] == "Matt Patricia" and det["n_games"] == 2 and "Darrell Bevell (1)" in det["note"]
    lv = h.filter((pl.col("team") == "LV") & (pl.col("role") == "head_coach")).row(0, named=True)
    assert lv["name"] == "Jon Gruden" and lv["team_code_that_season"] == "OAK"        # franchise code LV, the season's own code kept


def test_2025_is_unknown_and_never_read():
    h = CH.history(GAMES, SEED, seasons=(2020, 2025, 2026), teams=("DET",))
    s25 = h.filter(pl.col("season") == 2025)
    assert (s25["status"] == "UNKNOWN").all() and s25["note"].str.contains("holdout").all()
    with pytest.raises(config.HoldoutError):
        CH.load_head_coach_games(last=2025)

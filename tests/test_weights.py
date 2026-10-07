import math

import pytest

from features import weights as w

TEST_PENALTIES = {"rec_yds": {"QB": 0.8, "HC": 0.95, "role": 0.5}}


def test_recency_half_life():
    assert w.recency_weight(0) == 1.0
    assert w.recency_weight(6) == pytest.approx(0.5)
    assert w.recency_weight(12) == pytest.approx(0.25)
    assert w.recency_weight(1) == pytest.approx(0.5 ** (1 / 6))
    with pytest.raises(ValueError):
        w.recency_weight(-1)


def test_games_elapsed_within_and_across_seasons():
    assert w.games_elapsed(2024, 5, 2024, 6) == 1
    assert w.games_elapsed(2024, 1, 2024, 10) == 9
    # last game of 2023 -> first of 2024: 0 remaining + 8 offseason + 1
    assert w.games_elapsed(2023, 17, 2024, 1) == 9
    # game 15 of 2023 -> game 3 of 2024: 2 + 8 + 3
    assert w.games_elapsed(2023, 15, 2024, 3) == 13
    # skipping a whole season adds its 17 games and another offseason
    assert w.games_elapsed(2022, 17, 2024, 1) == 8 + 17 + 8 + 1
    # pre-2021 seasons are 16 games
    assert w.games_elapsed(2020, 16, 2021, 1) == 9
    assert w.games_elapsed(2020, 10, 2021, 1) == 6 + 8 + 1
    with pytest.raises(ValueError):
        w.games_elapsed(2024, 5, 2024, 5)


def test_continuity_is_product_of_changed_factors():
    assert w.continuity_weight({"QB": True, "HC": True, "role": False}, "rec_yds", TEST_PENALTIES) == pytest.approx(0.8 * 0.95)
    assert w.continuity_weight({"QB": False}, "rec_yds", TEST_PENALTIES) == 1.0
    assert w.continuity_weight({}, "rec_yds", TEST_PENALTIES) == 1.0


def test_continuity_fails_loudly_instead_of_silent_no_penalty():
    with pytest.raises(KeyError, match="no continuity penalties"):
        w.continuity_weight({"QB": True}, "not_a_market")
    with pytest.raises(KeyError, match="no penalty for factor"):
        w.continuity_weight({"OC": True}, "rec_yds", TEST_PENALTIES)
    with pytest.raises(KeyError, match="unknown continuity factors"):
        w.continuity_weight({"kicker": True}, "rec_yds", TEST_PENALTIES)


def test_quality_weights():
    assert [w.quality_weight(q) for q in ("observed", "derived", "estimated")] == [1.0, 0.9, 0.65]
    with pytest.raises(KeyError):
        w.quality_weight("guess")


def test_guide_penalty_table_is_complete_and_sane():
    import config
    assert len(config.CONTINUITY_PENALTIES) == 11
    for market, table in config.CONTINUITY_PENALTIES.items():
        assert set(table) == set(config.CONTINUITY_FACTORS), market
        assert all(0 < v <= 1 for v in table.values()), market
    # spot checks against the guide
    assert w.continuity_weight({"QB": True}, "pass_yds") == 0.25
    assert w.continuity_weight({"QB": True, "OC": True}, "pass_yds") == pytest.approx(0.25 * 0.65)
    assert set(config.BASELINE_TO_PENALTY_MARKET.values()) <= set(config.CONTINUITY_PENALTIES)

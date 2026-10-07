import config


def test_all_32_teams_present():
    assert len(config.stadium_coordinates) == 32


def test_entries_valid():
    roofs = {config.OUTDOORS, config.DOME, config.RETRACTABLE}
    for team, (lat, lon, roof) in config.stadium_coordinates.items():
        assert 24 < lat < 50, team
        assert -125 < lon < -66, team
        assert roof in roofs, team


def test_seasons():
    assert config.BACKTEST_SEASONS == [2020, 2021, 2022, 2023, 2024]
    assert config.HOLDOUT_SEASON not in config.BACKTEST_SEASONS

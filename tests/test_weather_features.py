"""4e weather features (features/weather_features.py): the three degradation variants, domes neutral, roof-unknown / missing = null, determinism."""
import polars as pl
import pytest

from features import weather_features as WF


def _wx(rows):
    """rows: (game_id, temp_f, wind_mph, precip_in, indoor)."""
    return pl.DataFrame([dict(game_id=g, temp_f=t, wind_mph=w, precip_in=p, indoor=i) for g, t, w, p, i in rows],
                        schema={"game_id": pl.String, "temp_f": pl.Float64, "wind_mph": pl.Float64, "precip_in": pl.Float64, "indoor": pl.Boolean})


def _row(f, gid):
    return f.filter(pl.col("game_id") == gid).row(0, named=True)


@pytest.mark.parametrize("mode", WF.MODES)
def test_an_indoor_game_gets_neutral_values_whatever_its_observed_weather(mode):
    a = WF.features(_wx([("in1", 20.0, 30.0, 0.5, True)]), mode)
    b = WF.features(_wx([("in1", 95.0, 0.0, 0.0, True)]), mode)
    assert a.equals(b)                                                               # the dome's features do not move with the observed row
    r = _row(a, "in1")
    assert r["wx_indoor"] == 1.0 and r["wx_windy"] == 0.0 and r["wx_cold"] == 0.0 and r["wx_precip"] == 0.0
    assert r["wx_wind"] == (WF.bin_wind(0.0) if mode == "A1" else 0.0) and r["wx_temp"] == (WF.bin_temp(70.0) if mode == "A1" else 70.0)


@pytest.mark.parametrize("mode", WF.MODES)
def test_roof_unknown_and_missing_weather_are_missing_features_not_neutral_values(mode):
    f = WF.features(_wx([("unk", 50.0, 5.0, 0.0, None), ("out", 50.0, None, None, False)]), mode)
    u = _row(f, "unk")
    assert all(u[c] is None for c in WF.WX_COLUMNS)                                  # unknown roof: nothing is pretended
    o = _row(f, "out")
    assert o["wx_indoor"] == 0.0 and o["wx_wind"] is None and o["wx_windy"] is None and o["wx_precip"] is None and o["wx_temp"] is not None
    games = pl.DataFrame({"game_id": ["unk", "out", "absent"]})
    assert WF.features(_wx([("unk", 50.0, 5.0, 0.0, None)]), mode, games=games).filter(pl.col("game_id") == "absent")["wx_wind"].null_count() == 1


def test_bins_follow_the_plan_edges():
    assert [WF.bin_wind(x) for x in (0, 9.99, 10, 14.99, 15, 19.99, 20, 40)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [WF.bin_temp(x) for x in (-5, 32, 32.01, 50, 50.01, 70, 70.01, 95)] == [0, 0, 1, 1, 2, 2, 3, 3]
    f = WF.features(_wx([("a", 32.0, 15.0, 0.01, False), ("b", 71.0, 9.0, 0.0, False)]), "A1")
    a, b = _row(f, "a"), _row(f, "b")
    assert (a["wx_wind"], a["wx_temp"], a["wx_precip"], a["wx_windy"], a["wx_cold"]) == (2.0, 0.0, 1.0, 1.0, 1.0)
    assert (b["wx_wind"], b["wx_temp"], b["wx_precip"], b["wx_windy"], b["wx_cold"]) == (0.0, 3.0, 0.0, 0.0, 0.0)


def test_exact_mode_returns_the_observed_values():
    r = _row(WF.features(_wx([("a", 28.0, 17.5, 0.2, False)]), "exact"), "a")
    assert (r["wx_wind"], r["wx_temp"], r["wx_precip"], r["wx_windy"], r["wx_cold"]) == (17.5, 28.0, 1.0, 1.0, 1.0)


def test_noise_is_deterministic_per_game_and_independent_of_the_other_games():
    rows = [(f"g{i}", 50.0, 20.0, 0.0, False) for i in range(5)]
    a = WF.features(_wx(rows), "A2")
    b = WF.features(_wx(rows[::-1]), "A2").sort("game_id")
    assert a.sort("game_id").equals(b)                                               # row order does not matter
    one = WF.features(_wx(rows[2:3]), "A2")
    assert _row(one, "g2") == _row(a, "g2")                                          # a game's draw depends on its id only
    assert a["wx_wind"].n_unique() == 5 and not all(abs(w - 20.0) < 1e-9 for w in a["wx_wind"])


def test_double_noise_is_the_same_draws_at_twice_the_size_and_its_flips_contain_the_single_noise_flips():
    rows = [(f"g{i}", 50.0, 30.0, 0.0, False) for i in range(300)]
    one, two = WF.features(_wx(rows), "A2").sort("game_id"), WF.features(_wx(rows), "A2x2").sort("game_id")
    d1, d2 = one["wx_wind"].to_numpy() - 30.0, two["wx_wind"].to_numpy() - 30.0
    assert d2 == pytest.approx(2 * d1)                                               # wind 30 mph: no clipping at 0
    t1, t2 = one["wx_temp"].to_numpy() - 50.0, two["wx_temp"].to_numpy() - 50.0
    assert t2 == pytest.approx(2 * t1)
    f1, f2 = one["wx_precip"].to_numpy() == 1.0, two["wx_precip"].to_numpy() == 1.0   # observed precipitation is no: a flip makes it yes
    assert (f1 <= f2).all() and f1.sum() < f2.sum()
    assert 0.1 < f1.mean() < 0.3 and 0.3 < f2.mean() < 0.5                           # flip probabilities 0.2 and 0.4
    assert d1.std() == pytest.approx(WF.A2_WIND_SD, rel=0.2)


def test_wind_never_goes_below_zero():
    f = WF.features(_wx([(f"g{i}", 50.0, 0.5, 0.0, False) for i in range(200)]), "A2x2")
    assert f["wx_wind"].min() >= 0.0


def test_severe_weather_games_are_outdoor_with_observed_wind_15_or_temperature_32_or_less():
    wx = _wx([("calm", 60.0, 5.0, 0.0, False), ("windy", 60.0, 15.0, 0.0, False), ("cold", 32.0, 3.0, 0.0, False), ("dome", 10.0, 30.0, 0.0, True),
              ("unk", 10.0, 30.0, 0.0, None)])
    assert sorted(WF.severe_games(wx)) == ["cold", "windy"]


def test_attach_joins_on_game_without_changing_rows_and_leaves_the_phase3_tables_without_weather():
    from features.volume_features import FeatureTables
    t = FeatureTables({"rush_att": pl.DataFrame({"player_id": ["p1", "p2"], "game_id": ["g1", "g1"], "team": ["A", "B"]})},
                      pl.DataFrame({"game_id": ["g1", "g1"], "team": ["A", "B"]}))
    f = WF.features(_wx([("g1", 50.0, 12.0, 0.0, False)]), "A2")
    out = WF.attach(t, f)
    assert out.players["rush_att"].height == 2 and out.teams.height == 2 and out.players["rush_att"]["wx_wind"].n_unique() == 1
    assert all(out.teams[c].dtype == pl.Float64 for c in WF.WX_COLUMNS)
    assert WF.attach(t, None) is t
    with pytest.raises(ValueError):
        WF.attach(t, pl.concat([f, f]))


def test_the_holdout_season_is_never_loaded():
    import config
    with pytest.raises(config.HoldoutError):
        WF.load_observed(last=2025)

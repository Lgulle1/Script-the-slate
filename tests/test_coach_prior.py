"""4d.1 coaching prior (features/coach_prior.py): stints and change events, the three tendencies, the stop weights, walk-forward, null when unknown."""
import polars as pl
import pytest

import config
from features import coach_prior as CP


def _games(rows):
    """rows: (season, week, team, coach)."""
    return pl.DataFrame([dict(season=s, week=w, game_id=f"{s}_{w:02d}_{t}", team=t, head_coach=c) for s, w, t, c in rows])


def test_stints_flag_a_change_only_when_the_coach_differs_from_the_teams_previous_game():
    g = _games([(2019, 1, "AAA", "Old"), (2019, 2, "AAA", "Old"), (2019, 3, "AAA", "Interim"), (2019, 4, "AAA", "Interim"),
                (2020, 1, "AAA", "New"), (2020, 2, "AAA", "New"), (2019, 1, "BBB", "Same"), (2020, 1, "BBB", "Same")])
    cg = CP.coach_games(g).sort("team", "season", "week")
    a = cg.filter(pl.col("team") == "AAA")
    assert a["change_event"].to_list() == [False, False, True, False, True, False]            # the first game observed is no change
    assert a["games_before"].to_list() == [0, 1, 0, 1, 0, 1]
    assert cg.filter(pl.col("team") == "BBB")["change_event"].sum() == 0


def test_a_returning_coach_starts_a_new_stint():
    g = _games([(2016, 1, "AAA", "X"), (2017, 1, "AAA", "Y"), (2018, 1, "AAA", "X")])
    cg = CP.coach_games(g).sort("season")
    assert cg["stint"].n_unique() == 3 and cg["change_event"].to_list() == [False, True, True]


def _plays(rows):
    """rows: dicts with defaults for a neutral scrimmage play."""
    base = dict(season=2018, week=1, game_id="g1", posteam="AAA", defteam="BBB", drive=1.0, qtr=2.0, down=1.0, ydstogo=10.0, yardline_100=60.0,
                play_type="run", **{"pass": 0.0}, rush=1.0, xpass=0.5, wp=0.5, game_seconds_remaining=2000.0, half_seconds_remaining=800.0,
                timeout=0.0, penalty=0.0, qb_spike=0.0, qb_kneel=0.0)
    out = []
    for i, r in enumerate(rows):
        d = dict(base, play_id=float(i + 1))
        d.update(r)
        out.append(d)
    return pl.DataFrame(out)


def test_pass_rate_over_expected_is_the_mean_of_pass_minus_xpass():
    p = _plays([{"pass": 1.0, "rush": 0.0, "play_type": "pass", "xpass": 0.5}, {"pass": 1.0, "rush": 0.0, "play_type": "pass", "xpass": 0.75},
                {"xpass": 0.25}, {"xpass": 0.5}, {"qb_kneel": 1.0, "play_type": "qb_kneel", "rush": 0.0, "xpass": 0.0}])
    assert CP.proe(p) == (pytest.approx((0.5 + 0.25 - 0.25 - 0.5) / 4), 4)          # kneel is no designed play; n = 4


def test_pace_is_the_median_seconds_between_neutral_plays_of_one_drive():
    clocks = [2000, 1960, 1930, 1890, 1860]                                          # gaps 40, 30, 40, 30 -> median 35
    p = _plays([{"game_seconds_remaining": float(c), "half_seconds_remaining": float(c) - 1200} for c in clocks])
    assert CP.pace(p) == (pytest.approx(35.0), 4)


def test_pace_leaves_out_non_neutral_states_late_halves_timeouts_and_other_drives():
    ok = [{"game_seconds_remaining": 2000.0 - 40 * i, "half_seconds_remaining": 800.0 - 40 * i} for i in range(3)]               # two gaps of 40
    blowout = {"game_seconds_remaining": 1900.0, "half_seconds_remaining": 700.0, "wp": 0.95}                                    # not neutral
    late = {"game_seconds_remaining": 1000.0, "half_seconds_remaining": 100.0, "drive": 2.0}                                    # last 2 minutes
    other = [{"game_seconds_remaining": 1700.0, "half_seconds_remaining": 500.0, "drive": 3.0}, {"game_seconds_remaining": 1600.0, "half_seconds_remaining": 400.0, "drive": 4.0}]
    p = _plays(ok + [blowout, late] + other)
    n = CP.pace(p)[1]
    assert CP.pace(p)[0] == pytest.approx(40.0) and n == 2                           # only the two in-drive gaps of the neutral drive count
    to = _plays([{"game_seconds_remaining": 2000.0, "half_seconds_remaining": 800.0}, {"timeout": 1.0, "rush": 0.0, "play_type": "no_play",
                 "game_seconds_remaining": 1990.0, "half_seconds_remaining": 790.0}, {"game_seconds_remaining": 1940.0, "half_seconds_remaining": 740.0}])
    assert CP.pace(to)[1] == 0                                                       # a timeout between two plays removes the gap


def test_fourth_down_aggressiveness_is_the_go_rate_over_the_league_rate_in_the_same_situation():
    def fourth(team, go, n):
        return [{"posteam": team, "down": 4.0, "ydstogo": 2.0, "yardline_100": 30.0, "play_type": "run" if go else "field_goal", "rush": 1.0 if go else 0.0}
                for _ in range(n)]
    p = _plays(fourth("AAA", True, 3) + fourth("AAA", False, 1) + fourth("BBB", False, 4))                              # league go rate 3/8
    g = CP.fourth_down(p, by="posteam")
    assert g.filter(pl.col("posteam") == "AAA")["go4"][0] == pytest.approx(0.75 - 3 / 8)
    assert g.filter(pl.col("posteam") == "BBB")["go4"][0] == pytest.approx(0 - 3 / 8)


def _stops(rows):
    """rows: (coach, franchise, last_season, last_week, n_games, proe, pace, go4)."""
    return pl.DataFrame([dict(coach=c, franchise=f, last_key=s * 100 + w, last_season=s, n_games=n, proe=a, pace=b, go4=d)
                         for c, f, s, w, n, a, b, d in rows],
                        schema={"coach": pl.String, "franchise": pl.String, "last_key": pl.Int64, "last_season": pl.Int64, "n_games": pl.Int64,
                                "proe": pl.Float64, "pace": pl.Float64, "go4": pl.Float64})


def test_the_prior_weights_each_stop_by_recency_and_length():
    stops = _stops([("C", "OLD", 2019, 17, 17, 0.10, 28.0, 0.2), ("C", "OLDER", 2016, 17, 34, 0.0, 30.0, 0.0)])
    pr = CP.prior(stops, "C", "NEW", 2022, 1)
    a, b = 0.5 ** (3 / 3) * 17 / 34, 0.5 ** (6 / 3) * 34 / 51
    assert pr["proe"] == pytest.approx((a * 0.10 + b * 0.0) / (a + b)) and pr["pace"] == pytest.approx((a * 28.0 + b * 30.0) / (a + b))
    assert pr["weight"] == pytest.approx(a + b)


def test_a_prior_never_uses_the_current_team_a_stop_that_has_not_ended_or_another_coach():
    stops = _stops([("C", "NEW", 2018, 17, 30, 0.9, 20.0, 0.9), ("C", "OLD", 2022, 5, 20, 0.5, 25.0, 0.5), ("D", "OLD", 2016, 17, 20, 0.3, 26.0, 0.3)])
    assert CP.prior(stops, "C", "NEW", 2022, 5) is None                              # the only other stop ends at 2022 wk 5, not before wk 5; NEW is his current team
    assert CP.prior(stops, "C", "NEW", 2022, 6)["proe"] == pytest.approx(0.5)        # one week later it has ended and counts


def test_state_and_null_features():
    g = _games([(2019, 1, "AAA", "Old"), (2019, 2, "AAA", "Old"), (2020, 1, "AAA", "Vet"), (2020, 2, "AAA", "Vet"),
                (2020, 1, "BBB", "Rookie"), (2020, 2, "BBB", "Rookie"), (2019, 1, "BBB", "Same"), (2019, 1, "VETSTOP", "Vet")])
    stops = _stops([("Vet", "VETSTOP", 2019, 1, 1, 0.2, 29.0, 0.1)])
    f = CP.features(g, stops).sort("team", "season", "week")
    vet = f.filter((pl.col("team") == "AAA") & (pl.col("season") == 2020))
    assert vet["coach_state"].to_list() == [1.0, 1.0] and vet["coach_prior_proe"].to_list() == pytest.approx([0.2, 0.2])
    assert vet["coach_games"].to_list() == [0.0, 1.0] and vet["coach_prior_weight"].to_list() == pytest.approx([1.0, 6 / 7])
    rook = f.filter((pl.col("team") == "BBB") & (pl.col("season") == 2020))
    assert rook["coach_state"].to_list() == [2.0, 2.0] and rook["coach_prior_proe"].null_count() == 2 and rook["coach_prior_weight"].null_count() == 2
    old = f.filter((pl.col("team") == "AAA") & (pl.col("season") == 2019))
    assert old["coach_state"].to_list() == [0.0, 0.0] and old["coach_prior_pace"].null_count() == 2        # no change: no prior, null not zero


def test_a_coach_is_new_only_for_his_first_17_games():
    rows = [(2019, w, "AAA", "Old") for w in range(1, 4)] + [(2020, w, "AAA", "New") for w in range(1, 19)]
    f = CP.features(_games(rows), _stops([])).filter(pl.col("season") == 2020).sort("week")
    assert f["coach_state"].to_list() == [2.0] * 17 + [0.0]


def test_the_windows():
    g = _games([(2019, 1, "AAA", "Old")] + [(2020, w, "AAA", "New") for w in range(1, 9)] + [(2021, w, "AAA", "New") for w in (1, 7)])
    f = CP.features(g, _stops([])).sort("season", "week")
    w = CP.windows(f)
    s20 = w.filter(pl.col("season") == 2020).sort("week")["window"].to_list()
    assert s20 == ["W1+W2"] * 6 + ["W0", "W0"]                                         # weeks 1-6 and the first six games under the new coach
    s21 = w.filter(pl.col("season") == 2021).sort("week")["window"].to_list()
    assert s21 == ["W1+W2", "W0"]                                                      # 2021 wk 1 is early season; wk 7 is the 10th game under him: outside both


def test_the_holdout_is_never_read():
    with pytest.raises(config.HoldoutError):
        CP.load_pbp(last=2025)


def _tables():
    from features.volume_features import FeatureTables
    players = {"rush_att": pl.DataFrame({"player_id": ["p1", "p2", "p3"], "game_id": ["g1", "g1", "g2"], "team": ["AAA", "AAA", "BBB"], "x": [1.0, 2.0, 3.0]})}
    return FeatureTables(players, pl.DataFrame({"game_id": ["g1", "g2"], "team": ["AAA", "BBB"], "y": [1.0, 2.0]}))


def _feats():
    return pl.DataFrame({"game_id": ["g1", "g2"], "team": ["AAA", "BBB"], "season": [2022, 2022], "week": [1, 1], "coach_state": [1.0, 0.0], "coach_games": [0.0, 5.0],
                         "coach_prior_proe": [0.02, None], "coach_prior_pace": [37.0, None], "coach_prior_go4": [0.01, None], "coach_prior_weight": [1.0, None]})


def test_attach_joins_the_columns_on_game_and_team_without_changing_the_rows():
    t = _tables()
    out = CP.attach(t, _feats())
    p, tm = out.players["rush_att"], out.teams
    assert p.height == 3 and tm.height == 2 and p["player_id"].to_list() == ["p1", "p2", "p3"]
    assert all(p[c].dtype == pl.Float64 for c in CP.COACH_COLUMNS) and p["coach_prior_proe"].to_list() == [0.02, 0.02, None]      # no prior stays null, not zero
    assert tm["coach_state"].to_list() == [1.0, 0.0]
    assert CP.attach(t, None) is t                                                    # no coach features: the Phase 3 tables unchanged


def test_attach_refuses_a_table_missing_requested_columns_or_with_duplicate_keys():
    with pytest.raises(KeyError):
        CP.attach(_tables(), _feats().drop("coach_prior_pace"))
    with pytest.raises(ValueError):
        CP.attach(_tables(), pl.concat([_feats(), _feats()]))

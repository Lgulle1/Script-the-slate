import math
from datetime import date, timedelta

import polars as pl
import pytest

import config
from eval import baselines as b
from features import weights as w


def make_log(rows):
    """rows: (player_id, season, game_num, gameday, opponent, family, slot, stat dict overrides)."""
    out = []
    for pid, season, g, day, opp, fam, slot, stats in rows:
        rec = dict(player_id=pid, season=season, week=g, game_id=f"{season}_{g}", team="AAA", opponent=opp,
                   family=fam, slot=slot, gameday=day, team_game_num=g, **{c: 0.0 for c in b.PLAYER_MARKETS.values()})
        rec.update(stats)
        out.append(rec)
    return pl.DataFrame(out)


D = lambda n: date(2024, 9, 1) + timedelta(days=7 * n)  # noqa: E731
LOG = make_log([("p1", 2024, i + 1, D(i), "BBB", "WR", 1, {"receiving_yards": float(10 * (i + 1)), "targets": float(i + 1)})
                for i in range(6)])
T = b.PlayerTarget("p1", D(6), 2024, 7, "BBB", "WR", 1)


def test_last3_and_season_average():
    assert b.player_last3(LOG, T, D(6))["rec_yds"] == pytest.approx((40 + 50 + 60) / 3)
    assert b.player_season_avg(LOG, T, D(6))["rec_yds"] == pytest.approx(35)
    assert b.player_season_avg(LOG, T, D(6))["targets"] == pytest.approx(3.5)


def test_recency_matches_manual_weights():
    vals = [10 * (i + 1) for i in range(6)]
    ws = [w.recency_weight(7 - (i + 1)) for i in range(6)]
    expect = sum(a * v for a, v in zip(ws, vals)) / sum(ws)
    assert b.player_recency(LOG, T, D(6))["rec_yds"] == pytest.approx(expect)
    assert expect > 35  # recent games weigh more than a plain average


def test_recency_applies_offseason_gap():
    old = make_log([("p1", 2023, 17, date(2023, 12, 31), "BBB", "WR", 1, {"receiving_yards": 100.0})])
    t = b.PlayerTarget("p1", date(2024, 9, 8), 2024, 1, "BBB", "WR", 1)
    assert b.player_recency(old, t, t.gameday)["rec_yds"] == 100.0  # single game: weight irrelevant
    both = pl.concat([old, make_log([("p1", 2024, 1, date(2024, 9, 8) - timedelta(days=1), "BBB", "WR", 1, {})]).with_columns(pl.col("receiving_yards") * 0)])
    # game 17 of 2023 is 9 games back, so it carries weight 0.5**(9/6) relative to a zero game 1 back
    wa, wb = w.recency_weight(9), w.recency_weight(0)
    t2 = b.PlayerTarget("p1", date(2024, 9, 8), 2024, 1, "BBB", "WR", 1)
    # new game has team_game_num 1 in same season as target -> elapsed 0 would be invalid; use game 1 vs target 2
    t2 = b.PlayerTarget("p1", date(2024, 9, 8), 2024, 2, "BBB", "WR", 1)
    got = b.player_recency(both, t2, t2.gameday)["rec_yds"]
    wa, wb = w.recency_weight(w.games_elapsed(2023, 17, 2024, 2)), w.recency_weight(1)
    assert got == pytest.approx(100 * wa / (wa + wb))


def test_no_history_returns_none():
    t = b.PlayerTarget("nobody", D(6), 2024, 7, "BBB", "WR", 1)
    for fn in b.PLAYER_BASELINES.values():
        assert all(v is None for v in fn(LOG, t, D(6)).values()) or fn is b.player_role_avg


def test_cutoff_after_game_date_is_rejected():
    with pytest.raises(ValueError, match="after the target game"):
        b.player_last3(LOG, T, D(6) + timedelta(days=1))


@pytest.mark.parametrize("method", b.METHODS)
def test_future_rows_cannot_change_predictions(method):
    """Leakage guard: scramble every row on/after the cutoff; predictions must not move."""
    extra = make_log([("p2", 2024, i + 1, D(i), "BBB", "WR", 1, {"receiving_yards": float(7 * i)}) for i in range(6)])
    full = pl.concat([LOG, extra])
    future = make_log([("p1", 2024, 7 + i, D(6 + i), "BBB", "WR", 1, {"receiving_yards": 999.0, "targets": 99.0}) for i in range(3)]
                      + [("p2", 2024, 7, D(6), "BBB", "WR", 1, {"receiving_yards": 555.0})])
    base = b.predict_player(method, full, T)
    assert base == b.predict_player(method, pl.concat([full, future]), T)
    assert base == b.predict_player(method, pl.concat([full, future.with_columns(pl.col("receiving_yards") * -1)]), T)


def test_role_average_pools_same_family_and_slot_only():
    log = make_log([("a", 2024, 1, D(0), "X", "WR", 1, {"receiving_yards": 80.0}),
                    ("b", 2024, 1, D(0), "X", "WR", 1, {"receiving_yards": 40.0}),
                    ("c", 2024, 1, D(0), "X", "WR", 2, {"receiving_yards": 500.0}),
                    ("d", 2024, 1, D(0), "X", "TE", 1, {"receiving_yards": 500.0})])
    t = b.PlayerTarget("zzz", D(1), 2024, 2, "Y", "WR", 1)
    assert b.player_role_avg(log, t, D(1))["rec_yds"] == pytest.approx(60)


def test_role_window_excludes_old_games():
    log = make_log([("a", 2022, 1, date(2022, 9, 1), "X", "WR", 1, {"receiving_yards": 999.0}),
                    ("a", 2024, 1, D(0), "X", "WR", 1, {"receiving_yards": 10.0})])
    t = b.PlayerTarget("zzz", D(1), 2024, 2, "Y", "WR", 1)
    assert b.player_role_avg(log, t, D(1))["rec_yds"] == 10.0


def test_blend_is_70_30_of_own_and_opponent_allowed():
    log = make_log([("p1", 2024, i + 1, D(i), "BBB", "WR", 1, {"receiving_yards": 100.0}) for i in range(3)]
                   + [("q", 2024, 1, D(0), "OPP", "WR", 1, {"receiving_yards": 50.0}),
                      ("q", 2024, 2, D(1), "OPP", "WR", 1, {"receiving_yards": 30.0})])
    t = b.PlayerTarget("p1", D(3), 2024, 4, "OPP", "WR", 1)
    assert b.player_blend(log, t, D(3))["rec_yds"] == pytest.approx(0.7 * 100 + 0.3 * 40)


def test_blend_falls_back_to_last3_in_week_one_and_to_own_without_opponent_data():
    prior = make_log([("p1", 2023, 17, date(2023, 12, 31), "BBB", "WR", 1, {"receiving_yards": 80.0})])
    t = b.PlayerTarget("p1", date(2024, 9, 8), 2024, 1, "NEWOPP", "WR", 1)
    assert b.player_blend(prior, t, t.gameday)["rec_yds"] == pytest.approx(80.0)


# ---- game markets
TL = pl.DataFrame([dict(team=t, opponent=o, gameday=D(i), season=2024, team_game_num=i + 1, pf=pf, pa=pa)
                   for t, o, pf, pa in [("H", "A", 30, 20), ("A", "H", 20, 30)] for i in range(3)])
G = b.GameTarget(D(3), 2024, "H", "A", 4, 4)


def test_game_baselines_basic():
    r = b.predict_game("season_avg", TL, G)
    assert (r["spread"], r["total"]) == (10, 50)
    assert r["moneyline"] == pytest.approx(0.5 * (1 + math.erf(10 / config.GAME_MARGIN_SD / math.sqrt(2))))
    assert b.predict_game("last3", TL, G)["total"] == 50
    r = b.predict_game("blend_70_30", TL, G)  # home: .7*30 + .3*(away allowed 30) = 30; away: .7*20 + .3*(home allowed 20)=20
    assert (r["spread"], r["total"]) == (10, 50)
    r = b.predict_game("role_avg", TL, G)  # league average: symmetric
    assert r["spread"] == 0 and r["total"] == 50 and r["moneyline"] == pytest.approx(0.5)


def test_game_future_rows_cannot_leak_and_missing_history_is_none():
    fut = pl.DataFrame([dict(team="H", opponent="A", gameday=D(3), season=2024, team_game_num=4, pf=99, pa=0)])
    for m in b.METHODS:
        assert b.predict_game(m, TL, G) == b.predict_game(m, pl.concat([TL, fut]), G)
    new = b.GameTarget(D(3), 2024, "X", "Y", 1, 1)
    assert b.predict_game("last3", TL, new) == {"spread": None, "moneyline": None, "total": None}
    with pytest.raises(ValueError):
        b.predict_game("last3", TL, G, cutoff=D(4))


def test_builders_on_real_data_and_real_leakage_check():
    import config as c
    if not c.RAW_DUCKDB_PATH.exists():
        pytest.skip("no raw db")
    plog = b.build_player_game_log().with_columns(pl.col(c).cast(pl.Float64) for c in b.PLAYER_MARKETS.values())
    assert {"player_id", "gameday", "family", "slot", "team_game_num"} <= set(plog.columns)
    assert plog["team_game_num"].min() == 1 and plog["slot"].max() == 3
    assert plog.select(pl.struct("player_id", "game_id").is_duplicated().any()).item() is False
    # take a real player-game from 2024 and check cutoff isolation on real data
    row = plog.filter((pl.col("season") == 2024) & (pl.col("family") == "WR") & (pl.col("team_game_num") == 10)).row(0, named=True)
    t = b.PlayerTarget(row["player_id"], row["gameday"], 2024, 10, row["opponent"], "WR", row["slot"])
    after = plog.filter(pl.col("gameday") >= t.gameday)
    scrambled = pl.concat([plog.filter(pl.col("gameday") < t.gameday),
                           after.with_columns([pl.col(col) * 0 + 12345.0 for col in b.PLAYER_MARKETS.values()])])
    for m in b.METHODS:
        assert b.predict_player(m, plog, t) == b.predict_player(m, scrambled, t)

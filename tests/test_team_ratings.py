"""4b.1 team ratings: walk-forward only, league-centred, deterministic, ridge sanity, coaching hook."""
import numpy as np
import polars as pl
import pytest

import config
from features import team_ratings as tr

db = pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="raw database not present")


@pytest.fixture(scope="module")
def rows():
    return tr.load_play_rows(max_season=2022)


def test_ridge_recovers_planted_ratings():
    rng = np.random.default_rng(0)
    T = 8
    o_true, d_true = rng.normal(0, 0.1, T), rng.normal(0, 0.1, T)
    o_true -= o_true.mean()
    d_true -= d_true.mean()
    off, de = [], []
    for a in range(T):
        for b in range(T):
            if a != b:
                off += [a] * 20
                de += [b] * 20
    off, de = np.array(off), np.array(de)
    home = rng.choice([-0.5, 0.5], len(off))
    y = 0.02 + 0.03 * home + o_true[off] + d_true[de] + rng.normal(0, 0.05, len(off))
    mu, h, o, d = tr.ridge_fit(off, de, home, y, np.ones(len(off)), T, lam=1.0)
    assert abs(o.sum()) < 1e-9 and abs(d.sum()) < 1e-9
    assert np.corrcoef(o, o_true)[0, 1] > 0.95 and np.corrcoef(d, d_true)[0, 1] > 0.95
    assert mu == pytest.approx(0.02, abs=0.01) and h == pytest.approx(0.03, abs=0.02)


@db
def test_ratings_for_week_w_use_only_games_before_w(rows):
    target = (2021, 9)
    full = tr.compute_ratings(rows, n_boot=20)
    cut = rows.filter((pl.col("season") < target[0]) | ((pl.col("season") == target[0]) & (pl.col("week") < target[1])))
    part = tr.compute_ratings(cut, n_boot=20)
    cols = [*tr.RATINGS, *[f"{r}_sd" for r in tr.RATINGS], "mu_pass", "mu_rush", "home_pass", "home_rush", "n_games"]
    a = full.filter((pl.col("season") == target[0]) & (pl.col("week") == target[1])).sort("team").select("team", *cols)
    b = part.filter((pl.col("season") == target[0]) & (pl.col("week") == target[1])).sort("team").select("team", *cols)
    assert a.height == 32 and a.equals(b)
    # and the week-1 ratings of a season do not depend on that season at all
    w1 = full.filter((pl.col("season") == 2022) & (pl.col("week") == 1)).sort("team").select("team", *tr.RATINGS)
    no22 = tr.compute_ratings(rows.filter(pl.col("season") < 2022), n_boot=20)
    final21 = no22.filter((pl.col("season") == 2021) & pl.col("is_final")).sort("team")
    for r in tr.RATINGS:   # week 1 = carryover x last season's final ratings (re-centred: already centred)
        assert np.allclose(w1[r].to_numpy(), tr.CARRYOVER * final21[r].to_numpy(), atol=1e-12)


@db
def test_offense_and_defense_ratings_sum_to_zero_across_the_league(rows):
    r = tr.compute_ratings(rows, n_boot=0)
    sums = r.group_by("season", "week").agg([pl.col(c).sum().abs().alias(c) for c in tr.RATINGS])
    for c in tr.RATINGS:
        assert sums[c].max() < 1e-9, c


@db
def test_identical_runs_give_identical_ratings(rows):
    a = tr.compute_ratings(rows, n_boot=20)
    b = tr.compute_ratings(rows, n_boot=20)
    assert a.equals(b)
    # bootstrap SDs exist everywhere except the very first history week (no data, league-average start, nothing to resample)
    first = (pl.col("season") == a["season"].min()) & (pl.col("week") == 1)
    assert a.filter(~first)["off_pass_sd"].min() > 0


@db
def test_coaching_hook_defaults_to_nothing_and_pulls_the_start_when_given(rows):
    base = tr.compute_ratings(rows, n_boot=0)
    hooked = tr.compute_ratings(rows, n_boot=0, coaching_prior={("KC", 2021): {"off_pass": (0.30, 6.0)}})
    assert tr.compute_ratings(rows, n_boot=0, coaching_prior=None).equals(base)
    get = lambda df, w: df.filter((pl.col("season") == 2021) & (pl.col("week") == w) & (pl.col("team") == "KC"))["off_pass"][0]
    assert get(hooked, 1) > get(base, 1)                   # pulled toward the supplied mean ...
    other = lambda df: df.filter((pl.col("season") == 2020)).sort("team", "week")["off_pass"].to_numpy()
    assert np.array_equal(other(hooked), other(base))       # ... and nothing changes before that season


def test_no_market_line_is_read():
    import inspect
    src = inspect.getsource(tr.load_play_rows)
    assert not any(c in src for c in tr.FORBIDDEN_COLUMNS)

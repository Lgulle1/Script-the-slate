"""4c.1 fingerprint vectors: windows, standardisation, lineup versions, no-leakage."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import config
from features import weights as wts
from models import comps as C
from models import comps_spec as cs

TEAMS = list("ABCDEFGH")
SEASONS = (2019, 2020)
WEEKS = 10
TEAM_KEYS = [k for u in C.TEAM_UNITS for k in C.feature_keys(u)]


def _games() -> pl.DataFrame:
    rows = []
    for s in SEASONS:
        start = date(s, 9, 8)
        for w in range(1, WEEKS + 1):
            order = TEAMS[w % 8:] + TEAMS[:w % 8]
            for a, b in zip(order[0::2], order[1::2]):
                gid = f"{s}_{w:02d}_{a}_{b}"
                day = start + timedelta(days=7 * (w - 1))
                rows.append(dict(game_id=gid, season=s, week=w, gameday=day, team=b, opponent=a, is_home=True))
                rows.append(dict(game_id=gid, season=s, week=w, gameday=day, team=a, opponent=b, is_home=False))
    g = pl.DataFrame(rows).sort("team", "gameday").with_columns(team_game_num=pl.col("gameday").rank("ordinal").over("team", "season").cast(pl.Int32))
    return g.sort("season", "week", "game_id", "team")


def synthetic_inputs(seed: int = 0, perturb_from: tuple | None = None, kill_units=(), perturb_lineups: bool = True, with_pregame: bool = True) -> C.Inputs:
    """A small league (8 teams, 2 seasons x 10 weeks) with random ledgers. `perturb_from=(season, week)` redraws every stat, lineup, slot and 4a
    expectation of the games at or after that week with a different seed, leaving everything earlier untouched. `kill_units` zero the ledger of a unit.
    `with_pregame` adds the pregame depth chart and the scored targets (as in production)."""
    games = _games()
    key = games["season"] * 100 + games["week"]
    late = (key >= perturb_from[0] * 100 + perturb_from[1]).to_numpy() if perturb_from else np.zeros(games.height, dtype=bool)
    # one stream per section and per version of the data: an early game draws the same numbers whether or not later games are redrawn
    def streams(section):
        return np.random.default_rng([seed, section]), np.random.default_rng([seed + 1000, section])
    rng, alt = streams(0)

    def draw(n, lo, hi, which):
        base, other = rng.uniform(lo, hi, n), alt.uniform(lo, hi, n)
        return np.where(which, other, base)

    cols = {"game_id": games["game_id"], "team": games["team"]}
    for k in TEAM_KEYS:
        unit = k.split(".")[0]
        d = draw(games.height, 20, 60, late)
        n = d * draw(games.height, 0.05, 0.9, late)
        if unit in kill_units:
            d = np.zeros(games.height)
            n = np.zeros(games.height)
        cols[f"{k}|n"], cols[f"{k}|d"] = n, d
    team = pl.DataFrame(cols).with_columns(pl.lit(25.0).alias("x.carries"), pl.lit(35.0).alias("x.targets"), pl.lit(38.0).alias("x.dropbacks"))
    lineups = games.select("team", "season", "week").with_columns(
        **{c: pl.Series(draw(games.height, 0, 3, late if perturb_lineups else np.zeros(games.height, dtype=bool)).astype(int)) for c in C.LINEUP_COMPONENTS})
    # players: QB1, QB2, RB1-3, WR1-3, TE1 per team
    roster = [("QB", 1), ("QB", 2), ("RB", 1), ("RB", 2), ("RB", 3), ("WR", 1), ("WR", 2), ("WR", 3), ("TE", 1)]
    pkeys = sorted(set(C.RUN_KEYS + C.PASS_KEYS + C.EXTRA_KEYS + ["x.dropback_share"] + [f"{u}.{f.name}" for u in cs.ARCHETYPE_UNITS for f in cs.FEATURES[u]]))
    prng, palt = streams(1)
    prow, pos, slots = [], [], []
    shares = {"QB": [0.85, 0.15], "RB": [0.6, 0.3, 0.1], "WR": [0.45, 0.35, 0.2], "TE": [1.0]}
    for gi, g in enumerate(games.iter_rows(named=True)):
        r = palt if late[gi] else prng
        for i, (p, slot) in enumerate(roster):
            pid = f"{g['team']}{p}{slot}"
            if p == "QB" and slot == 2 and r.random() > 0.3:
                continue
            if p == "QB" and slot == 1 and r.random() < 0.08:
                continue
            w = {"QB": 0.0, "RB": 25 * r.dirichlet(np.ones(3) * 3)[slot - 1], "WR": 0, "TE": 0}[p] if p == "RB" else 0.0
            rec = {"QB": 0.0, "RB": 0.15, "WR": 0.3, "TE": 0.2}[p] * 35 * r.uniform(0.3, 1.7)
            row = dict(game_id=g["game_id"], team=g["team"], player_id=pid, season=g["season"], week=g["week"], gameday=g["gameday"],
                       team_game_num=g["team_game_num"], opponent=g["opponent"], position=p, group={"RB": "rb", "WR": "wr", "TE": "te"}.get(p),
                       **{"c.carries": float(round(w)) if p == "RB" else (float(r.integers(0, 4)) if p == "QB" else 0.0),
                          "c.targets": float(round(rec)), "c.dropbacks": float(round(38 * r.uniform(0.7, 1.0))) if p == "QB" else 0.0,
                          "c.air_yards": float(round(rec * 8)), "c.targets_air": float(round(rec)), "c.gl_carries": 0.0, "offense_pct": float(r.uniform(0.2, 1))})
            for k in pkeys:
                d = float(r.uniform(2, 30))
                row[f"{k}|d"], row[f"{k}|n"] = d, d * float(r.uniform(0.1, 0.9))
            prow.append(row)
            pos.append(dict(gsis_id=pid, season=g["season"], week=g["week"], position=p, height=70.0 + i, weight=200.0 + i))
            slots.append(dict(gsis_id=pid, season=g["season"], week=g["week"], slot=slot, family=p))
    player = pl.DataFrame(prow)
    positions = pl.DataFrame(pos).unique(["gsis_id", "season", "week"]).with_columns(key=pl.col("season") * 100 + pl.col("week")).sort("key")
    slots_df = pl.DataFrame(slots).unique(["gsis_id", "season", "week"])
    # 4a expectations: baseline shares b_*, expected shares exp_* (a redistribution when a starter is "out")
    drows = []
    drng, dalt = streams(2)
    for gi, g in enumerate(games.iter_rows(named=True)):
        r = dalt if late[gi] else drng
        for p, slot in roster:
            pid = f"{g['team']}{p}{slot}"
            b = {"QB": [0.85, 0.15], "RB": [0.6, 0.3, 0.1], "WR": [0.45, 0.35, 0.2], "TE": [1.0]}[p][slot - 1]
            out = r.random() < 0.15 and slot == 1
            drows.append(dict(game_id=g["game_id"], team=g["team"], player_id=pid, b_carry=b * 0.9 if p == "RB" else 0.0, b_target=b * 0.8 if p != "QB" else 0.0,
                              b_dropback=b if p == "QB" else 0.0, exp_carry=0.0 if (out and p == "RB") else (b * 0.9 if p == "RB" else 0.0),
                              exp_target=0.0 if out else (b * 0.8 if p != "QB" else 0.0), exp_dropback=0.0 if (out and p == "QB") else (b if p == "QB" else 0.0)))
    detail = pl.DataFrame(drows)
    # the pregame depth chart lists the whole roster every week (whether or not a player then plays); the scored targets are the QB1 / RB1 / WR1 / TE1
    # of every 2020 game, also when he did not play
    ros = pl.DataFrame([dict(family=p, slot=s) for p, s in roster])
    depth = (games.select("team", "season", "week").unique().join(ros, how="cross")
             .with_columns(gsis_id=pl.col("team") + pl.col("family") + pl.col("slot").cast(pl.String)).select("team", "season", "week", "gsis_id", "family", "slot")
             .sort("team", "season", "week", "gsis_id"))
    targets = (games.filter(pl.col("season") == 2020).select("game_id", "team").join(ros.filter(pl.col("slot") == 1), how="cross")
               .select("game_id", "team", player_id=pl.col("team") + pl.col("family") + "1", family="family").sort("game_id", "team", "player_id"))
    return C.Inputs(games=games, lineups=lineups, team=team, player=player, slots=slots_df, positions=positions, detail=detail,
                    depth_chart=depth if with_pregame else None, targets=targets if with_pregame else None)


@pytest.fixture(scope="module")
def vec():
    return C.build_vectors(synthetic_inputs(0))


# --------------------------------------------------------------------------------------------------------------------------------- windows
def _timeline(days, seasons, game_nums, lin=None, slot=0, fam=0, team=0):
    n = len(days)
    return C.Timeline(np.array(days), C.game_clock(np.array(seasons), np.array(game_nums)), np.array(seasons),
                      np.array(lin if lin is not None else [[-1] * n] * 4), np.full(n, slot), np.full(n, fam), np.full(n, team))


def test_clock_matches_games_elapsed():
    for past, target in (((2019, 10), (2019, 13)), ((2019, 16), (2020, 2)), ((2018, 16), (2021, 3)), ((2020, 16), (2021, 1))):
        d = C.game_clock(np.array([target[0]]), np.array([target[1]])) - C.game_clock(np.array([past[0]]), np.array([past[1]]))
        assert d[0] == wts.games_elapsed(past[0], past[1], target[0], target[1])


def test_window_weights_pick_the_right_games():
    hist = _timeline([10, 20, 30, 40, 50], [2019] * 4 + [2020], [1, 2, 3, 4, 1])
    query = _timeline([45], [2020], [1])                       # cutoff 45: games on days 10..40 are before it, day 50 is not
    assert C.window_weights(hist, query, ("last_3", None))[0].tolist() == [0, 1, 1, 1, 0]
    assert C.window_weights(hist, query, ("last_6", None))[0].tolist() == [1, 1, 1, 1, 0]
    assert C.window_weights(hist, query, ("season_to_date", None))[0].tolist() == [0, 0, 0, 0, 0]       # nothing yet in 2020
    rec = C.window_weights(hist, query, ("recency_weighted", None))[0]
    assert rec[4] == 0
    for j in range(4):
        assert rec[j] == pytest.approx(wts.recency_weight(wts.games_elapsed(2019, [1, 2, 3, 4][j], 2020, 1)))


def test_cutoff_excludes_the_target_week_and_later():
    hist = _timeline([10, 20, 30], [2019] * 3, [1, 2, 3])
    query = _timeline([30], [2019], [3])                       # the game on day 30 is in the target week: not in its window
    for v in C.VARIANTS:
        assert C.window_weights(hist, query, v)[0][2] == 0


def test_continuity_weight_penalises_a_changed_qb():
    hist = _timeline([10, 20], [2019, 2019], [1, 2], lin=[[1, 2], [5, 5], [6, 6], [7, 7]])
    query = _timeline([30], [2019], [3], lin=[[2], [5], [6], [7]])      # QB group 2 now: game 1 had group 1
    cont = C.window_weights(hist, query, ("continuity_weighted", "pass_yds"))[0]
    rec = C.window_weights(hist, query, ("recency_weighted", None))[0]
    assert cont[1] == pytest.approx(rec[1]) and cont[0] == pytest.approx(rec[0] * config.CONTINUITY_PENALTIES["pass_yds"]["QB"])


def test_a_team_change_flags_every_factor():
    hist = C.Timeline(np.array([10]), C.game_clock(np.array([2019]), np.array([1])), np.array([2019]), np.array([[1], [1], [1], [1]]), np.array([1]), np.array([2]), np.array([0]))
    query = C.Timeline(np.array([30]), C.game_clock(np.array([2019]), np.array([2])), np.array([2019]), np.array([[1], [1], [1], [1]]), np.array([1]), np.array([2]), np.array([1]))
    cont = C.window_weights(hist, query, ("continuity_weighted", "rush_att"))[0][0]
    rec = C.window_weights(hist, query, ("recency_weighted", None))[0][0]
    assert cont < rec * 0.5


# --------------------------------------------------------------------------------------------------------------------------------- standardisation
def test_league_z_uses_only_its_own_week_and_clips():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 1))
    values[0, 0] = 100.0
    den = np.full((40, 1), 50.0)
    keys = np.repeat([201901, 201902], 20)
    z, mu, sd = C.league_z(values, den, keys)
    assert mu[0, 0] == pytest.approx(values[:20, 0].mean()) and mu[25, 0] == pytest.approx(values[20:, 0].mean())
    assert z[0, 0] == C.Z_CLIP                                      # clipped
    values2 = values.copy()
    values2[20:, 0] += 5.0                                          # changing the later week leaves the earlier week's z alone
    z2, _, _ = C.league_z(values2, den, keys)
    assert np.array_equal(z[:20], z2[:20])


def test_league_z_is_missing_without_a_league():
    z, mu, sd = C.league_z(np.arange(5.0)[:, None], np.full((5, 1), 10.0), np.zeros(5, dtype=int))
    assert np.isnan(z).all() and np.isnan(mu).all()                 # fewer than MIN_LEAGUE_ENTITIES teams: no standardisation, not zeros


def test_pooled_rate_is_missing_not_zero_without_opportunities():
    rate, den = C.pooled(np.array([[1.0, 1.0]]), np.array([[0.0], [0.0]]), np.array([[0.0], [0.0]]))
    assert np.isnan(rate[0, 0]) and den[0, 0] == 0


# --------------------------------------------------------------------------------------------------------------------------------- vectors
def test_vectors_exist_for_every_unit_space_and_version(vec):
    t = vec.team
    assert set(t["unit"].unique()) == set(C.TEAM_UNITS)
    assert set(t["version"].unique()) == {"healthy", "adjusted", "actual"}
    for unit in C.TEAM_UNITS:
        assert set(t.filter(pl.col("unit") == unit)["space"].unique()) == set(C.stored_spaces(unit))
    assert set(vec.player["unit"].unique()) == set(cs.ARCHETYPE_UNITS)
    assert set(t["window"].unique()) == set(C.vname(v) for v in C.VARIANTS)


def test_every_vector_is_stored_with_its_as_of_cutoff_and_aligned_lists(vec):
    t = vec.team
    assert t["as_of"].null_count() == 0
    meta = vec.features
    width = {(r["unit"], r["space"]): r["n"] for r in meta.group_by("unit", "space").agg(n=pl.len()).iter_rows(named=True)}
    sample = t.group_by("unit", "space").agg(pl.col("z").list.len().first()).iter_rows()
    for unit, space, n in sample:
        assert n == width[(unit, space)]
    assert set(meta["quality"].unique()) <= {"observed", "derived", "estimated"}
    assert {"source", "quality", "feature", "first_season"} <= set(meta.columns)


def test_as_of_cutoff_is_the_first_kickoff_of_the_week(vec):
    games = _games()
    first = games.group_by("season", "week").agg(c=pl.col("gameday").min())
    j = vec.team.select("season", "week", "as_of").unique().join(first, on=["season", "week"])
    assert (j["as_of"] == j["c"]).all()


def test_earliest_games_have_missing_vectors_not_zero(vec):
    first = vec.team.filter((pl.col("season") == 2019) & (pl.col("week") == 1) & (pl.col("version") == "healthy") & (pl.col("window") == "last_3"))
    assert first.height > 0
    assert first["n_present"].max() == 0 and first["z"].list.eval(pl.element().is_null()).list.all().all()


def test_a_unit_with_no_source_returns_missing_not_zeros():
    inp = synthetic_inputs(0, kill_units=("coverage_mix",))
    v = C.build_vectors(inp)
    rows = v.team.filter((pl.col("unit") == "coverage_mix") & (pl.col("version") == "healthy"))
    assert rows.height > 0 and rows["n_present"].max() == 0 and not rows["complete"].any()
    flat = rows["z"].explode()
    assert flat.null_count() == flat.len()


def test_extended_is_complete_only_when_every_extended_feature_is_present():
    inp = synthetic_inputs(0)
    team = inp.team.with_columns(pl.lit(0.0).alias("pass_offense.play_action_rate|d"))         # FTN column absent for every game
    v = C.build_vectors(C.Inputs(**{**inp.__dict__, "team": team}))
    ext = v.team.filter((pl.col("unit") == "pass_offense") & (pl.col("space") == "extended") & (pl.col("version") == "healthy") & (pl.col("window") == "recency_weighted"))
    base = v.team.filter((pl.col("unit") == "pass_offense") & (pl.col("space") == "base") & (pl.col("version") == "healthy") & (pl.col("window") == "recency_weighted"))
    assert not ext["complete"].any() and base["complete"].any()


def test_quality_tags_cover_every_feature():
    meta = C.feature_meta()
    assert set(meta["quality"].unique()) <= {"observed", "derived", "estimated"}
    assert (meta.filter(pl.col("unit") == "coverage_mix")["quality"] == "estimated").all()


# --------------------------------------------------------------------------------------------------------------------------------- no leakage
def _frame_key(df):
    return df.sort(["version", "window", "unit", "space", "season", "week", "game_id", "team", "player_id"], nulls_last=True)


def test_changing_stats_scores_or_lineups_from_week_w_leaves_earlier_vectors_unchanged():
    W = (2020, 4)
    a = C.build_vectors(synthetic_inputs(0))
    b = C.build_vectors(synthetic_inputs(0, perturb_from=W))
    before = lambda df: df.filter(pl.col("season") * 100 + pl.col("week") < W[0] * 100 + W[1])
    for name in ("team", "player"):
        x, y = _frame_key(before(getattr(a, name))), _frame_key(before(getattr(b, name)))
        assert x.height == y.height and x.height > 0
        assert x.equals(y), f"{name} vectors before week {W} changed when later data changed"
    # and the perturbation really does change the later vectors
    late = lambda df: _frame_key(df.filter(pl.col("season") * 100 + pl.col("week") >= W[0] * 100 + W[1] + 1))
    assert not late(a.team).equals(late(b.team))


def test_healthy_vectors_of_week_w_itself_ignore_week_w_stats():
    """A game's own pregame depth chart is an input (it is known before kickoff, and the continuity windows compare earlier games with it), so the
    stats of week W are redrawn here and its lineups are not; the vectors of week W are unchanged in every window."""
    W = (2020, 4)
    a = C.build_vectors(synthetic_inputs(0))
    b = C.build_vectors(synthetic_inputs(0, perturb_from=W, perturb_lineups=False))
    at_w = lambda df: _frame_key(df.filter((pl.col("season") == W[0]) & (pl.col("week") == W[1]) & (pl.col("version") == "healthy")))
    assert at_w(a.team).height > 0 and at_w(a.team).equals(at_w(b.team))


def test_actual_version_reflects_who_played_not_the_touches_in_the_game():
    W = (2020, 4)
    base = synthetic_inputs(0)
    key = base.player["season"] * 100 + base.player["week"]
    noisy = base.player.with_columns(pl.when(key == W[0] * 100 + W[1]).then(pl.col("c.carries") * 3 + 1).otherwise(pl.col("c.carries")).alias("c.carries"),
                                     pl.when(key == W[0] * 100 + W[1]).then(pl.col("c.targets") * 2 + 1).otherwise(pl.col("c.targets")).alias("c.targets"))
    a = C.build_vectors(base)
    b = C.build_vectors(C.Inputs(**{**base.__dict__, "player": noisy}))
    at_w = lambda v: _frame_key(v.team.filter((pl.col("season") == W[0]) & (pl.col("week") == W[1]) & (pl.col("version") == "actual")))
    assert at_w(a).height > 0 and at_w(a).equals(at_w(b))


def test_build_is_deterministic():
    a = C.build_vectors(synthetic_inputs(3))
    b = C.build_vectors(synthetic_inputs(3))
    assert a.team.equals(b.team) and a.player.equals(b.player) and a.lineup_change.equals(b.lineup_change)


# --------------------------------------------------------------------------------------------------------------------------------- lineup versions
def test_adjusted_equals_healthy_when_expected_equals_baseline():
    inp = synthetic_inputs(0)
    d = inp.detail.with_columns(exp_carry=pl.col("b_carry"), exp_target=pl.col("b_target"), exp_dropback=pl.col("b_dropback"))
    v = C.build_vectors(C.Inputs(**{**inp.__dict__, "detail": d}))
    adj = _frame_key(v.team.filter((pl.col("version") == "adjusted") & (pl.col("window") == "recency_weighted") & (pl.col("unit") == "run_offense")))
    hl = _frame_key(v.team.filter((pl.col("version") == "healthy") & (pl.col("window") == "recency_weighted") & (pl.col("unit") == "run_offense")))
    hl = hl.join(adj.select("game_id", "team", "space"), on=["game_id", "team", "space"])
    assert adj.height == hl.height and adj.height > 0
    z1, z2 = adj["z"].to_list(), hl["z"].to_list()
    for a, b in zip(z1, z2):
        for p, q in zip(a, b):
            assert (p is None and q is None) or p == pytest.approx(q, abs=1e-5)


def test_lineup_versions_exist_only_for_units_with_a_player_decomposition(vec):
    t = vec.team
    assert set(t.filter(pl.col("version") != "healthy")["unit"].unique()) == set(C.LINEUP_UNITS)
    assert set(C.LINEUP_NOT_ADJUSTED) == set(k for u in ("ol_protection",) + cs.DEFENSE_UNITS for k in C.feature_keys(u)) | set(C.SCHEME_KEYS)
    assert not set(C.DELTA_KEYS) & set(C.LINEUP_NOT_ADJUSTED)


def test_target_and_pool_versions():
    assert C.target_version("run_offense") == "adjusted" and C.target_version("pass_rush") == "healthy"
    assert C.pool_version("rb_rotation") == "actual" and C.pool_version("run_defense") == "healthy"


def test_struct_values_top3_split_and_rotation():
    group = np.array(["rb", "rb", "wr", "wr", "te"])
    pid = np.array(["a", "b", "c", "d", "e"])
    sc = np.array([0.6, 0.3, 0.0, 0.0, 0.0])
    st = np.array([0.1, 0.05, 0.4, 0.25, 0.2])
    out = C.struct_values(group, pid, sc, st, np.array([1.0, 2.0, 8.0, 12.0, 6.0]), np.array([0.5, 0.2, 0, 0, 0]), np.array([0.7, 0.4, 0, 0, 0]))
    assert out["receiver_usage.top3_target_share"] == pytest.approx(0.85)
    assert out["receiver_usage.target_split_wr"] == pytest.approx(0.65) and out["receiver_usage.target_split_rb"] == pytest.approx(0.15)
    assert out["receiver_usage.adot_wr"] == pytest.approx((0.4 * 8 + 0.25 * 12) / 0.65)
    assert out["rb_rotation.rb1_carry_share"] == pytest.approx(0.6 / 0.9) and out["rb_rotation.rb2_carry_share"] == pytest.approx(0.3 / 0.9)
    assert out["rb_rotation.rb1_snap_share"] == 0.7 and out["rb_rotation.goal_line_carry_share"] == 0.5
    assert np.isnan(C.struct_values(group, pid, np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5))["rb_rotation.rb1_carry_share"])


# --------------------------------------------------------------------------------------------------------------------------------- retrieval change
def test_retrieval_change_overlap_and_shift():
    healthy = [("a", 0.5, 10.0), ("b", 0.3, 20.0), ("c", 0.2, 30.0)]
    same = C.retrieval_change(healthy, healthy)
    assert same["overlap_top_k"] == 1.0 and same["weight_mass_shared"] == pytest.approx(1.0) and same["outcome_shift"] == pytest.approx(0.0)
    adjusted = [("a", 0.5, 10.0), ("b", 0.3, 20.0), ("z", 0.2, 50.0)]
    ch = C.retrieval_change(healthy, adjusted, k=3)
    assert ch["overlap_top_k"] == pytest.approx(2 / 3) and ch["weight_mass_shared"] == pytest.approx(0.8)
    assert ch["outcome_shift"] == pytest.approx((5 + 6 + 10) - (5 + 6 + 6))


# --------------------------------------------------------------------------------------------------------------------------------- real data
@pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="raw database not available")
def test_real_ledger_covers_every_spec_feature_and_every_franchise():
    inp = C.load_inputs(max_season=2021, lineup_expectations=False)
    missing = [k for k in TEAM_KEYS if f"{k}|n" not in inp.team.columns or f"{k}|d" not in inp.team.columns]
    assert not missing
    assert inp.games["team"].n_unique() == 32                                  # OAK / SD fold into LV / LAC: no franchise left without plays
    carries = inp.team.join(inp.games.select("game_id", "team"), on=["game_id", "team"])["x.carries"]
    assert (carries > 0).all()
    q = inp.player.filter(pl.col("c.dropbacks") > 0)
    assert q["c.dropbacks"].sum() == inp.team["x.dropbacks"].sum()              # scrambles credited to the QB: player and team dropbacks agree
    assert max(inp.games["season"]) <= config.BACKTEST_SEASONS[-1]


# ================================================================================================================================ 4c.2 distance / similarity
def _rand(unit, space, n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, len(C.unit_features(unit, space))))


def test_identical_vectors_have_distance_zero_and_similarity_one():
    for unit, space in (("run_offense", "base"), ("pass_offense", "extended"), ("rb_archetype", "extended"), ("coverage_mix", "extended")):
        A = _rand(unit, space, 6, 1)
        d = C.unit_distance(A, A, unit, space)
        assert (np.diag(d.d2) == 0).all()
        assert (np.diag(C.similarity(d.d2, 0.8)) == 1.0).all()
        assert (np.diag(d.completeness) == 0).all()


def test_similarity_falls_as_distance_rises():
    A = _rand("run_offense", "base", 1, 2)
    sims, d2s = [], []
    for scale in (0.0, 0.2, 0.5, 1.0, 2.0):
        B = A + scale * np.sign(_rand("run_offense", "base", 1, 3))
        d2s.append(C.unit_distance(A, B, "run_offense", "base").d2[0, 0])
        sims.append(C.similarity(d2s[-1], 1.0))
    assert d2s == sorted(d2s) and sims == sorted(sims, reverse=True) and sims[0] == 1.0 and sims[-1] < sims[1]


def test_similarity_uses_d2_once_not_squared_again():
    assert C.similarity(2.0, 1.5) == pytest.approx(np.exp(-2.0 / (2 * 1.5 ** 2)))


def test_group_collapse_matches_a_hand_computation():
    unit, space = "run_offense", "base"
    names = [f.name for f in C.unit_features(unit, space)]
    A, B = np.zeros((1, len(names))), np.zeros((1, len(names)))
    B[0, names.index("rush_epa_per_play")] = 2.0                         # one feature of the 4-feature efficiency group differs by 2
    q = C.feature_quality(unit, space)
    eff = [names.index(n) for n in ("rush_epa_per_play", "rush_success_rate", "yards_per_carry", "explosive_run_rate")]
    d2_eff = (q[eff[0]] * 4.0) / q[eff].sum()
    groups = C.group_columns(unit, space)
    assert len(groups) == 5
    assert C.unit_distance(A, B, unit, space).d2[0, 0] == pytest.approx(d2_eff / 5)          # the unit distance is the equal-weight mean over the 5 groups


def test_a_missing_or_quality_zero_feature_changes_nothing_but_the_completeness_penalty():
    unit, space = "rb_archetype", "extended"
    A, B = _rand(unit, space, 1, 4), _rand(unit, space, 1, 5)
    names = [f.name for f in C.unit_features(unit, space)]
    j = names.index("yards_after_contact")
    B_missing = B.copy(); B_missing[0, j] = np.nan
    d_full = C.unit_distance(A, B_missing, unit, space)
    A_changed = A.copy(); A_changed[0, j] += 7.0                          # the feature is missing on B: its value on A must not matter
    d_changed = C.unit_distance(A_changed, B_missing, unit, space)
    assert d_changed.d2[0, 0] == d_full.d2[0, 0] and d_changed.completeness[0, 0] == d_full.completeness[0, 0] > 0
    zero = {"yards_per_carry": 0.0}
    A2 = A.copy(); A2[0, names.index("yards_per_carry")] += 9.0
    assert C.unit_distance(A2, B, unit, space, quality=zero).d2[0, 0] == pytest.approx(C.unit_distance(A, B, unit, space, quality=zero).d2[0, 0])
    assert C.unit_distance(A2, B, unit, space).d2[0, 0] != pytest.approx(C.unit_distance(A, B, unit, space).d2[0, 0])


def test_quality_weights_discount_estimated_features():
    unit, space = "coverage_mix", "extended"
    assert C.feature_quality(unit, space).tolist() == [cs.QUALITY["estimated"]] * 3
    assert C.feature_quality("run_offense", "base")[0] == cs.QUALITY["observed"] and C.feature_quality("rb_rotation", "base")[0] == cs.QUALITY["derived"]


def test_distance_is_missing_when_nothing_can_be_compared():
    A = np.full((1, len(C.unit_features("pass_rush", "extended"))), np.nan)
    d = C.unit_distance(A, _rand("pass_rush", "extended", 2, 6), "pass_rush", "extended")
    assert np.isnan(d.d2).all() and (d.completeness == 1.0).all()


def test_choose_space():
    assert C.choose_space("pass_offense", True, True) == "extended" and C.choose_space("pass_offense", True, False) == "base"
    assert C.choose_space("run_offense", True, True) == "base"                               # no extended features: BASE only
    assert C.choose_space("coverage_mix", False, True) is None and C.choose_space("coverage_mix", True, True) == "extended"


def test_groups_partition_every_unit():
    for unit in cs.UNITS:
        assert sorted(f for _, fs in cs.GROUPS[unit] for f in fs) == sorted(f.name for f in cs.FEATURES[unit])


@pytest.fixture(scope="module")
def sigma_pair():
    W = (2020, 4)
    a, b = C.build_vectors(synthetic_inputs(0)), C.build_vectors(synthetic_inputs(0, perturb_from=W))
    return a, b, C.build_sigma(a.team, a.player), C.build_sigma(b.team, b.player), W


def test_sigma_is_walk_forward(sigma_pair):
    a, b, sa, sb, W = sigma_pair
    key = lambda df: df["season"] * 100 + df["week"]
    early = lambda df: df.filter(key(df) <= W[0] * 100 + W[1]).sort("unit", "window", "space", "season", "week")
    assert early(sa).height > 0 and early(sa).equals(early(sb))           # sigma at weeks up to W uses only targets before them
    first = sa.filter((pl.col("season") == 2019) & (pl.col("week") == 1))
    assert first["sigma"].null_count() == first.height                      # no earlier targets: sigma is missing, not a guess
    s = sa.filter((pl.col("unit") == "run_offense") & (pl.col("window") == "recency_weighted") & (pl.col("space") == "base")).sort("season", "week")
    assert s["n_targets"].is_sorted() and s["sigma"].drop_nulls().len() > 0
    assert (s["sigma"].drop_nulls() > 0).all()


def test_similarities_before_week_w_ignore_later_data(sigma_pair):
    a, b, sa, sb, W = sigma_pair
    unit, space, window = "run_offense", "base", "recency_weighted"
    nf = len(C.unit_features(unit, space))
    pick = lambda v: v.team.filter((pl.col("unit") == unit) & (pl.col("space") == space) & (pl.col("window") == window) & (pl.col("version") == "actual")
                                   & (pl.col("season") * 100 + pl.col("week") < W[0] * 100 + W[1])).sort("season", "week", "game_id", "team")
    ra, rb = pick(a), pick(b)
    Za, Zb = C.vector_matrix(ra, nf), C.vector_matrix(rb, nf)
    sig = C.sigma_at(sa, unit, window, space, 2020, 3)
    sim_a = C.similarity(C.unit_distance(Za, Za, unit, space).d2, sig)
    sim_b = C.similarity(C.unit_distance(Zb, Zb, unit, space).d2, C.sigma_at(sb, unit, window, space, 2020, 3))
    assert np.array_equal(sim_a, sim_b, equal_nan=True) and np.isfinite(sig)


def test_sigma_excludes_a_targets_own_team():
    from models.comps import _sigma_block
    inp = synthetic_inputs(0)
    v = C.build_vectors(inp)
    df = v.team.filter((pl.col("unit") == "run_offense") & (pl.col("space") == "base") & (pl.col("window") == "recency_weighted") & (pl.col("version") == "actual"))
    keys, d_ex, d_in = _sigma_block(df, "run_offense", "base", "team", None)
    assert len(keys) > 0 and (d_ex >= d_in - 1e-12).all()              # removing the own team's near-duplicate games can only push the 20th neighbour out


# ================================================================================================================================ 4c.3 searches
W3 = "recency_weighted"
RUN4 = "S4:run_offense*run_defense"          # the run pair of S4 (rush, QB rush and game markets)
RB3 = "S3:rb_archetype*run_defense"           # S3 of a back against the run defense


def _pool(inp=None, vec=None, seed=0, perturb_from=None):
    inp = inp or synthetic_inputs(seed, perturb_from=perturb_from)
    vec = vec or C.build_vectors(inp)
    sigma = C.build_sigma(vec.team, vec.player)
    return C.Pool(vec.team, vec.player, None, inp.games, inp.lineups, inp.slots, sigma, W3), inp, vec


@pytest.fixture(scope="module")
def pool3():
    return _pool()


def _team_target(inp, season=2020, week=8, market="spread"):
    g = inp.games.filter((pl.col("season") == season) & (pl.col("week") == week)).row(0, named=True)
    return C.Target(market, g["game_id"], g["team"], g["opponent"], season, week)


def _player_target(inp, market="rush_att", season=2020, week=8):
    r = inp.player.filter((pl.col("season") == season) & (pl.col("week") == week) & (pl.col("position") == "RB")).row(0, named=True)
    return C.Target(market, r["game_id"], r["team"], r["opponent"], season, week, r["player_id"])


def test_a_planted_near_duplicate_is_retrieved_first():
    inp = synthetic_inputs(0)
    vec = C.build_vectors(inp)
    tg = _team_target(inp)
    g = inp.games.filter((pl.col("game_id") == tg.game_id)).select("game_id", "team", "opponent")
    # plant: a 2019 team-game whose offense units equal the target's lineup-adjusted ones and whose opposing defense equals tonight's opponent
    H = inp.games.filter((pl.col("season") == 2019) & (pl.col("week") == 6) & ~pl.col("team").is_in([tg.team, tg.opponent])
                         & ~pl.col("opponent").is_in([tg.team, tg.opponent])).row(0, named=True)      # a game of two other teams: S4 looks at other entities
    team_df = vec.team
    plants = []
    for unit in C.TEAM_UNITS:
        if unit in cs.OFFENSE_UNITS and unit in C.MARKET_UNITS_TEAM:
            src = team_df.filter((pl.col("game_id") == tg.game_id) & (pl.col("team") == tg.team) & (pl.col("unit") == unit) & (pl.col("version") == C.target_version(unit)))
            dst_version, dst_key = C.pool_version(unit), (H["game_id"], H["team"])
        elif unit in cs.DEFENSE_UNITS and unit in C.MARKET_UNITS_TEAM:
            src = team_df.filter((pl.col("game_id") == tg.game_id) & (pl.col("team") == tg.opponent) & (pl.col("unit") == unit) & (pl.col("version") == "healthy"))
            dst_version, dst_key = "healthy", (H["game_id"], H["opponent"])
        else:
            continue
        plants.append((unit, dst_version, dst_key, src.select("space", "window", "z", "raw", "n", "n_present", "complete")))
    t2 = team_df
    for unit, version, (gid, tm), src in plants:
        keep = t2.filter(~((pl.col("game_id") == gid) & (pl.col("team") == tm) & (pl.col("unit") == unit) & (pl.col("version") == version)))
        old = t2.filter((pl.col("game_id") == gid) & (pl.col("team") == tm) & (pl.col("unit") == unit) & (pl.col("version") == version)).drop("z", "raw", "n", "n_present", "complete")
        new = old.join(src, on=["space", "window"], how="inner")
        t2 = pl.concat([keep, new.select(keep.columns)])
    vec2 = C.Vectors(t2, vec.player, vec.features, vec.completeness, vec.lineup_change)
    pool, _, _ = _pool(inp, vec2)
    res = pool.search(tg, which=("S4",))[RUN4]
    assert res.matches is not None and res.matches.height > 0
    top = res.matches.row(0, named=True)
    assert (top["obs_game_id"], top["obs_team"]) == (H["game_id"], H["team"])
    assert top["sim_combined"] == pytest.approx(1.0, abs=1e-9)
    assert res.matches["final_weight"][0] > 0


def test_nothing_above_threshold_returns_no_match(pool3):
    pool, inp, _ = pool3
    for tg in (_team_target(inp), _player_target(inp)):
        res = pool.search(tg, keep=True, sim_threshold=0.9999999)
        for s, r in res.items():
            if not r.summary["applicable"]:
                continue
            assert r.summary["no_match"] and r.summary["shift"] == 0.0 and r.summary["widened_uncertainty"] and r.summary["n_matches"] == 0
            assert r.matches is None or r.matches.height == 0


def test_s5_is_no_match_when_its_ftn_features_are_missing(pool3):
    pool, inp, _ = pool3                                   # the synthetic pool has no S5 results-side statistics: required FTN features are missing
    r = pool.search(_team_target(inp), which=("S5",))["S5"]
    assert r.summary["no_match"] and r.summary["reason"] in ("missing_required_ftn_features", "no_similarity") and r.summary["shift"] == 0.0
    r = pool.search(_player_target(inp), which=("S5",))["S5"]
    assert r.summary["no_match"]


def test_s3_is_not_applicable_to_team_markets(pool3):
    pool, inp, _ = pool3
    r = pool.search(_team_target(inp), which=("S3",))["S3"]
    assert not r.summary["applicable"] and not r.summary["no_match"] and r.summary["reason"] == "not_applicable"
    assert all(r.summary["applicable"] for r in pool.search(_player_target(inp), which=("S3",)).values())


def test_matches_have_separate_weight_columns_and_the_final_weight_is_their_product(pool3):
    pool, inp, _ = pool3
    found = 0
    for tg in (_team_target(inp, week=9), _player_target(inp, week=9), _player_target(inp, "pass_att", week=9)):
        for s, r in pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0).items():
            m = r.matches
            if m is None or m.height == 0:
                continue
            found += 1
            for c in ("sim_combined", "recency_weight", "continuity_weight", "quality_weight", "final_weight", "completeness_penalty"):
                assert c in m.columns
            assert (m["final_weight"] - m["sim_combined"] * m["recency_weight"] * m["continuity_weight"] * m["quality_weight"]).abs().max() < 1e-12
            assert any(c.startswith("sim_") and c != "sim_combined" for c in m.columns)
            assert ((m["continuity_weight"] > 0) & (m["continuity_weight"] <= 1.0)).all()          # continuity from weights.py, every search (plan 4c.4)
            other = m.filter(pl.col("obs_team") != tg.team)                         # another team's game flags every factor: the product of the whole penalty row
            if other.height:
                full = np.prod(list(config.CONTINUITY_PENALTIES[config.BASELINE_TO_PENALTY_MARKET[tg.market]].values()))
                assert np.allclose(other["continuity_weight"].to_numpy(), full)
            if s in ("S2", "S4"):                                                   # two-sided: the similarity is offense x defense
                assert np.allclose(m["sim_combined"].to_numpy(), (m["sim_offense"] * m["sim_defense"]).to_numpy())
            if s == "S1":
                assert m["sim_offense"].is_nan().all() and np.allclose(m["sim_combined"].to_numpy(), m["sim_defense"].to_numpy())
    assert found >= 4


def test_n_eff_is_the_kish_effective_sample_size_of_the_final_weights_per_historical_team_game(pool3):
    pool, inp, _ = pool3
    for tg in (_player_target(inp, week=9), _team_target(inp, week=9)):
        r = pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0)[RUN4]
        w = r.matches.group_by("obs_game_id", "obs_team").agg(pl.col("final_weight").sum())["final_weight"].to_numpy()
        assert r.summary["n_eff"] == pytest.approx(w.sum() ** 2 / (w ** 2).sum())
        assert r.summary["n_matches"] == r.matches.height and r.summary["best_similarity"] == pytest.approx(r.matches["sim_check"].max())


def test_cluster_neff_counts_one_team_game_once():
    w = np.ones(5)
    assert C.cluster_neff(w, ["g1"] * 5) == pytest.approx(1.0)                        # five receivers of one past game: one game's evidence
    assert C.cluster_neff(w, [f"g{i}" for i in range(5)]) == pytest.approx(5.0)          # one observation per game: the plain Kish n_eff
    assert C.cluster_neff(np.array([1.0, 1.0, 2.0]), ["a", "a", "b"]) == pytest.approx(2.0)
    assert C.cluster_neff(np.array([]), []) == 0.0


def test_s5_player_matches_of_one_team_game_do_not_inflate_n_eff(pool3):
    """S5 compares team-game matchups, so every player of a past team-game has the same S5 similarity: n_eff must count the team-game once."""
    pool, inp, _ = pool3
    tg = _player_target(inp, "targets", week=9)
    k = pool._kmax(tg.season * 100 + tg.week)
    saved = (pool.prof_off_t.copy(), pool.prof_off_p.copy(), pool.prof_def.copy(), pool.admit5.copy(), dict(pool.sig5), pool.prof_off_th.copy())
    try:
        rng = np.random.default_rng(3)
        pool.prof_off_t[:] = rng.normal(size=pool.prof_off_t.shape)
        pool.prof_off_th[:] = pool.prof_off_t                                   # the searches read the healthy profile
        pool.prof_off_p[:] = rng.normal(size=pool.prof_off_p.shape)
        pool.prof_def[:] = rng.normal(size=pool.prof_def.shape)
        pool.admit5[:] = True
        pool.sig5 = {kk: 5.0 for kk in pool.sig5}
        pool.clear_cache()
        r = pool.search(tg, which=("S5",), keep=True, sim_threshold=0.0, min_neff=0)["S5"]
        m = r.matches
        assert m.height > m.select("obs_game_id", "obs_team").unique().height                # several receivers per past team-game
        per_game = m.group_by("obs_game_id", "obs_team").agg(pl.col("sim_combined").n_unique())
        assert (per_game["sim_combined"] == 1).all()                                          # one matchup, one similarity
        w = m.group_by("obs_game_id", "obs_team").agg(pl.col("final_weight").sum())["final_weight"].to_numpy()
        assert r.summary["n_eff"] == pytest.approx(w.sum() ** 2 / (w ** 2).sum())
        assert r.summary["n_eff"] <= m.select("obs_game_id", "obs_team").unique().height + 1e-9
    finally:
        pool.prof_off_t[:], pool.prof_off_p[:], pool.prof_def[:], pool.admit5[:] = saved[:4]
        pool.sig5, pool.prof_off_th[:] = saved[4], saved[5]
        pool.clear_cache()


def test_no_match_is_declared_for_a_low_n_eff_even_with_a_good_best_similarity(pool3):
    pool, inp, _ = pool3
    r = pool.search(_player_target(inp, week=9), keep=True, sim_threshold=0.0, min_neff=1e9)[RUN4]
    assert r.summary["no_match"] and r.summary["reason"] == "n_eff_below_minimum" and r.summary["shift"] == 0.0 and r.summary["best_similarity"] > 0


def test_the_pool_never_contains_the_target_game_or_anything_after(pool3):
    pool, inp, _ = pool3
    for tg in (_team_target(inp, week=9), _player_target(inp, week=9)):
        key = tg.season * 100 + tg.week
        for s, r in pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0).items():
            if r.matches is not None and r.matches.height:
                assert ((r.matches["obs_season"] * 100 + r.matches["obs_week"]) < key).all()


def test_search_observation_sets_follow_the_plan(pool3):
    """S1: the target's own past games; S2: past games against tonight's defense; S3 ("on any team") and S4: every past observation, the player's own
    games and tonight's opponent's included (build plan 4c.0.6)."""
    pool, inp, _ = pool3
    tg = _player_target(inp, week=10)
    res = pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0, obs_positions=True)
    s1, s2 = res["S1:run_defense"].matches, res["S2:run_offense"].matches
    assert (s1["obs_player_id"] == tg.player_id).all() and s1.height > 0                # S1: the player's own history
    opp_of = lambda m: [inp.games.filter((pl.col("game_id") == g) & (pl.col("team") == t))["opponent"][0] for g, t in zip(m["obs_game_id"], m["obs_team"])]
    assert all(o == tg.opponent for o in opp_of(s2))                                   # S2: tonight's defense
    # the observation sets themselves (positions in the archetype population): every unit search of a search shares its search's set
    pos = lambda u: set(res[u].summary["obs_pos"].tolist())
    assert pos(RB3) == pos(RUN4) and pos("S1:run_defense") < pos(RUN4) and pos("S2:run_offense") < pos(RUN4)      # S3 / S4 exclude nobody
    assert pos("S2:run_offense") == pos("S2:rb_archetype") and pos(RB3) == pos("S5")
    ttg = _team_target(inp, week=10)
    tt = pool.search(ttg, which=("S4",), keep=True, sim_threshold=0.0, min_neff=0)[RUN4].matches
    assert (tt["obs_team"] == ttg.team).any() and any(o == ttg.opponent for o in opp_of(tt))      # the team's own games and its opponent's defense too


def test_search_is_walk_forward():
    W = (2020, 5)
    pa, inp_a, _ = _pool()
    pb, inp_b, _ = _pool(perturb_from=W)
    cols = ["n_matches", "n_eff", "best_similarity", "no_match"]
    for tg_fn in (lambda i: _team_target(i, week=4), lambda i: _player_target(i, week=4), lambda i: _player_target(i, "pass_att", week=3)):
        ra = pa.search(tg_fn(inp_a), keep=True, sim_threshold=0.0, min_neff=0)
        rb = pb.search(tg_fn(inp_b), keep=True, sim_threshold=0.0, min_neff=0)
        for s in ra:
            for c in cols:
                x, y = ra[s].summary[c], rb[s].summary[c]
                assert x == y or (x != x and y != y)
            if ra[s].matches is not None and ra[s].matches.height:
                assert ra[s].matches.equals(rb[s].matches)


def test_search_is_deterministic(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, week=10)
    a = pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0)
    pool.clear_cache()
    b = pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0)
    for s in a:
        assert str(a[s].summary) == str(b[s].summary)          # str: NaN != NaN
        assert (a[s].matches is None and b[s].matches is None) or a[s].matches.equals(b[s].matches)


def test_the_search_log_counts_no_match_by_season_market_search_and_unit(pool3):
    pool, inp, _ = pool3
    targets = [_team_target(inp, week=9), _player_target(inp, week=9), _player_target(inp, "pass_att", week=9)]
    summary = C.run_search_log(pool, targets)
    log = C.aggregate_search_log(summary)
    assert {"season", "market", "search", "unit", "no_match_rate", "n_targets", "unit_below_threshold_rate"} <= set(log.columns)
    assert set(log["market"].unique()) >= {"rush_att", "rush_yds", "pass_att", "pass_cmp", "pass_yds", "spread", "moneyline", "total"}
    assert (log.filter(pl.col("search") == "S3")["market"].is_in(["spread", "total", "moneyline"]).sum()) == 0         # S3 does not apply to team markets
    assert set(log.filter(pl.col("market") == "rush_att")["unit"].unique()) >= {"combined", "run_defense", "rb_archetype"}


def test_a_player_who_played_without_a_touch_is_an_observation_and_has_a_vector():
    """A zero-target game of a receiver who played (snap counts) is an observation of 0 targets, and his target vector exists: restricting the archetype
    pool to players with a touch in the game would drop the zeros and bias every outcome upward."""
    inp = synthetic_inputs(0)
    base = inp.player.filter(pl.col("player_id") == "AWR1")
    ghost = base.filter(pl.col("season") * 100 + pl.col("week") <= 202007).with_columns(pl.lit("AGHOST").alias("player_id"))       # touches through week 7 only
    pos = inp.positions.filter(pl.col("gsis_id") == "AWR1").with_columns(pl.lit("AGHOST").alias("gsis_id"))
    slots = inp.slots.filter(pl.col("gsis_id") == "AWR1").with_columns(pl.lit("AGHOST").alias("gsis_id"))
    snaps = inp.games.filter(pl.col("team") == "A").select("game_id", "team").with_columns(gsis_id=pl.lit("AGHOST"))
    inp2 = C.Inputs(**{**inp.__dict__, "player": pl.concat([inp.player, ghost]), "positions": pl.concat([inp.positions, pos]), "slots": pl.concat([inp.slots, slots]), "snaps": snaps})
    vec = C.build_vectors(inp2)
    rows = vec.player.filter((pl.col("player_id") == "AGHOST") & (pl.col("unit") == "receiver_archetype") & (pl.col("window") == W3))
    weeks = set(zip(rows["season"].to_list(), rows["week"].to_list()))
    assert (2020, 8) in weeks and (2020, 10) in weeks                                  # games with a snap but no touch
    late = rows.filter((pl.col("season") == 2020) & (pl.col("week") == 8))
    assert late["n_present"].max() > 0                                                  # his vector comes from the touches of his earlier games
    pool = C.Pool(vec.team, vec.player, None, inp2.games, inp2.lineups, inp2.slots, C.build_sigma(vec.team, vec.player), W3)
    g = inp2.games.filter((pl.col("season") == 2020) & (pl.col("week") == 8) & (pl.col("team") == "A")).row(0, named=True)
    res = pool.search(C.Target("targets", g["game_id"], "A", g["opponent"], 2020, 8, "AGHOST"), which=("S1", "S2"), sim_threshold=0.0, min_neff=0)
    assert all(r.summary["reason"] != "no_target_vector" for r in res.values())


# ================================================================================================================================ 4c.4 residuals and shifts
def _wf_rows(seed=0, perturb_from=None):
    """A small walkforward_predictions-like table: 3 roles x 6 players, 2 seasons x 10 weeks, one quantity."""
    rng = np.random.default_rng(seed)
    alt = np.random.default_rng(seed + 99)
    rows = []
    for s in (2019, 2020):
        for wk in range(1, 11):
            late = perturb_from is not None and s * 100 + wk >= perturb_from[0] * 100 + perturb_from[1]
            r = alt if late else rng
            for i in range(6):
                exp = 10.0 + i
                act = exp + r.normal(0, 2 + i % 3)
                rows.append(dict(season=s, week=wk, game_id=f"{s}_{wk:02d}_G{i // 2}", player_id=f"P{i}", team=f"T{i}", role=["RB1", "WR1", "TE1"][i % 3],
                                 quantity="targets", expected=exp, actual=act, residual=act - exp))
    return pl.DataFrame(rows)


def test_standardized_residuals_use_only_earlier_weeks():
    W = (2020, 4)
    a = C.standardized_residuals(_wf_rows())
    b = C.standardized_residuals(_wf_rows(perturb_from=W))
    early = lambda t: t.filter(pl.col("season") * 100 + pl.col("week") <= W[0] * 100 + W[1]).sort("season", "week", "player_id")
    # z of week W itself uses only earlier sigmas; its own residual changed, so compare sigma there and z strictly before
    assert early(a)["sigma"].to_list() == early(b)["sigma"].to_list()
    strictly = lambda t: t.filter(pl.col("season") * 100 + pl.col("week") < W[0] * 100 + W[1]).sort("season", "week", "player_id")
    assert strictly(a)["z"].to_list() == strictly(b)["z"].to_list()


def test_standardized_residuals_blend_toward_the_players_own_spread():
    z = C.standardized_residuals(_wf_rows())
    r = z.filter((pl.col("player_id") == "P2") & (pl.col("season") == 2020) & (pl.col("week") == 10)).row(0, named=True)
    w = _wf_rows().filter(pl.col("season") * 100 + pl.col("week") < 202010)
    own = w.filter(pl.col("player_id") == "P2")["residual"].to_numpy()
    role = w.filter(pl.col("role") == "TE1")["residual"].to_numpy()
    lam = len(own) / (len(own) + C.Z_OWN_K)
    expect = lam * np.sqrt((own ** 2).mean()) + (1 - lam) * np.sqrt((role ** 2).mean())
    assert r["sigma"] == pytest.approx(expect) and r["z"] == pytest.approx(r["residual"] / expect)
    first = z.filter((pl.col("season") == 2019) & (pl.col("week") == 1))
    assert first["z"].null_count() == first.height                         # no earlier residuals: no sigma, no z


def test_cap_shares_holds_the_cap_and_keeps_the_total():
    w = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    g = np.array(["a", "b", "c", "d", "e", "f", "a"])
    out = C.cap_shares(w, g, 0.25)
    share = {k: out[g == k].sum() / out.sum() for k in set(g)}
    assert max(share.values()) <= 0.25 + 1e-9 and out.sum() == pytest.approx(w.sum())
    assert out[0] / out[6] == pytest.approx(w[0] / w[6])                  # weights inside a group keep their proportions
    few = C.cap_shares(np.array([3.0, 1.0]), np.array(["x", "y"]), 0.25)  # 2 groups cannot respect a 25% cap: left as they are
    assert np.allclose(few, [3.0, 1.0])
    same = C.cap_shares(np.ones(8), np.arange(8), 0.25)
    assert np.allclose(same, np.ones(8))                                   # nothing above the cap: unchanged


def test_cap_across_searches_limits_a_teams_average_share():
    rng = np.random.default_rng(1)
    per = {}
    for s in C.SEARCHES:
        teams = np.array(["DOM"] * 6 + [f"T{i}" for i in range(14)])
        per[s] = (rng.uniform(0.5, 1.0, 20) * np.where(teams == "DOM", 6.0, 1.0), np.array([f"g{i}" for i in range(20)]), teams)
    out = C.cap_across_searches(per)
    avg = sum(out[s][per[s][2] == "DOM"].sum() / out[s].sum() for s in C.SEARCHES if s not in C.CAP_AVG_EXEMPT) / len(C.SEARCHES)
    assert avg <= C.cs.TEAM_CAP_AVG_ACROSS_SEARCHES + 1e-6
    for s in C.SEARCHES:
        tg = per[s][1]
        assert max(out[s][tg == g].sum() / out[s].sum() for g in set(tg)) <= C.cs.TEAM_CAP_PER_SEARCH + 1e-9


def test_shift_is_the_shrunk_weighted_mean_z():
    w, z = np.array([1.0, 2.0, 1.0]), np.array([1.0, -1.0, 3.0])
    s, n = C.shift_value(w, z)
    assert n == pytest.approx(16 / 6) and s == pytest.approx((1 - 2 + 3) / 4 * n / (n + C.cs.SHRINK_K))
    assert C.shift_value(np.array([]), np.array([])) == (0.0, 0.0)
    assert C.shift_value(np.array([1.0]), np.array([np.nan]))[0] == 0.0     # no expectation: no shift


@pytest.fixture(scope="module")
def shifts44(pool3):
    pool, inp, vec = pool3
    tgs = [_team_target(inp, week=9), _player_target(inp, "rush_att", week=9), _player_target(inp, "targets", week=9)]
    # a z for every past observation of the synthetic league (stand-in for walkforward_predictions)
    rng = np.random.default_rng(5)
    zl = {}
    for q in ("team_plays", "pts_per_play", "rush_att", "ypc", "targets", "catch_rate", "yds_per_rec"):
        d = {}
        for r in inp.games.iter_rows(named=True):
            d[(r["game_id"], r["team"])] = float(rng.normal())
        for r in inp.player.iter_rows(named=True):
            d[(r["game_id"], r["player_id"])] = float(rng.normal())
        zl[q] = d
    counts = {q: {k: float(rng.integers(1, 15)) for k in zl[q]} for q in ("pts_per_play", "ypc", "catch_rate", "yds_per_rec")}
    return C.comp_shifts(pool, tgs, zl, sim_threshold=0.0, min_neff=0.0, counts=counts), tgs, zl


def test_shift_features_have_one_row_per_target_and_market_with_the_plan_columns(shifts44):
    (feats, matches, detail, _), tgs, _ = shifts44
    markets = sorted(feats["market"].unique().to_list())
    assert {"spread", "total", "moneyline", "rush_att", "rush_yds", "targets", "rec", "rec_yds"} <= set(markets)
    for s in C.SEARCHES:
        for c in (f"shift_vol_{s}", f"shift_eff_{s}", f"n_eff_{s}", f"nomatch_{s}", f"best_sim_{s}"):
            assert c in feats.columns
    assert feats.select("game_id", "team", "player_id", "market").is_unique().all()
    assert set(["z_vol", "z_eff", "weight_capped_vol", "final_weight", "recency_weight", "continuity_weight", "quality_weight", "sim_combined"]) <= set(matches.columns)


def test_shifts_match_a_recomputation_from_the_match_table(shifts44):
    """Each unit search's shift and n_eff recompute from its matches; each search's feature is the n_eff-weighted mean of its matching units."""
    (feats, matches, detail, _), tgs, zl = shifts44
    r = feats.filter((pl.col("market") == "rush_yds")).row(0, named=True)
    det = detail.filter((pl.col("market") == "rush_yds") & (pl.col("game_id") == r["game_id"]) & (pl.col("player_id") == r["player_id"]))
    checked = 0
    for d in det.iter_rows(named=True):
        m = matches.filter((pl.col("market") == "rush_yds") & (pl.col("search") == d["search"]) & (pl.col("target_game_id") == r["game_id"])
                           & (pl.col("target_player_id") == r["player_id"]))
        if d["nomatch"]:
            assert d["shift_vol"] == 0.0 and d["shift_eff"] == 0.0
            continue
        sv, nv = C.shift_value(m["weight_capped_vol"].to_numpy(), m["z_vol"].to_numpy(), (m["obs_game_id"] + "|" + m["obs_team"]).to_numpy())
        assert d["shift_vol"] == pytest.approx(sv) and d["n_eff_vol"] == pytest.approx(nv)
        g = m.with_columns(tg=pl.col("obs_game_id") + "|" + pl.col("obs_team")).group_by("tg").agg(pl.col("weight_capped_vol").sum()).filter(pl.col("weight_capped_vol") > 0)
        if g.height * C.cs.TEAM_CAP_PER_SEARCH >= 1:
            assert (g["weight_capped_vol"] / g["weight_capped_vol"].sum()).max() <= C.cs.TEAM_CAP_PER_SEARCH + 1e-9
        checked += 1
    assert checked >= 3
    for s in C.SEARCHES:
        u = det.filter((pl.col("parent") == s) & pl.col("applicable") & ~pl.col("nomatch"))
        if u.height == 0:
            assert r[f"nomatch_{s}"] and r[f"shift_vol_{s}"] == 0.0
            continue
        w = u["n_eff_vol"].to_numpy()
        assert not r[f"nomatch_{s}"] and r[f"shift_vol_{s}"] == pytest.approx((w * u["shift_vol"].to_numpy()).sum() / w.sum())
        all_u = det.filter((pl.col("parent") == s) & pl.col("applicable"))
        assert r[f"n_eff_{s}"] == pytest.approx(all_u["n_eff_vol"].max()) and r[f"best_sim_{s}"] == pytest.approx(all_u["best_similarity"].max())


def test_a_no_match_search_has_shift_zero(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, "rush_att", week=9)
    feats, _, detail, _ = C.comp_shifts(pool, [tg], {"rush_att": {}, "ypc": {}}, sim_threshold=0.99999999)
    for s in C.SEARCHES:
        assert feats[f"nomatch_{s}"].all() and (feats[f"shift_vol_{s}"] == 0).all()


def test_matches_without_an_expectation_give_no_shift(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, "rush_att", week=9)
    feats, _, detail, _ = C.comp_shifts(pool, [tg], {}, sim_threshold=0.0, min_neff=0.0)
    assert all(feats[f"nomatch_{s}"].all() for s in C.SEARCHES)
    assert set(detail.filter(pl.col("applicable"))["reason"].unique().to_list()) <= {"no_expectations", "best_similarity_below_threshold", "missing_required_ftn_features", "no_similarity", "no_target_vector"}


# ================================================================================================================================ review fixes
def test_rb2_share_of_a_bell_cow_game_is_zero_not_missing():
    from features import comps_ledger as L
    player = pl.DataFrame({"game_id": ["g1", "g2", "g2"], "team": ["A", "A", "A"], "player_id": ["r1", "r1", "r2"], "group": ["rb", "rb", "rb"],
                           "c.carries": [20.0, 14.0, 6.0], "c.targets": [0.0, 0.0, 0.0], "c.air_yards": [0.0] * 3, "c.targets_air": [0.0] * 3,
                           "c.gl_carries": [0.0] * 3, "offense_pct": [0.9, 0.7, 0.3]})
    team = pl.DataFrame({"game_id": ["g1", "g2"], "team": ["A", "A"], "x.targets": [30.0, 30.0], "x.gl_carries": [0.0, 0.0]})
    st = L.structure_ledger(player, team).sort("game_id")
    assert st["rb_rotation.rb2_carry_share|n"].to_list() == [0.0, 6.0] and st["rb_rotation.rb2_carry_share|d"].to_list() == [20.0, 20.0]
    pooled = sum(st["rb_rotation.rb2_carry_share|n"]) / sum(st["rb_rotation.rb2_carry_share|d"])
    assert pooled == pytest.approx(0.15)                                       # not 0.30: the bell-cow game counts as a 0 share


# ================================================================================================================================ review fixes (4c.1-4c.4)
def test_a_target_who_did_not_play_has_a_vector_but_is_no_observation():
    """Whether a scored player plays is known only after the game: his target vector must exist either way, and a game he missed must never be an
    observation of a later search."""
    inp = synthetic_inputs(0)
    wk = min(inp.player.filter((pl.col("player_id") == "AQB1") & (pl.col("season") == 2020) & pl.col("week").is_between(3, 7))["week"].to_list())
    gone = (pl.col("player_id") == "AQB1") & (pl.col("season") == 2020) & (pl.col("week") == wk)
    inp2 = C.Inputs(**{**inp.__dict__, "player": inp.player.filter(~gone)})
    vec = C.build_vectors(inp2)
    row = vec.player.filter((pl.col("player_id") == "AQB1") & (pl.col("season") == 2020) & (pl.col("week") == wk) & (pl.col("unit") == "qb_archetype")
                            & (pl.col("window") == W3))
    assert row.height == len(C.stored_spaces("qb_archetype")) and row["is_target"].all() and not row["in_pool"].any()
    assert row["n_present"].max() > 0                                                    # built from his earlier games, like any target
    pool, _, _ = _pool(inp2, vec)
    g = inp2.games.filter((pl.col("season") == 2020) & (pl.col("week") == 8) & (pl.col("team") == "A")).row(0, named=True)
    tg = C.Target("pass_att", g["game_id"], "A", g["opponent"], 2020, 8, "AQB1")
    missed = inp2.games.filter((pl.col("season") == 2020) & (pl.col("week") == wk) & (pl.col("team") == "A"))["game_id"][0]
    res = pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0)
    assert res["S1:pass_coverage"].matches.height > 0
    for s, r in res.items():
        if r.matches is not None and r.matches.height:
            assert not ((r.matches["obs_game_id"] == missed) & (r.matches["obs_player_id"] == "AQB1")).any(), s


def test_archetype_z_of_week_w_does_not_depend_on_who_played_in_week_w():
    """The league mean / SD and the shrinkage mean of a week come from that week's pregame depth chart, not from who turned out to play."""
    inp = synthetic_inputs(0)
    gone = (pl.col("player_id") == "BWR2") & (pl.col("season") == 2020) & (pl.col("week") == 6)
    assert inp.player.filter(gone).height == 1
    a = C.build_vectors(inp).player
    b = C.build_vectors(C.Inputs(**{**inp.__dict__, "player": inp.player.filter(~gone)})).player
    wk = lambda v: _frame_key(v.filter((pl.col("season") == 2020) & (pl.col("week") == 6) & (pl.col("player_id") != "BWR2")))
    assert wk(a).height == wk(b).height and wk(a).select("z", "raw").equals(wk(b).select("z", "raw"))


def test_extended_falls_back_to_base_when_the_extended_sigma_is_missing(pool3):
    pool, inp, _ = pool3
    unit = next(u for u in cs.DEFENSE_UNITS if len(C.stored_spaces(u)) == 2)
    tg = _team_target(inp, week=9)
    key = tg.season * 100 + tg.week
    k = pool._kmax(key)
    o = pool.idx[(tg.game_id, tg.opponent)]
    saved = dict(pool.sig)
    try:
        for kk in list(pool.sig):
            if kk[0] == unit and kk[1] == "extended":
                pool.sig[kk] = float("nan")
        pool.clear_cache()
        sim = pool.unit_sims(unit, "def", o, k, key)[0]
        d = C.unit_distance(pool.T[unit]["base"][0][o][None, :], pool.P[unit]["base"][0][:k], unit, "base")
        expect = C.similarity(d.d2[0], pool._sigma(unit, "base", key))
        assert np.isfinite(sim).sum() > 0 and np.allclose(sim, expect, equal_nan=True)
        # the archetype similarity of a player search falls back the same way
        arch = "rb_archetype"
        for kk in list(pool.sig):
            if kk[0] == arch and kk[1] == "extended":
                pool.sig[kk] = float("nan")
        pool.clear_cache()
        m = pool.search(_player_target(inp, week=9), which=("S3",), keep=True, sim_threshold=0.0, min_neff=0)[RB3].matches
        assert m.height > 0 and m[f"sim_{arch}"].is_finite().all()
    finally:
        pool.sig = saved
        pool.clear_cache()


def test_the_healthy_run_reads_the_healthy_target_vectors(pool3):
    pool, inp, _ = pool3
    seen = {True: 0, False: 0}
    for gm in inp.games.filter((pl.col("season") == 2020) & (pl.col("week") == 9)).iter_rows(named=True):
        tg = C.Target("spread", gm["game_id"], gm["team"], gm["opponent"], 2020, 9)
        g = pool.idx[(tg.game_id, tg.team)]
        differs = pool.adjusted_differs(g, cs.MARKET_UNITS["spread"])
        pool.clear_cache()
        a = pool.search(tg, which=("S1", "S4"), keep=True, sim_threshold=0.0, min_neff=0, version="adjusted")
        h = pool.search(tg, which=("S1", "S4"), keep=True, sim_threshold=0.0, min_neff=0, version="healthy")
        assert all(a[u].matches.equals(h[u].matches) for u in a if u.startswith("S1"))     # S1 reads only the defenses faced: one version
        same = all(a[u].matches.equals(h[u].matches) for u in a if u.startswith("S4"))
        assert same != differs
        seen[differs] += 1
    assert seen[True] > 0 and seen[False] > 0


def test_retrieval_change_is_stored_for_every_target_market_and_search(shifts44):
    (feats, _, detail, retrieval), tgs, _ = shifts44
    assert retrieval.height == detail.height                                              # one row per target, market and unit search
    assert {"overlap_top_k", "weight_mass_shared", "shift_vol_change", "shift_eff_change", "adjusted_differs", "nomatch_healthy"} <= set(retrieval.columns)
    assert retrieval["adjusted_differs"].any()                                            # the comparison run happened
    summ = C.retrieval_summary(retrieval)
    assert {"share_adjusted_differs", "mean_overlap_top_k", "mean_abs_shift_vol_change"} <= set(summ.columns)


def test_aggregate_median_ignores_targets_without_similarity():
    rows = [dict(search="S1", market="spread", game_id=f"g{i}", team="A", player_id=None, season=2020, week=1, applicable=True, n_matches=0, n_eff=0.0,
                 best_similarity=b, no_match=True, widened_uncertainty=True, shift=0.0, reason="r", unit_best="{}") for i, b in enumerate([float("nan")] * 3 + [0.0, 0.2])]
    log = C.aggregate_search_log(pl.DataFrame(rows))
    assert log.filter(pl.col("unit") == "combined")["median_best_similarity"].to_list() == [pytest.approx(0.1)] * 3


def test_the_sensitivity_columns_recount_matches_at_other_thresholds(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, week=9)
    log = C.run_search_log(pool, [tg], sensitivity=(0.0, 0.5))
    import json
    for r in log.iter_rows(named=True):
        sens = json.loads(r["sensitivity"])
        if not r["applicable"] or r["reason"] in ("no_target_vector", "missing_required_ftn_features"):
            assert sens == {}
            continue
        res = pool.search(tg, which=(r["parent"],), keep=True, sim_threshold=0.5, min_neff=0)[r["search"]]
        assert sens["0.5"][0] == res.summary["n_matches"] and sens["0.5"][1] == pytest.approx(res.summary["n_eff"])
        assert sens["0"][0] >= sens["0.5"][0]


def test_nomatch_sensitivity_reproduces_the_live_rule_at_the_live_threshold(pool3):
    pool, inp, _ = pool3
    tgs = [_team_target(inp, week=9), _player_target(inp, week=9), _player_target(inp, "pass_att", week=9)]
    log = C.run_search_log(pool, tgs, sensitivity=(0.3, cs.SIM_THRESHOLD))
    sens = C.nomatch_sensitivity(log, thresholds=(0.3, cs.SIM_THRESHOLD), min_neffs=(cs.MIN_NEFF,))
    live = log.filter(pl.col("applicable")).group_by("season", "market", "search").agg(pl.col("no_match").mean())
    j = sens.filter(pl.col("threshold") == cs.SIM_THRESHOLD).join(live, on=["season", "market", "search"])
    assert j.height == live.height and np.allclose(j["no_match_rate"].to_numpy(), j["no_match"].to_numpy())


def test_s1_own_history_is_exempt_from_the_across_search_average():
    """S1 holds only the target's own games, so its team has 100% of S1 by definition. Counting S1 in the across-search average would make the cap
    impossible and push that team out of every other search."""
    per = {"S1": (np.ones(12), np.array([f"own{i}" for i in range(12)]), np.array(["OWN"] * 12))}
    for s in ("S2", "S3", "S4", "S5"):
        teams = np.array(["OWN"] + [f"T{i}" for i in range(9)])
        per[s] = (np.ones(10), np.array([f"{s}g{i}" for i in range(10)]), teams)
    out = C.cap_across_searches(per)
    assert np.allclose(out["S1"], per["S1"][0])                                          # S1 untouched (no team-game above 25%)
    for s in ("S2", "S3", "S4", "S5"):
        assert np.allclose(out[s], per[s][0])                                             # OWN averages 0.4/5 = 8% outside S1: below the 15% cap


def test_team_vectors_do_not_depend_on_the_rows_kept_for_archetypes():
    """The lineup correction normalises the normal usage shares over the game's lineup players only. A player kept for the archetype vectors (on the
    pregame depth chart, or a scored target) who has not played for the team in its last six games must not rescale them."""
    inp = synthetic_inputs(0)
    early = inp.player.filter((pl.col("player_id") == "AWR1") & (pl.col("season") == 2019) & (pl.col("week") <= 3)).with_columns(player_id=pl.lit("AWR9"))
    pos = inp.positions.filter(pl.col("gsis_id") == "AWR1").with_columns(gsis_id=pl.lit("AWR9"))
    slots = inp.slots.filter(pl.col("gsis_id") == "AWR1").with_columns(gsis_id=pl.lit("AWR9"), slot=pl.lit(3, dtype=inp.slots["slot"].dtype))
    dc = inp.depth_chart.filter(pl.col("gsis_id") == "AWR1").with_columns(gsis_id=pl.lit("AWR9"), slot=pl.lit(3, dtype=inp.depth_chart["slot"].dtype))
    base = dict(inp.__dict__, player=pl.concat([inp.player, early]), positions=pl.concat([inp.positions, pos]), slots=pl.concat([inp.slots, slots]))
    inp_dc = C.Inputs(**dict(base, depth_chart=pl.concat([inp.depth_chart, dc])))
    with_dc = C.build_vectors(inp_dc)
    without = C.build_vectors(C.Inputs(**dict(base, depth_chart=None, targets=None)))
    assert _frame_key(with_dc.team).equals(_frame_key(without.team))
    late = C.archetype_rows(inp_dc).filter((pl.col("player_id") == "AWR9") & pl.col("is_reference") & ~pl.col("in_pool"))
    assert late.height == len(SEASONS) * WEEKS - 3                                      # a pregame reference row in every week he did not play


def test_top_any_lists_the_closest_past_games_whatever_the_threshold(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, week=9)
    live = pool.search(tg, which=("S4",), keep=True, sim_threshold=0.9999999, min_neff=0, top_any=10)[RUN4]
    full = pool.search(tg, which=("S4",), keep=True, sim_threshold=0.0, min_neff=0)[RUN4].matches
    assert live.summary["n_matches"] == 0 and len(live.summary["top_any"]) == 10            # nothing above the threshold, the closest games still listed
    want = full.sort("final_weight", descending=True).head(10)
    assert [t[1] for t in live.summary["top_any"]] == pytest.approx(want["final_weight"].to_list())
    ids = {f"{g}|{t}|{p or ''}" for g, t, p in zip(want["obs_game_id"], want["obs_team"], want["obs_player_id"])}
    assert {t[0] for t in live.summary["top_any"]} == ids


def test_closest_10_overlap_is_one_when_the_vectors_agree(pool3):
    pool, inp, _ = pool3
    gm = next(r for r in inp.games.filter((pl.col("season") == 2020) & (pl.col("week") == 9)).iter_rows(named=True)
              if not pool.adjusted_differs(pool.idx[(r["game_id"], r["team"])], cs.MARKET_UNITS["spread"]))
    tg = C.Target("spread", gm["game_id"], gm["team"], gm["opponent"], 2020, 9)
    _, _, _, retrieval = C.comp_shifts(pool, [tg], {}, sim_threshold=0.0, min_neff=0.0)
    r = retrieval.filter(pl.col("search").is_in(["S1", "S2", "S4"]))                       # S3: team market; S5: no FTN data in the synthetic league
    assert (~r["adjusted_differs"]).all() and (r["overlap_closest_10"] == 1.0).all() and (r["shift_vol_change"] == 0.0).all()


def test_s1_retrieval_never_depends_on_the_offense_version(shifts44):
    (_, _, _, retrieval), _, _ = shifts44
    s1 = retrieval.filter(pl.col("parent") == "S1")
    assert s1.height > 0 and (s1["overlap_closest_10"] == 1.0).all() and not s1["adjusted_differs"].any()     # S1 reads only the defenses faced


# ---------------------------------------------------------------------------------------------------------------- 4c.4 review fixes
def _fake_result(search, rows):
    """A SearchResult whose matches are `rows` = [(obs_game_id, obs_team, final_weight)] (team market)."""
    m = pl.DataFrame({"obs_game_id": [r[0] for r in rows], "obs_team": [r[1] for r in rows], "obs_player_id": pl.Series([None] * len(rows), dtype=pl.String),
                      "final_weight": [float(r[2]) for r in rows], "sim_combined": [0.9] * len(rows)})
    summ = dict(search=search, applicable=True, no_match=False, reason=None, n_matches=len(rows), n_eff=float(len(rows)), best_similarity=0.9)
    return C.SearchResult(search, summ, m)


def _empty_result(search):
    summ = dict(search=search, applicable=True, no_match=True, reason="best_similarity_below_threshold", n_matches=0, n_eff=0.0, best_similarity=0.1)
    return C.SearchResult(search, summ, None)


def test_a_search_that_ends_no_match_does_not_move_the_other_searches():
    """S2 has three past games but only team X's has an expectation, so S2 ends no_match. It must not enter the across-search cap: X keeps its 20% of
    S4 and S4's shift is the plain shrunk mean."""
    tg = C.Target("spread", "G", "T", "O", 2022, 5)
    s4 = [(f"g{i}", "X" if i < 2 else f"T{i}", 1.0) for i in range(10)]
    res = {"S1": _empty_result("S1"), "S2": _fake_result("S2", [("h0", "X", 1.0), ("h1", "Y", 1.0), ("h2", "Z", 1.0)]),
           "S3": _empty_result("S3"), "S4": _fake_result("S4", s4), "S5": _empty_result("S5")}
    zl = {"team_plays": {("h0", "X"): 1.0} | {(g, t): (3.0 if t == "X" else 0.0) for g, t, _ in s4}}
    vals, det, _, caps = C._market_shifts(res, tg, "spread", zl, cs.MIN_NEFF)
    assert vals["nomatch_S2"] and not vals["nomatch_S4"]
    n = 10.0
    assert vals["shift_vol_S4"] == pytest.approx(0.6 * n / (n + cs.SHRINK_K))
    assert np.allclose(caps["vol"]["S4"], 1.0)


def test_a_team_whose_share_cannot_be_reduced_is_frozen_not_zeroed():
    """DEN holds all of S2 (two games, nobody else) and 2 of S4's 10 games. Its average over the five searches is at least 1/5 = 20% whatever happens in
    S4, so the 15% cap cannot hold: DEN is left as it is (and logged), not pushed out of S4."""
    per = {"S2": (np.ones(2), np.array(["a", "b"]), np.array(["DEN", "DEN"])),
           "S4": (np.ones(10), np.array([f"g{i}" for i in range(10)]), np.array(["DEN", "DEN"] + [f"T{i}" for i in range(8)]))}
    info = {}
    out = C.cap_across_searches(per, info=info)
    assert np.allclose(out["S4"], 1.0) and np.allclose(out["S2"], 1.0)
    assert info["frozen"] == ["DEN"]


def test_the_across_search_cap_reduces_only_what_it_must():
    """X averages 0.2 + 0.4 + 0.4 = 1.0 / 5 = 20% over S2-S4 (no irreducible share): one common factor brings it to exactly 15%."""
    per = {}
    for s, k in (("S2", 1), ("S3", 2), ("S4", 2)):
        teams = np.array(["X"] * k + [f"{s}{i}" for i in range(5 - k)])
        per[s] = (np.ones(5), np.array([f"{s}g{i}" for i in range(5)]), teams)
    info = {}
    out = C.cap_across_searches(per, info=info)
    avg = sum(out[s][per[s][2] == "X"].sum() / out[s].sum() for s in per) / 5
    assert avg == pytest.approx(cs.TEAM_CAP_AVG_ACROSS_SEARCHES, abs=1e-6) and info["frozen"] == []
    for s in per:
        assert out[s].sum() == pytest.approx(5.0)                                          # totals kept


def test_an_infeasible_per_search_cap_leaves_the_weights_alone():
    """Fewer than 1 / 0.25 = 4 past team-games with weight: no split can respect the 25% cap. The weights stay as they are, so n_eff (1.04 here) decides,
    instead of an equal split that would claim n_eff = 3."""
    w = np.array([0.98, 0.01, 0.01])
    assert np.allclose(C.cap_shares(w, np.array(["a", "b", "c"]), 0.25), w)
    padded = C.cap_shares(np.array([0.98, 0.02, 0.0, 0.0]), np.array(["a", "b", "c", "d"]), 0.25)      # zero-weight games cannot absorb weight
    assert np.allclose(padded, [0.98, 0.02, 0.0, 0.0])


def test_n_eff_of_two_equal_games_meets_min_neff_two():
    assert C.meets_min_neff(1.9999999999999998, 2.0) and not C.meets_min_neff(1.99, 2.0)


def test_a_thin_efficiency_side_is_flagged_in_the_features():
    tg = C.Target("rush_yds", "G", "T", "O", 2022, 5, "P")
    rows = [(f"g{i}", f"T{i}", 1.0) for i in range(8)]
    m = pl.DataFrame({"obs_game_id": [r[0] for r in rows], "obs_team": [r[1] for r in rows], "obs_player_id": [f"P{i}" for i in range(8)],
                      "final_weight": [1.0] * 8, "sim_combined": [0.9] * 8})
    res = {s: _empty_result(s) for s in C.SEARCHES}
    res["S3"] = C.SearchResult("S3", dict(search="S3", applicable=True, no_match=False, reason=None, n_matches=8, n_eff=8.0, best_similarity=0.9), m)
    zl = {"rush_att": {(f"g{i}", f"P{i}"): 0.5 for i in range(8)}, "ypc": {("g0", "P0"): 2.0}}     # one game has an efficiency expectation
    vals, det, _, _ = C._market_shifts(res, tg, "rush_yds", zl, cs.MIN_NEFF, {"ypc": {("g0", "P0"): 12.0}})
    assert not vals["nomatch_S3"] and vals["nomatch_eff_S3"] and vals["shift_eff_S3"] == 0.0 and vals["n_eff_eff_S3"] == pytest.approx(1.0)
    assert vals["n_eff_S3"] == pytest.approx(8.0)


def test_a_dead_search_cannot_knock_a_live_one_out_through_the_cap():
    """S2 ends no_match (n_eff 1.1). Had it entered the across-search cap, team X (95% of S2, 50% of S4) would be cut in S4 to 26%, leaving S4 with an
    n_eff of 1.6 and no match. Only matching searches enter the cap, so S4 keeps its two equal games (n_eff 2) and matches."""
    tg = C.Target("spread", "G", "T", "O", 2022, 5)
    res = {"S1": _empty_result("S1"), "S2": _fake_result("S2", [("h0", "X", 0.95), ("h1", "Y", 0.05)]), "S3": _empty_result("S3"),
           "S4": _fake_result("S4", [("g0", "X", 1.0), ("g1", "A", 1.0)]), "S5": _empty_result("S5")}
    zl = {"team_plays": {("h0", "X"): 1.0, ("h1", "Y"): 1.0, ("g0", "X"): 1.0, ("g1", "A"): -1.0}}
    vals, _, _, _ = C._market_shifts(res, tg, "spread", zl, cs.MIN_NEFF)
    assert vals["nomatch_S2"] and not vals["nomatch_S4"] and vals["n_eff_S4"] == pytest.approx(2.0)


def test_retrieval_change_does_not_depend_on_the_hash_seed():
    """A set of string ids iterates in a different order in every process (hash randomisation): a float sum over one must not follow it."""
    import subprocess
    import sys
    code = ("import numpy as np; from models import comps as C; rng = np.random.default_rng(0); "
            "ids = [f'g{i}|T{i % 7}|P{i}' for i in range(400)]; w = rng.uniform(0, 1, 400).tolist(); "
            "h = [(i, x, 0.0) for i, x in zip(ids, w)]; a = [(i, x * 1.1, 0.0) for i, x in zip(ids[::-1], w)]; "
            "print(repr(C.retrieval_change(h, a, 10)['weight_mass_shared']))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(config.ROOT),
                           env={**__import__("os").environ, "PYTHONHASHSEED": str(seed)}).stdout.strip() for seed in range(6)}
    assert len(outs) == 1 and outs != {""}


# ================================================================================================================================ 4c.5 input shares
def test_tag_shares_follow_the_effective_feature_weights():
    """Group 1 = an observed (q 1) and an estimated (q 0.65) feature, group 2 = a derived one. Each group carries half of the unit distance."""
    A = np.array([[0.0, 1.0, 2.0]])
    B = np.array([[1.0, 1.0, np.nan], [0.0, 0.0, 0.0]])
    d = C.group_distance(A, B, [[0, 1], [2]], np.ones(3), np.array([1.0, 0.65, 0.9]), tags=np.array([0, 2, 1]))
    sh = d.shares[:, 0, :]
    assert sh[:, 0] == pytest.approx([1 / 1.65, 0.0, 0.65 / 1.65])                            # the derived feature is missing: only group 1 counts
    assert sh[:, 1] == pytest.approx([0.5 / 1.65, 0.5, 0.5 * 0.65 / 1.65])
    assert np.allclose(sh.sum(axis=0), 1.0)
    assert C.group_distance(A, B, [[0, 1], [2]], np.ones(3), np.ones(3)).shares is None       # only on request


def test_every_match_carries_its_input_shares(pool3):
    pool, inp, _ = pool3
    for tg in (_team_target(inp, week=9), _player_target(inp, week=9)):
        for s, r in pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0).items():
            m = r.matches
            if m is None or m.height == 0:
                continue
            tot = (m["share_obs"] + m["share_der"] + m["share_est"]).to_numpy()
            assert np.allclose(tot, 1.0), s


def test_search_shares_are_the_shift_weighted_mean_of_the_matches(shifts44):
    """A unit search's shares are its matches' shares weighted by the volume shift's weights; a search's, the n_eff-weighted mean over its matching
    unit searches."""
    (feats, matches, detail, _), _, _ = shifts44
    seen = 0
    for r in feats.iter_rows(named=True):
        tgt = (pl.col("game_id") == r["game_id"]) & (pl.col("team") == r["team"]) & ((pl.col("player_id") == r["player_id"]) if r["player_id"] else pl.col("player_id").is_null())
        for s in C.SEARCHES:
            u = detail.filter(tgt & (pl.col("market") == r["market"]) & (pl.col("parent") == s) & pl.col("applicable") & ~pl.col("nomatch"))
            if r[f"nomatch_{s}"]:
                assert u.height == 0
                continue
            per_unit = []
            for d in u.iter_rows(named=True):
                m = matches.filter((pl.col("market") == r["market"]) & (pl.col("search") == d["search"]) & (pl.col("target_game_id") == r["game_id"])
                                   & (pl.col("target_team") == r["team"]) & ((pl.col("target_player_id") == r["player_id"]) if r["player_id"] else pl.col("target_player_id").is_null()))
                w = m["weight_capped_vol"].to_numpy()
                per_unit.append(((w * m["share_obs"].to_numpy()).sum() / w.sum(), (w * m["completeness_penalty"].to_numpy()).sum() / w.sum()))
            nw = u["n_eff_vol"].to_numpy()
            assert r[f"share_obs_{s}"] == pytest.approx((nw * np.array([x[0] for x in per_unit])).sum() / nw.sum())
            assert r[f"completeness_{s}"] == pytest.approx((nw * np.array([x[1] for x in per_unit])).sum() / nw.sum())
            assert r[f"share_obs_{s}"] + r[f"share_der_{s}"] + r[f"share_est_{s}"] == pytest.approx(1.0)
            seen += 1
    assert seen > 0


def _feats_table():
    rows = []
    for market, pid in (("rush_att", "P1"), ("rush_yds", "P1"), ("targets", "P2"), ("total", None), ("spread", None)):
        r = dict(season=2022, week=3, game_id="G1", team="A", opponent="B", player_id=pid, market=market)
        for s in C.SEARCHES:
            r.update({f"shift_vol_{s}": 0.1 if market == "rush_att" else 0.2, f"shift_eff_{s}": 0.3, f"n_eff_{s}": 2.5, f"nomatch_{s}": s != "S1",
                      f"best_sim_{s}": 0.8, f"n_eff_eff_{s}": 2.0, f"nomatch_eff_{s}": s != "S1", f"share_obs_{s}": 0.7, f"share_der_{s}": 0.2,
                      f"share_est_{s}": 0.1, f"completeness_{s}": 0.05})
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def test_comparable_features_join_the_tables_of_their_market():
    from features import comps_features as cf
    from features.volume_features import FeatureTables
    players = {"rush_att": pl.DataFrame({"player_id": ["P1", "P9"], "game_id": ["G1", "G1"], "x": [1.0, 2.0]}),
               "targets": pl.DataFrame({"player_id": ["P2"], "game_id": ["G1"], "x": [3.0]})}
    teams = pl.DataFrame({"game_id": ["G1", "G1"], "team": ["A", "B"], "x": [1.0, 2.0]})
    specs = {"rush_att": {"market": "rush_att"}, "targets": {"market": "targets"}}
    out = cf.attach(FeatureTables(players, teams), _feats_table(), specs)
    ra = out.players["rush_att"]
    assert ra.height == 2 and ra["cmp_shift_vol_S1"].to_list() == [0.1, None]           # P9 is no scored target: missing, never 0
    assert ra["cmp_nomatch_S2"].dtype == pl.Float64 and ra["cmp_nomatch_S2"][0] == 1.0 and ra["cmp_nomatch_S1"][0] == 0.0
    assert {"cmp_share_obs_S3", "cmp_share_der_S3", "cmp_share_est_S3", "cmp_completeness_S3"} <= set(ra.columns)
    assert out.players["targets"]["cmp_shift_vol_S1"].to_list() == [0.2]
    assert out.teams["cmp_shift_vol_S5"].to_list() == [0.2, None]                         # the team tables read the game market `total`
    only1 = cf.attach(FeatureTables(players, teams), _feats_table(), specs, searches=("S1",))
    assert {c for c in only1.players["rush_att"].columns if c.startswith("cmp_")} == {"cmp_" + c for c in cf.columns(("S1",))}


def test_input_shares_are_separate_columns_per_search():
    from features import comps_features as cf
    sh = cf.input_shares(_feats_table())
    for s in C.SEARCHES:
        assert {f"share_obs_{s}", f"share_der_{s}", f"share_est_{s}", f"completeness_{s}"} <= set(sh.columns)


# ---------------------------------------------------------------------------------------------------------------- 4c.5 review fixes
def test_an_all_null_comparable_column_reaches_lightgbm_as_a_float():
    """S2 and S4 never match at the plan's threshold, so their share columns are null in every row: they must still be numeric for LightGBM."""
    from features import comps_features as cf
    from features.volume_features import FeatureTables
    from models import volume as vm
    f = _feats_table().with_columns(*[pl.lit(None).alias(f"{c}_S2") for c in ("share_obs", "share_der", "share_est", "completeness")])
    assert f["share_obs_S2"].dtype == pl.Null
    rng = np.random.default_rng(0)
    players = {"rush_att": pl.DataFrame({"player_id": ["P1"] * 40, "game_id": ["G1"] + [f"H{i}" for i in range(39)], "x": rng.normal(size=40),
                                         "label": rng.normal(size=40)})}
    out = cf.attach(FeatureTables(players, pl.DataFrame({"game_id": ["G1"], "team": ["A"]})), f, {"rush_att": {"market": "rush_att"}})
    df = out.players["rush_att"]
    assert all(df[c].dtype == pl.Float64 for c in df.columns if c.startswith("cmp_"))
    vm._fit(df, dict(vm.PARAMS, n_estimators=5, min_child_samples=2))                     # raised "pandas dtypes must be int, float or bool" before


def test_attach_refuses_a_search_selection_it_cannot_honour():
    from features import comps_features as cf
    from features.volume_features import FeatureTables
    tables = FeatureTables({"rush_att": pl.DataFrame({"player_id": ["P1"], "game_id": ["G1"]})}, pl.DataFrame({"game_id": ["G1"], "team": ["A"]}))
    specs = {"rush_att": {"market": "rush_att"}}
    with pytest.raises(TypeError):
        cf.attach(tables, _feats_table(), specs, searches="S1")                            # a string would iterate over its characters
    for bad in ((), ("S9",)):
        with pytest.raises(ValueError):
            cf.attach(tables, _feats_table(), specs, searches=bad)
    with pytest.raises(KeyError):
        cf.attach(tables, _feats_table().drop("share_obs_S1"), specs)                       # a stale table without the 4c.5 columns


def test_a_no_match_search_still_reports_its_input_shares_from_its_closest_games(pool3):
    """'For every search and target' (plan 4c.5.1): without a match the shares and completeness come from the search's 10 closest past games."""
    pool, inp, _ = pool3
    tg = _player_target(inp, "rush_att", week=9)
    pool.clear_cache()
    feats, _, _, _ = C.comp_shifts(pool, [tg], {}, sim_threshold=0.9999999)
    r = feats.filter(pl.col("market") == "rush_att").row(0, named=True)
    full = pool.search(tg, which=("S1",), keep=True, sim_threshold=0.0, min_neff=0)["S1:run_defense"].matches.sort("final_weight", descending=True).head(10)
    w = full["final_weight"].to_numpy()
    # rush_att has one defense unit, so S1 is one unit search and its closest games are S1's
    assert r["nomatch_S1"] and r["share_obs_S1"] == pytest.approx((w * full["share_obs"].to_numpy()).sum() / w.sum())
    assert r["completeness_S1"] == pytest.approx((w * full["completeness_penalty"].to_numpy()).sum() / w.sum())


def test_a_pairs_shares_are_the_mean_of_its_two_units():
    """A pair's similarity is the product of its two units: each unit carries half of the share. A one-unit search carries its unit's."""
    one = lambda v: (np.ones(2), np.ones(2), np.ones(2), np.array(v, dtype=float)[:, None] * np.ones((3, 2)))
    sh = C._match_shares({"run_offense": ("off", one([1, 0, 0])), "run_defense": ("def", one([0, 0, 1]))}, 2)
    assert sh[:, 0] == pytest.approx([1 / 2, 0, 1 / 2])
    assert C._match_shares({"run_defense": ("def", one([0, 0, 1]))}, 2)[:, 0] == pytest.approx([0, 0, 1])


# ================================================================================================================================ 4c.6 window selection
def test_unit_retrieval_scores_each_unit_against_the_targets_own_z(pool3):
    from models import comps_windows as W
    pool, inp, _ = pool3
    tgs = [_team_target(inp, week=9), _player_target(inp, "rush_att", week=9)]
    rng = np.random.default_rng(2)
    zl = {q: {**{(r["game_id"], r["team"]): float(rng.normal()) for r in inp.games.iter_rows(named=True)},
              **{(r["game_id"], r["player_id"]): float(rng.normal()) for r in inp.player.iter_rows(named=True)}}
          for q in ("team_plays", "pts_per_play", "rush_att", "ypc")}
    e = W.unit_retrieval_errors(pool, tgs, zl, counts={q: {k: 5.0 for k in zl[q]} for q in ("pts_per_play", "ypc")})
    assert set(e["market"].unique()) == {"spread", "total", "moneyline", "rush_att", "rush_yds"}
    assert set(e.filter(pl.col("market") == "rush_att")["unit"].unique()) <= set(cs.MARKET_UNITS["rush_att"])
    assert set(e.filter(pl.col("market") == "rush_yds")["side"].unique()) == {"vol", "eff"} and (e["n_neighbours"] <= W.K_NEIGHBOURS).all()
    assert (e["sq_error"] >= 0).all()


def test_window_choice_compares_windows_on_every_target_and_breaks_ties_by_order():
    from models import comps_windows as W
    rows = []
    for wf, err in (("last_3", 1.0), ("last_6", 0.5), ("recency_weighted", 0.5)):
        for i in range(3):
            rows.append(dict(window_family=wf, market="rush_att", unit="run_defense", side="vol", game_id=f"G{i}", team="A", player_id="P", sq_error=err))
    for wf in ("last_6", "recency_weighted"):
        rows.append(dict(window_family=wf, market="rush_att", unit="run_defense", side="vol", game_id="G9", team="A", player_id="P", sq_error=0.0))
    zl = {"rush_att": {(f"G{i}", "P"): 1.0 for i in (0, 1, 2, 9)}}
    summ, chosen = W.choose(pl.DataFrame(rows), zl)
    assert chosen.row(0, named=True)["window_family"] == "last_6"                       # tie with recency_weighted: the earlier window wins
    assert summ.filter(pl.col("window_family") == "last_3")["error"][0] == pytest.approx((3 * 1.0 + 1.0) / 4)      # G9 charged z^2 = 1
    assert W.window_name("continuity_weighted", "rush_yds") == "continuity_weighted:rush_yds" and W.window_name("last_6", "rush_yds") == "last_6"


@pytest.fixture(scope="module")
def two_windows(pool3):
    pool, inp, vec = pool3
    sigma = C.build_sigma(vec.team, vec.player)
    return {W3: pool, "last_6": C.Pool(vec.team, vec.player, None, inp.games, inp.lineups, inp.slots, sigma, "last_6")}, inp


def test_a_windowed_pool_on_the_base_window_searches_exactly_like_the_base_pool(two_windows):
    pools, inp = two_windows
    choice = {m: {u: W3 for u in cs.MARKET_UNITS[m]} for m in cs.MARKET_UNITS}
    wp = C.WindowedPool(pools, choice, base=W3)
    for tg in (_team_target(inp, week=9), _player_target(inp, week=9)):
        wp.clear_cache()
        a = wp.search(tg, keep=True, sim_threshold=0.0, min_neff=0)
        pools[W3].clear_cache()
        b = pools[W3].search(tg, keep=True, sim_threshold=0.0, min_neff=0)
        for s in a:
            assert str(a[s].summary) == str(b[s].summary)
            assert (a[s].matches is None and b[s].matches is None) or a[s].matches.equals(b[s].matches)


def test_each_unit_reads_its_own_window(two_windows):
    pools, inp = two_windows
    tg = _player_target(inp, "rush_att", week=9)
    choice = {m: {u: W3 for u in cs.MARKET_UNITS[m]} for m in cs.MARKET_UNITS}
    choice["rush_att"] = dict(choice["rush_att"], run_defense="last_6", rb_archetype="last_6")
    wp = C.WindowedPool(pools, choice, base=W3)
    wp.clear_cache()
    m = wp.search(tg, which=("S3",), keep=True, sim_threshold=0.0, min_neff=0)[RB3].matches
    pools["last_6"].clear_cache()
    ref = pools["last_6"].search(tg, which=("S3",), keep=True, sim_threshold=0.0, min_neff=0)[RB3].matches       # S3 = archetype x defense
    key = ["obs_game_id", "obs_team", "obs_player_id"]
    j = m.select(*key, "sim_run_defense", "sim_rb_archetype").join(ref.select(*key, "sim_run_defense", "sim_rb_archetype"), on=key, suffix="_ref")
    assert j.height > 0 and np.allclose(j["sim_run_defense"], j["sim_run_defense_ref"]) and np.allclose(j["sim_rb_archetype"], j["sim_rb_archetype_ref"])
    pools[W3].clear_cache()
    base = pools[W3].search(tg, which=("S3",), keep=True, sim_threshold=0.0, min_neff=0)[RB3].matches
    jb = m.select(*key, "sim_run_defense").join(base.select(*key, "sim_run_defense"), on=key, suffix="_base")
    assert not np.allclose(jb["sim_run_defense"], jb["sim_run_defense_base"])                 # and not the base window's


def test_markets_whose_units_read_different_windows_search_separately():
    tg = C.Target("rush_att", "G", "A", "B", 2022, 5, "P")
    choice = {"rush_att": {"run_defense": "last_6"}, "rush_yds": {"run_defense": "last_3"}}
    rows = C._target_rows([tg], choice)
    assert sorted(ms for _, m in rows for ms in m) == ["rush_att", "rush_yds"] and len(rows) == 2
    assert len(C._target_rows([tg])) == 1                                                    # one window for all: one search


def test_window_choice_keeps_team_markets_and_charges_a_window_for_targets_it_cannot_score():
    """Team targets have player_id null: the choice must keep them. A window that cannot score a target (no vector, e.g. season_to_date in week 1)
    is charged the error of a zero prediction, z^2, as a no-match shift of 0 would be."""
    from models import comps_windows as W
    zl = {"team_plays": {("G1", "A"): 2.0, ("G2", "A"): 1.0}, "rush_att": {("G1", "P"): 1.0}}
    rows = [dict(window_family="last_3", market="spread", unit="run_defense", side="vol", game_id="G1", team="A", player_id=None, sq_error=1.0),
            dict(window_family="last_3", market="spread", unit="run_defense", side="vol", game_id="G2", team="A", player_id=None, sq_error=0.25),
            dict(window_family="season_to_date", market="spread", unit="run_defense", side="vol", game_id="G2", team="A", player_id=None, sq_error=0.0),
            dict(window_family="last_3", market="rush_att", unit="run_defense", side="vol", game_id="G1", team="A", player_id="P", sq_error=0.5),
            dict(window_family="season_to_date", market="rush_att", unit="run_defense", side="vol", game_id="G1", team="A", player_id="P", sq_error=0.1)]
    summ, chosen = W.choose(pl.DataFrame(rows), zl)
    assert set(chosen["market"]) == {"spread", "rush_att"}
    sp = summ.filter(pl.col("market") == "spread").sort("window_family")
    assert sp["error"].to_list() == pytest.approx([(1.0 + 0.25) / 2, (4.0 + 0.0) / 2])     # season_to_date charged z^2 = 4 for G1
    assert chosen.filter(pl.col("market") == "spread")["window_family"][0] == "last_3"
    assert chosen.filter(pl.col("market") == "rush_att")["window_family"][0] == "season_to_date"


def test_a_windowed_pool_needs_a_window_for_every_unit_of_every_market(two_windows):
    pools, _ = two_windows
    with pytest.raises(ValueError):
        C.WindowedPool(pools, {"rush_att": {"run_defense": "last_6"}}, base=W3)              # markets and units missing: no silent fallback
    full = {m: {u: W3 for u in cs.MARKET_UNITS[m]} for m in cs.MARKET_UNITS}
    with pytest.raises(ValueError):
        C.WindowedPool(pools, dict(full, rush_att=dict(full["rush_att"], run_defense="last_9")), base=W3)     # a window no pool holds
    import copy
    copy.copy(C.WindowedPool(pools, full, base=W3))                                        # no RecursionError through __getattr__


def test_window_summary_reports_the_targets_a_window_could_not_score():
    from models import comps_windows as W
    rows = [dict(window_family=w, market="rush_att", unit="run_defense", side="vol", game_id=g, team="A", player_id="P", sq_error=0.5)
            for w, g in (("last_3", "G1"), ("last_3", "G2"), ("season_to_date", "G2"))]
    summ, _ = W.choose(pl.DataFrame(rows), {"rush_att": {("G1", "P"): 1.0, ("G2", "P"): 1.0}})
    assert dict(zip(summ["window_family"], summ["n_unscored"])) == {"last_3": 0, "season_to_date": 1}


# ================================================================================================================================ decisions of 2026-10-09
def test_check_similarity_is_the_geometric_mean_of_the_factors_with_a_floor_on_the_weaker():
    """A pair is checked on sqrt(a x b), so 0.63 / 0.63 reads 0.63 (the product would read 0.40); a pair whose weaker factor is below SIDE_FLOOR never
    passes on the average (0.95 / 0.30). A one-unit search is checked on its similarity."""
    a, b = np.array([0.63, 0.95, 0.9, np.nan]), np.array([0.63, 0.30, 0.9, 0.8])
    chk = C.check_similarity(a * b, a, b)
    assert chk[0] == pytest.approx(0.63) and np.isnan(chk[1]) and chk[2] == pytest.approx(0.9) and np.isnan(chk[3])
    one = np.array([0.5, 0.2, np.nan])
    assert np.allclose(C.check_similarity(one), one, equal_nan=True)                    # one unit: no floor, no square root
    assert C.SIDE_FLOOR == 0.40


def test_a_pair_matches_on_the_checked_similarity_and_weighs_the_product(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, week=9)
    floor_cut = 0
    res0 = pool.search(tg, which=("S3", "S4"), keep=True, sim_threshold=0.0, min_neff=0, top_any=10 ** 6)
    res5 = pool.search(tg, which=("S3", "S4"), keep=True, sim_threshold=0.5, min_neff=0)
    assert set(res0) == {RB3, RUN4}
    for u, full in res0.items():
        m = full.matches
        assert m.height > 0
        assert np.allclose(m["sim_check"].to_numpy(), np.sqrt((m["sim_offense"] * m["sim_defense"]).to_numpy()))
        assert (np.minimum(m["sim_offense"].to_numpy(), m["sim_defense"].to_numpy()) >= C.SIDE_FLOOR).all()
        assert np.allclose(m["sim_combined"].to_numpy(), (m["sim_offense"] * m["sim_defense"]).to_numpy())          # the weight stays the product
        assert (m["final_weight"] - m["sim_combined"] * m["recency_weight"] * m["continuity_weight"] * m["quality_weight"]).abs().max() < 1e-12
        assert full.summary["best_similarity"] == pytest.approx(m["sim_check"].max())
        floor_cut += len(full.summary["top_any"]) - m.height          # observations with a similarity that the floor kept out
        half, want = res5[u].matches, m.filter(pl.col("sim_check") >= 0.5)
        assert half.height == want.height and half["final_weight"].to_list() == want["final_weight"].to_list()
    assert floor_cut > 0
    # one-unit searches: the check is the unit's similarity, no floor
    one = pool.search(tg, which=("S1", "S2"), keep=True, sim_threshold=0.0, min_neff=0)
    assert set(one) == {"S1:run_defense", "S2:run_offense", "S2:rb_rotation", "S2:ol_protection", "S2:rb_archetype"}
    for u, r in one.items():
        m = r.matches
        sim = m["sim_defense"] if u.startswith("S1") else m["sim_offense"]
        assert np.allclose(m["sim_check"].to_numpy(), sim.to_numpy()) and np.allclose(m["sim_combined"].to_numpy(), sim.to_numpy())


def test_searches_run_on_the_healthy_target_vectors(pool3):
    """Decision 0: the lineup-adjusted vectors double-count a player already missing from the window; the searches read the healthy vectors and the
    adjusted run is the stored comparison."""
    pool, inp, _ = pool3
    gm = next(r for r in inp.games.filter((pl.col("season") == 2020) & (pl.col("week") == 9)).iter_rows(named=True)
              if pool.adjusted_differs(pool.idx[(r["game_id"], r["team"])], cs.MARKET_UNITS["spread"]))
    tg = C.Target("spread", gm["game_id"], gm["team"], gm["opponent"], 2020, 9)
    pool.clear_cache()
    d = pool.search(tg, which=("S4",), keep=True, sim_threshold=0.0, min_neff=0)
    h = pool.search(tg, which=("S4",), keep=True, sim_threshold=0.0, min_neff=0, version="healthy")
    a = pool.search(tg, which=("S4",), keep=True, sim_threshold=0.0, min_neff=0, version="adjusted")
    assert all(d[u].matches.equals(h[u].matches) for u in d) and not all(d[u].matches.equals(a[u].matches) for u in d)
    feats, _, detail, retrieval = C.comp_shifts(pool, [tg], {}, sim_threshold=0.0, min_neff=0.0)
    for u in d:
        r = retrieval.filter((pl.col("search") == u) & (pl.col("market") == "spread")).row(0, named=True)
        x = detail.filter((pl.col("search") == u) & (pl.col("market") == "spread")).row(0, named=True)
        assert r["adjusted_differs"] and r["nomatch_healthy"] == x["nomatch"] and r["shift_vol_healthy"] == x["shift_vol"]


def _eff_result(counts, z_eff=None, one_team=False):
    """An S3 result for a rec_yds target with one match per past game; volume z 0.5 everywhere; efficiency z and counts per game. one_team: every past
    game is the same team's (its share cannot move to another team, so the across-search cap leaves the weights alone)."""
    n = len(counts)
    teams = ["TX"] * n if one_team else [f"T{i}" for i in range(n)]
    m = pl.DataFrame({"obs_game_id": [f"g{i}" for i in range(n)], "obs_team": teams, "obs_player_id": [f"P{i}" for i in range(n)],
                      "final_weight": [1.0] * n, "sim_combined": [0.9] * n})
    res = {s: _empty_result(s) for s in C.SEARCHES}
    res["S3"] = C.SearchResult("S3", dict(search="S3", applicable=True, no_match=False, reason=None, n_matches=n, n_eff=float(n), best_similarity=0.9), m)
    z_eff = z_eff if z_eff is not None else [1.0] * n
    zl = {"targets": {(f"g{i}", f"P{i}"): 0.5 for i in range(n)}, "yds_per_rec": {(f"g{i}", f"P{i}"): z_eff[i] for i in range(n)}}
    cnt = {"yds_per_rec": {(f"g{i}", f"P{i}"): float(c) for i, c in enumerate(counts) if c is not None}}
    return res, zl, cnt


def test_efficiency_matches_are_weighted_by_their_count():
    """Decision 8: a 1-catch game's yards per reception is mostly noise; it gets a tenth of a 10-catch game's weight. The volume side is unchanged."""
    tg = C.Target("rec_yds", "G", "T", "O", 2022, 5, "P")
    res, zl, cnt = _eff_result([1, 10], z_eff=[4.0, 0.0], one_team=True)
    vals, _, sets, caps = C._market_shifts(res, tg, "rec_yds", zl, 0.0, cnt)
    assert np.allclose(sets["S3"][4], [1.0, 10.0]) and caps["eff"]["S3"][1] == pytest.approx(10 * caps["eff"]["S3"][0]) and np.allclose(caps["vol"]["S3"], 1.0)
    n = C.neff(np.array([1.0, 10.0]))
    assert vals["n_eff_eff_S3"] == pytest.approx(n) and vals["n_eff_S3"] == pytest.approx(2.0)
    assert vals["shift_eff_S3"] == pytest.approx((4.0 * 1 + 0.0 * 10) / 11 * n / (n + cs.SHRINK_K))


def test_count_weights_lower_the_efficiency_n_eff_when_a_few_games_carry_the_counts():
    tg = C.Target("rec_yds", "G", "T", "O", 2022, 5, "P")
    vals_eq = C._market_shifts(*_eff_result([3] * 8)[:1], tg, "rec_yds", _eff_result([3] * 8)[1], cs.MIN_NEFF, _eff_result([3] * 8)[2])[0]
    assert vals_eq["n_eff_eff_S3"] == pytest.approx(vals_eq["n_eff_S3"]) == pytest.approx(8.0)       # equal counts: nothing changes
    res, zl, cnt = _eff_result([1, 1, 1, 1, 1, 1, 2, 2])
    vals, _, _, caps = C._market_shifts(res, tg, "rec_yds", zl, cs.MIN_NEFF, cnt)
    assert vals["n_eff_eff_S3"] < vals["n_eff_S3"] == pytest.approx(8.0)
    assert vals["n_eff_eff_S3"] == pytest.approx(C.cluster_neff(caps["eff"]["S3"], np.array([f"g{i}|T{i}" for i in range(8)])))


def test_a_match_with_an_efficiency_expectation_but_no_count_is_an_error():
    tg = C.Target("rec_yds", "G", "T", "O", 2022, 5, "P")
    res, zl, cnt = _eff_result([1, None])
    with pytest.raises(ValueError, match="count"):
        C._market_shifts(res, tg, "rec_yds", zl, 0.0, cnt)
    with pytest.raises(ValueError, match="count"):
        C._market_shifts(res, tg, "rec_yds", zl, 0.0, None)


def test_efficiency_counts_are_the_denominators_of_the_ratios():
    pl_log = pl.DataFrame({"game_id": ["g1"], "player_id": ["P1"], "team": ["T"], "attempts": [30], "completions": [20], "carries": [12], "targets": [7],
                           "receptions": [5], "rush_att_ex_kneel": [3]})
    wf = pl.DataFrame({"game_id": ["g1", "g1"], "player_id": [None, "P1"], "team": ["T", "T"], "quantity": ["team_plays", "targets"], "actual": [64.0, 7.0]})
    c = C.efficiency_counts(pl_log, wf)
    k = ("g1", "P1")
    assert (c["comp_rate"][k], c["yds_per_cmp"][k], c["ypc"][k], c["catch_rate"][k], c["yds_per_rec"][k], c["qb_ypc"][k]) == (30, 20, 12, 7, 5, 3)
    assert c["pts_per_play"][("g1", "T")] == 64.0 and set(c) == {q for _, q in C.MARKET_QUANTITIES.values() if q}


def test_efficiency_count_columns_are_the_denominators_of_the_phase3_ratios():
    from features import phase4_inputs as p4
    from models import efficiency
    want = {p4.QUANTITIES[q]: spec["ratio"][1] for q, spec in efficiency.EFFICIENCY_SPECS.items()}
    assert C.EFF_DENOMINATORS == want


def test_window_selection_weighs_efficiency_neighbours_by_their_count(pool3):
    """The unit-only retrieval predicts the efficiency z the way the searches will: neighbour weight = similarity x count. Uniform counts change
    nothing, varying counts move only the efficiency side, and an efficiency neighbour without a count is an error."""
    from models import comps_windows as W
    pool, inp, _ = pool3
    tgs = [_player_target(inp, "rush_att", week=9), _player_target(inp, "rush_att", week=10)]
    rng = np.random.default_rng(2)
    keys = [(r["game_id"], r["player_id"]) for r in inp.player.iter_rows(named=True)]
    zl = {q: {k: float(rng.normal()) for k in keys} for q in ("rush_att", "ypc")}
    ones = W.unit_retrieval_errors(pool, tgs, zl, counts={"ypc": {k: 1.0 for k in keys}})
    sevens = W.unit_retrieval_errors(pool, tgs, zl, counts={"ypc": {k: 7.0 for k in keys}})
    varied = W.unit_retrieval_errors(pool, tgs, zl, counts={"ypc": {k: float(rng.integers(1, 30)) for k in keys}})
    assert np.allclose(ones["sq_error"].to_numpy(), sevens["sq_error"].to_numpy())
    vol = lambda e: e.filter(pl.col("side") == "vol")["sq_error"].to_numpy()
    eff = lambda e: e.filter(pl.col("side") == "eff")["sq_error"].to_numpy()
    assert np.array_equal(vol(ones), vol(varied)) and not np.allclose(eff(ones), eff(varied))
    with pytest.raises(ValueError, match="count"):
        W.unit_retrieval_errors(pool, tgs, zl)


# ================================================================================================================================ per-unit searches (2026-10-09)
def test_unit_searches_follow_the_plan():
    ids = lambda m, p: [u for u, _, _ in C.unit_searches(m, p)]
    assert ids("rush_att", True) == ["S1:run_defense", "S2:run_offense", "S2:rb_rotation", "S2:ol_protection", "S2:rb_archetype",
                                     "S3:rb_archetype*run_defense", RUN4, "S5"]
    assert [u for u in ids("pass_att", True) if u.startswith("S4")] == ["S4:pass_offense*pass_coverage", "S4:ol_protection*pass_rush"]
    assert [u for u in ids("targets", True) if u.startswith("S4")] == ["S4:pass_offense*pass_coverage"]
    assert [u for u in ids("spread", False) if u.startswith(("S3", "S4"))] == [RUN4, "S4:pass_offense*pass_coverage"]
    assert [u for u in ids("pass_att", True) if u.startswith("S1")] == [f"S1:{d}" for d in cs.MARKET_UNITS["pass_att"] if d in cs.DEFENSE_UNITS]
    for m in cs.MARKET_UNITS:
        for uid, s, factors in C.unit_searches(m, m not in ("spread", "total", "moneyline")):
            assert C.parent(uid) == s and all(u in cs.MARKET_UNITS[m] for _, u in factors)     # only the market's own units
            assert len(factors) == (2 if s in ("S3", "S4") else 1 if s in ("S1", "S2") else 0)


def test_a_game_that_resembles_tonights_run_matchup_counts_for_the_run_pair_whatever_its_passing():
    """The point of the per-unit searches: a past game equal to tonight on the run offense and the run defense it faced, and unlike tonight on every
    passing unit, is the run pair's best match at similarity 1 (the units averaged together would have diluted it)."""
    inp = synthetic_inputs(0)
    vec = C.build_vectors(inp)
    tg = _team_target(inp)
    H = inp.games.filter((pl.col("season") == 2019) & (pl.col("week") == 6) & ~pl.col("team").is_in([tg.team, tg.opponent])
                         & ~pl.col("opponent").is_in([tg.team, tg.opponent])).row(0, named=True)
    t2 = vec.team
    for unit, src_team, dst_team in (("run_offense", tg.team, H["team"]), ("run_defense", tg.opponent, H["opponent"])):
        ver_src = C.target_version(unit) if unit in cs.OFFENSE_UNITS else "healthy"
        ver_dst = C.pool_version(unit) if unit in cs.OFFENSE_UNITS else "healthy"
        src = t2.filter((pl.col("game_id") == tg.game_id) & (pl.col("team") == src_team) & (pl.col("unit") == unit) & (pl.col("version") == ver_src))
        sel = (pl.col("game_id") == H["game_id"]) & (pl.col("team") == dst_team) & (pl.col("unit") == unit) & (pl.col("version") == ver_dst)
        new = t2.filter(sel).drop("z", "raw", "n", "n_present", "complete").join(src.select("space", "window", "z", "raw", "n", "n_present", "complete"),
                                                                                 on=["space", "window"], how="inner")
        t2 = pl.concat([t2.filter(~sel), new.select(t2.columns)])
    pool, _, _ = _pool(inp, C.Vectors(t2, vec.player, vec.features, vec.completeness, vec.lineup_change))
    res = pool.search(tg, which=("S4",), version="healthy")
    top = res[RUN4].matches.row(0, named=True)
    assert (top["obs_game_id"], top["obs_team"]) == (H["game_id"], H["team"]) and top["sim_combined"] == pytest.approx(1.0, abs=1e-9)
    pas = res["S4:pass_offense*pass_coverage"].matches
    planted = pas.filter((pl.col("obs_game_id") == H["game_id"]) & (pl.col("obs_team") == H["team"]))
    assert planted.height == 0 or planted["sim_combined"][0] < 0.999                     # its passing does not resemble tonight's


def test_the_across_search_cap_divides_by_the_number_of_unit_searches():
    """X holds 30% of four of the target's eight unit searches: 1.2 / 8 = 15%, at the cap, so nothing moves (dividing by five searches would read 24%
    and cut X)."""
    tg = C.Target("spread", "G", "T", "O", 2022, 5)
    rows = lambda u: [(f"{u}g{i}", "X" if i < 3 else f"{u}T{i}", 1.0) for i in range(10)]
    res = {u: _fake_result(u, rows(u)) for u in ("S2:run_offense", "S2:pass_offense", "S4:run_offense*run_defense", "S4:pass_offense*pass_coverage")}
    res |= {u: _empty_result(u) for u in ("S1:run_defense", "S1:pass_rush", "S1:pass_coverage", "S5")}
    zl = {"team_plays": {(g, t): 1.0 for r in res.values() if r.matches is not None for g, t in zip(r.matches["obs_game_id"], r.matches["obs_team"])}}
    vals, det, _, caps = C._market_shifts(res, tg, "spread", zl, cs.MIN_NEFF)
    for u in ("S2:run_offense", "S2:pass_offense", "S4:run_offense*run_defense", "S4:pass_offense*pass_coverage"):
        assert np.allclose(caps["vol"][u], 1.0), u


def test_only_unit_searches_that_read_a_lineup_corrected_vector_count_as_changed(shifts44):
    """Defense units and archetypes have one version: S1, S3 and S2:<archetype> retrieve the same games on either vector and are not 'changed'."""
    (_, _, _, retrieval), _, _ = shifts44
    one_version = retrieval.filter(pl.col("search").str.starts_with("S1") | pl.col("search").str.starts_with("S3") | pl.col("search").str.contains("archetype"))
    assert one_version.height > 0 and not one_version["adjusted_differs"].any()
    assert retrieval.filter(pl.col("search").str.starts_with("S4"))["adjusted_differs"].any()

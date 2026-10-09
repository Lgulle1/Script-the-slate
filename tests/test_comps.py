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


def synthetic_inputs(seed: int = 0, perturb_from: tuple | None = None, kill_units=(), perturb_lineups: bool = True) -> C.Inputs:
    """A small league (8 teams, 2 seasons x 10 weeks) with random ledgers. `perturb_from=(season, week)` redraws every stat, lineup, slot and 4a
    expectation of the games at or after that week with a different seed, leaving everything earlier untouched. `kill_units` zero the ledger of a unit."""
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
    return C.Inputs(games=games, lineups=lineups, team=team, player=player, slots=slots_df, positions=positions, detail=detail)


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
    res = pool.search(tg, which=("S4",))["S4"]
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
    assert pool.search(_player_target(inp), which=("S3",))["S3"].summary["applicable"]


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
            if s != "S1":
                assert (m["continuity_weight"] == 1.0).all()                         # continuity compares an entity's own earlier games with its present lineup
            else:
                assert (m["continuity_weight"] <= 1.0).all()
    assert found >= 4


def test_n_eff_is_the_kish_effective_sample_size_of_the_final_weights(pool3):
    pool, inp, _ = pool3
    r = pool.search(_player_target(inp, week=9), keep=True, sim_threshold=0.0, min_neff=0)["S4"]
    w = r.matches["final_weight"].to_numpy()
    assert r.summary["n_eff"] == pytest.approx(w.sum() ** 2 / (w ** 2).sum())
    assert r.summary["n_matches"] == len(w) and r.summary["best_similarity"] == pytest.approx(r.matches["sim_combined"].max())


def test_no_match_is_declared_for_a_low_n_eff_even_with_a_good_best_similarity(pool3):
    pool, inp, _ = pool3
    r = pool.search(_player_target(inp, week=9), keep=True, sim_threshold=0.0, min_neff=1e9)["S4"]
    assert r.summary["no_match"] and r.summary["reason"] == "n_eff_below_minimum" and r.summary["shift"] == 0.0 and r.summary["best_similarity"] > 0


def test_the_pool_never_contains_the_target_game_or_anything_after(pool3):
    pool, inp, _ = pool3
    for tg in (_team_target(inp, week=9), _player_target(inp, week=9)):
        key = tg.season * 100 + tg.week
        for s, r in pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0).items():
            if r.matches is not None and r.matches.height:
                assert ((r.matches["obs_season"] * 100 + r.matches["obs_week"]) < key).all()


def test_search_roles_are_disjoint_as_defined(pool3):
    pool, inp, _ = pool3
    tg = _player_target(inp, week=10)
    res = pool.search(tg, keep=True, sim_threshold=0.0, min_neff=0)
    s1, s2, s3, s4 = (res[k].matches for k in ("S1", "S2", "S3", "S4"))
    assert (s1["obs_player_id"] == tg.player_id).all()                                  # S1: the player's own history
    opp_of = lambda m: [inp.games.filter((pl.col("game_id") == g) & (pl.col("team") == t))["opponent"][0] for g, t in zip(m["obs_game_id"], m["obs_team"])]
    assert all(o == tg.opponent for o in opp_of(s2))                                   # S2: tonight's defense
    for m in (s3, s4):                                                                  # S3 / S4: other players against other defenses
        assert (m["obs_player_id"] != tg.player_id).all() and all(o != tg.opponent for o in opp_of(m))


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

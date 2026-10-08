"""The four tables Phase 4 reads, built from the Phase 3 models and features only (no comparables, injuries or
simulation), over 2020-2024 (config.FEATURE_HISTORY_START .. the last backtest season; 2016-2019 is stored but unused).

  walkforward_predictions  one row per player-game / team-game per quantity: the comp-free expectation, the actual, the residual
  market_predictions       one row per player-game / game per market: the combined model, the five baselines, the best baseline, the actual
  player_game_roles        one row per player-game: expected role (pre-game depth chart), played role (snap counts), snap and usage shares
  team_game_rates          one row per team-game: plays, dropbacks, pass rate, points, sack/scramble rates, score-state shares and dropback rates

Timing contract. Every row carries `cutoff_date` (the first kickoff of its week: the cutoff every pre-game column in the row
was built with) and `outcome_known_from` (the day after the game: when the row's outcome columns -- actual, residual, played_role,
snap and usage shares, plays, points, score-state rates -- first became public). A pre-game column never uses information from
its own week or later; a consumer predicting week W may use a row's outcome columns only if outcome_known_from <= W's cutoff_date.
"""
from __future__ import annotations

from datetime import timedelta

import duckdb
import polars as pl

import config
from eval import baselines as bl

MODEL_VERSION = f"phase3_{config.ELIGIBLE_PLAYER_RULE['version']}"
STATES = ("trail9", "trail1_8", "tied", "lead1_8", "lead9")  # trailing 9+, trailing 1-8, tied, leading 1-8, leading 9+
ROLES = ("QB1", "RB1", "RB2", "FB", "WR1", "WR2", "WR3", "slot", "TE1", "TE2", "other")
_ROLE_PRIORITY = {r: i for i, r in enumerate(ROLES)}


def week_cutoffs(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """(season, week) -> cutoff_date: the first kickoff of the week, the same cutoff the walk-forward harness uses."""
    t = bl.build_team_game_log(raw_db, max_season)
    return t.group_by("season", "week").agg(cutoff_date=pl.col("gameday").min()).sort("season", "week")


def _timing(df: pl.DataFrame, cutoffs: pl.DataFrame, gameday_col: str = "gameday") -> pl.DataFrame:
    """Adds cutoff_date and outcome_known_from (the day after the game)."""
    return (df.join(cutoffs, on=["season", "week"], how="left")
              .with_columns(outcome_known_from=pl.col(gameday_col) + timedelta(days=1)))


# ------------------------------------------------------------------ 4. team_game_rates
def build_team_game_rates(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """One row per team-game from the play-by-play (regular season).

    plays           scrimmage plays: play_type pass or run (kneel-downs, spikes, penalties-no-play excluded)
    dropbacks       plays with qb_dropback = 1: pass attempts + sacks + scrambles
    pass_rate       dropbacks / plays
    rush_attempts   run plays (designed runs + scrambles);  designed_runs = rush_attempts - scrambles
                    (so plays = dropbacks + designed_runs; scrambles sit inside dropbacks)
    sacks_per_dropback, scrambles_per_dropback
    share_<state>   share of plays run in that score state (the offense's lead before the snap)
    dropback_rate_<state>   dropbacks / plays within the state (null when the team ran no plays in it)
    States: trail9 (down 9+), trail1_8, tied, lead1_8, lead9 (up 9+).
    Known exception: 2020_16_PHI_DAL (PHI) has plays = dropbacks + designed_runs + 1. One of its plays is typed `pass` but
    flagged qb_scramble = 1 and sack = 1 (a scramble that ended in a sack); it is counted in dropbacks and in scrambles but not
    in rush_attempts, so designed_runs (= rush_attempts - scrambles) comes out 1 too low for that team-game. Every other
    team-game satisfies the identity.
    Note: Phase-3 team_volume counts dropbacks as attempts + sacks (scrambles are carries there); this table follows nflfastR.
    """
    config.cap_season(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        pbp = con.execute(
            "SELECT game_id, season, week, posteam AS team, defteam AS opponent, play_type, "
            "coalesce(qb_dropback, 0) AS qb_dropback, coalesce(qb_scramble, 0) AS qb_scramble, coalesce(sack, 0) AS sack, "
            "score_differential FROM (SELECT DISTINCT ON (game_id, play_id) * FROM pbp WHERE season_type = 'REG'"
            f"{bl._season_cap(max_season)} ORDER BY game_id, play_id, pulled_at DESC) "
            "WHERE play_type IN ('pass', 'run') AND posteam IS NOT NULL AND score_differential IS NOT NULL").pl()
        sched = con.execute(
            "SELECT game_id, gameday, home_team, away_team, home_score, away_score FROM "
            "(SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) "
            "WHERE game_type = 'REG' AND home_score IS NOT NULL").pl()
    finally:
        con.close()
    d = pbp.with_columns(
        state=pl.when(pl.col("score_differential") <= -9).then(pl.lit("trail9"))
        .when(pl.col("score_differential") < 0).then(pl.lit("trail1_8"))
        .when(pl.col("score_differential") == 0).then(pl.lit("tied"))
        .when(pl.col("score_differential") < 9).then(pl.lit("lead1_8")).otherwise(pl.lit("lead9")),
        run=(pl.col("play_type") == "run").cast(pl.Int32))
    key = ["game_id", "season", "week", "team", "opponent"]
    agg = d.group_by(key).agg(
        plays=pl.len(), dropbacks=pl.col("qb_dropback").sum(), rush_attempts=pl.col("run").sum(),
        scrambles=pl.col("qb_scramble").sum(), sacks=pl.col("sack").sum())
    per_state = d.group_by([*key, "state"]).agg(n=pl.len(), db=pl.col("qb_dropback").sum())
    wide = agg
    for s in STATES:
        st = (per_state.filter(pl.col("state") == s).select(*key, **{f"_n_{s}": pl.col("n"), f"_db_{s}": pl.col("db")}))
        wide = wide.join(st, on=key, how="left")
    wide = wide.with_columns([pl.col(f"_n_{s}").fill_null(0) for s in STATES] + [pl.col(f"_db_{s}").fill_null(0) for s in STATES])
    out = wide.with_columns(
        pass_rate=pl.col("dropbacks") / pl.col("plays"), designed_runs=pl.col("rush_attempts") - pl.col("scrambles"),
        sacks_per_dropback=pl.col("sacks") / pl.col("dropbacks"), scrambles_per_dropback=pl.col("scrambles") / pl.col("dropbacks"),
        **{f"share_{s}": pl.col(f"_n_{s}") / pl.col("plays") for s in STATES},
        **{f"dropback_rate_{s}": pl.when(pl.col(f"_n_{s}") > 0).then(pl.col(f"_db_{s}") / pl.col(f"_n_{s}")).otherwise(None) for s in STATES},
    ).drop([f"_n_{s}" for s in STATES] + [f"_db_{s}" for s in STATES])
    out = (out.join(sched, on="game_id", how="inner")
              .with_columns(is_home=pl.col("team") == pl.col("home_team"),
                            points_scored=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("home_score")).otherwise(pl.col("away_score")),
                            points_allowed=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_score")).otherwise(pl.col("home_score")))
              .with_columns(pl.col("gameday").str.to_date()))
    cutoffs = week_cutoffs(raw_db, max_season)
    out = _timing(out, cutoffs)
    cols = ["season", "week", "game_id", "team", "opponent", "is_home", "plays", "dropbacks", "pass_rate", "rush_attempts", "designed_runs",
            "points_scored", "points_allowed", "sacks", "scrambles", "sacks_per_dropback", "scrambles_per_dropback",
            *[f"share_{s}" for s in STATES], *[f"dropback_rate_{s}" for s in STATES], "cutoff_date", "outcome_known_from"]
    return out.select(cols).sort("season", "week", "game_id", "team")


# ------------------------------------------------------------------ 3. player_game_roles
_SNAP_POS = {"QB": "QB", "RB": "RB", "HB": "RB", "RB/W": "RB", "RB/F": "RB", "FB": "FB", "FB/D": "FB", "FB/R": "FB", "FB/T": "FB",
             "WR": "WR", "WR/R": "WR", "TE": "TE"}
_PS_POS = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "FB", "WR": "WR", "TE": "TE"}


def _trailing_usage(raw_db, max_season, cutoffs: pl.DataFrame) -> pl.DataFrame:
    """Per (player_id, season, week): mean carries and targets over the player's previous 4 games played, as of the
    week's cutoff_date (games on or after the cutoff, i.e. the week itself, never count). Used only to order players who
    share a depth_team number on the pre-game chart."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        ps = con.execute(
            f"SELECT player_id, season, week, game_id, team, coalesce(carries, 0) AS carries, coalesce(targets, 0) AS targets "
            f"FROM {bl._latest(con, 'player_stats', 'player_id, season, week')} WHERE season_type = 'REG'{bl._season_cap(max_season)}").pl()
    finally:
        con.close()
    gd = bl.build_team_game_log(raw_db, max_season).select("game_id", "team", "gameday")
    ps = ps.join(gd, on=["game_id", "team"], how="inner").sort("player_id", "gameday")
    ps = ps.with_columns(use_carries=pl.col("carries").cast(pl.Float64).rolling_mean(4, min_samples=1).over("player_id"),
                         use_targets=pl.col("targets").cast(pl.Float64).rolling_mean(4, min_samples=1).over("player_id"))
    return ps.select("player_id", "gameday", "use_carries", "use_targets")


def _expected_roles(raw_db, max_season, cutoffs: pl.DataFrame) -> pl.DataFrame:
    """(season, week, team, player_id) -> expected_role from the weekly pre-game depth chart."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT season, week, club_code AS team, gsis_id AS player_id, position, depth_position, depth_team FROM "
            "(SELECT DISTINCT ON (season, week, club_code, gsis_id, depth_team, position) * FROM depth_charts "
            f" WHERE season IS NOT NULL AND week IS NOT NULL AND game_type = 'REG' AND formation = 'Offense' "
            f" AND gsis_id IS NOT NULL{bl._season_cap(max_season)} ORDER BY season, week, club_code, gsis_id, depth_team, position, pulled_at DESC)").pl()
    finally:
        con.close()
    d = d.with_columns(dt=pl.col("depth_team").cast(pl.Int32, strict=False),
                       group=pl.when((pl.col("position") == "FB") | (pl.col("depth_position") == "FB")).then(pl.lit("FB"))
                       .otherwise(pl.col("position"))).filter(pl.col("group").is_in(["QB", "RB", "FB", "WR", "TE"])).drop_nulls("dt")
    d = d.group_by("season", "week", "team", "player_id", "group").agg(pl.col("dt").min())
    # a player listed in two groups (e.g. TE and FB) keeps one: the lowest depth_team, then FB > RB > WR > TE > QB
    gp = {"FB": 0, "RB": 1, "WR": 2, "TE": 3, "QB": 4}
    d = (d.with_columns(_g=pl.col("group").replace_strict(gp)).sort("season", "week", "team", "player_id", "dt", "_g")
          .unique(["season", "week", "team", "player_id"], keep="first", maintain_order=True).drop("_g"))
    d = d.join(cutoffs, on=["season", "week"], how="left").sort("cutoff_date")
    usage = _trailing_usage(raw_db, max_season, cutoffs).sort("gameday")
    d = d.join_asof(usage, left_on="cutoff_date", right_on="gameday", by="player_id", strategy="backward", allow_exact_matches=False, check_sortedness=False)
    d = d.with_columns(use=pl.when(pl.col("group") == "RB").then(pl.col("use_carries")).otherwise(pl.col("use_targets")).fill_null(-1.0))
    key = ["season", "week", "team", "group"]
    d = d.sort([*key, "dt", "use", "player_id"], descending=[False, False, False, False, False, True, False]).with_columns(
        rk=pl.int_range(pl.len()).over(key) + 1)
    role = (pl.when(pl.col("group") == "QB").then(pl.when(pl.col("rk") == 1).then(pl.lit("QB1")).otherwise(pl.lit("other")))
            .when(pl.col("group") == "FB").then(pl.when(pl.col("rk") == 1).then(pl.lit("FB")).otherwise(pl.lit("other")))
            .when(pl.col("group") == "RB").then(pl.when(pl.col("rk") == 1).then(pl.lit("RB1")).when(pl.col("rk") == 2).then(pl.lit("RB2")).otherwise(pl.lit("other")))
            .when(pl.col("group") == "WR").then(pl.when(pl.col("rk") <= 3).then(pl.format("WR{}", pl.col("rk"))).otherwise(pl.lit("other")))
            .otherwise(pl.when(pl.col("rk") == 1).then(pl.lit("TE1")).when(pl.col("rk") == 2).then(pl.lit("TE2")).otherwise(pl.lit("other"))))
    return d.with_columns(expected_role=role).select("season", "week", "team", "player_id", "expected_role")


def build_player_game_roles(raw_db=config.RAW_DUCKDB_PATH, max_season=None, main_db=config.DUCKDB_PATH) -> pl.DataFrame:
    """One row per skill-position player-game (QB/RB/FB/WR/TE who took an offensive snap or recorded a pass attempt, carry
    or target).

    expected_role   pre-game depth chart for that week (QB1, RB1, RB2, FB, WR1-WR3, TE1, TE2, other). Players tied on a
                    depth_team number are ordered by their mean carries (RB) / targets (WR, TE) over the previous 4 games played
                    as of the week's cutoff. A player not on that week's chart is `other` (on_chart = False).
    played_role     the same vocabulary, assigned from the game's snap counts: players ranked by offensive snaps within the
                    position group of the team-game (position as listed by snap_counts).
    snap_share      offense_pct (null for a player with a stat line but no snap row)
    carry_share, target_share, dropback_share   the player's carries / targets / (attempts + sacks) over the team's totals
                    for the game (carries include QB kneel-downs, as nflverse's official count does).
    `slot` is in the role vocabulary but is never assigned: no free source says who lined up in the slot.
    """
    from ingest import ids

    config.cap_season(max_season)
    cutoffs = week_cutoffs(raw_db, max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        snaps = con.execute(
            "SELECT game_id, pfr_player_id, team, position, offense_snaps, offense_pct FROM "
            f"{bl._latest(con, 'snap_counts', 'game_id, pfr_player_id')} WHERE game_type = 'REG' AND offense_snaps > 0{bl._season_cap(max_season)}").pl()
        ps = con.execute(
            "SELECT player_id, game_id, season, week, team, position, coalesce(carries, 0) AS carries, coalesce(targets, 0) AS targets, "
            "coalesce(attempts, 0) + coalesce(sacks_suffered, 0) AS dropbacks FROM "
            f"{bl._latest(con, 'player_stats', 'player_id, season, week')} WHERE season_type = 'REG'{bl._season_cap(max_season)}").pl()
    finally:
        con.close()
    ps = ps.filter(pl.col("game_id").is_not_null())
    totals = ps.group_by("game_id", "team").agg(t_carries=pl.col("carries").sum(), t_targets=pl.col("targets").sum(), t_dropbacks=pl.col("dropbacks").sum())
    stats = ps.filter(pl.col("position").is_in(list(_PS_POS))).filter((pl.col("carries") > 0) | (pl.col("targets") > 0) | (pl.col("dropbacks") > 0))

    snaps = snaps.with_columns(pos=pl.col("position").replace_strict(_SNAP_POS, default=None)).drop_nulls("pos")
    snaps = ids.add_canonical_gsis_id(snaps, main_db)
    snaps = snaps.with_columns(player_id=pl.coalesce("gsis_id", pl.format("pfr:{}", pl.col("pfr_player_id")))).unique(["game_id", "player_id"], keep="first")
    both = snaps.select("game_id", "player_id", "offense_snaps", snap_team=pl.col("team"), snap_pos=pl.col("pos"), snap_share=pl.col("offense_pct")).join(
        stats.select("player_id", "game_id", "season", "week", "team", "position", "carries", "targets", "dropbacks"),
        on=["game_id", "player_id"], how="full", coalesce=True)
    sched = bl.build_team_game_log(raw_db, max_season).select("game_id", "season", "week", "team", "gameday")
    both = both.with_columns(team=pl.coalesce("team", "snap_team"),
                             pos=pl.coalesce(pl.col("snap_pos"), pl.col("position").replace_strict(_PS_POS, default=None))).drop("season", "week", strict=False)
    both = both.join(sched, on=["game_id", "team"], how="inner")            # drops other-season / non-REG leftovers
    both = both.with_columns(pl.col("pos").alias("position"))
    # played role: rank by offensive snaps within the position group of the team-game
    k = ["game_id", "team", "pos"]
    both = both.sort([*k, "offense_snaps", "player_id"], descending=[False, False, False, True, False], nulls_last=True).with_columns(
        rk=pl.int_range(pl.len()).over(k) + 1)
    played = (pl.when(pl.col("pos") == "QB").then(pl.when(pl.col("rk") == 1).then(pl.lit("QB1")).otherwise(pl.lit("other")))
              .when(pl.col("pos") == "FB").then(pl.when(pl.col("rk") == 1).then(pl.lit("FB")).otherwise(pl.lit("other")))
              .when(pl.col("pos") == "RB").then(pl.when(pl.col("rk") == 1).then(pl.lit("RB1")).when(pl.col("rk") == 2).then(pl.lit("RB2")).otherwise(pl.lit("other")))
              .when(pl.col("pos") == "WR").then(pl.when(pl.col("rk") <= 3).then(pl.format("WR{}", pl.col("rk"))).otherwise(pl.lit("other")))
              .otherwise(pl.when(pl.col("rk") == 1).then(pl.lit("TE1")).when(pl.col("rk") == 2).then(pl.lit("TE2")).otherwise(pl.lit("other"))))
    both = both.with_columns(played_role=pl.when(pl.col("offense_snaps").is_null()).then(None).otherwise(played))
    exp = _expected_roles(raw_db, max_season, cutoffs)
    both = both.join(exp, on=["season", "week", "team", "player_id"], how="left")
    both = both.with_columns(on_chart=pl.col("expected_role").is_not_null()).with_columns(pl.col("expected_role").fill_null("other"))
    both = both.join(totals, on=["game_id", "team"], how="left")
    safe = lambda n, d: pl.when(pl.col(d) > 0).then(pl.col(n).fill_null(0).cast(pl.Float64) / pl.col(d)).otherwise(None)
    out = both.with_columns(carry_share=safe("carries", "t_carries"), target_share=safe("targets", "t_targets"),
                            dropback_share=safe("dropbacks", "t_dropbacks"))
    out = _timing(out, cutoffs)
    return (out.select("season", "week", "game_id", "team", "player_id", "position", "expected_role", "played_role", "on_chart",
                       "snap_share", "carry_share", "target_share", "dropback_share", "cutoff_date", "outcome_known_from")
               .sort("season", "week", "game_id", "team", "player_id"))


# ------------------------------------------------------------------ 1 + 2. walk-forward predictions
# harness quantity -> table quantity
QUANTITIES = {"pass_att": "pass_att", "rush_att": "rush_att", "targets": "targets", "qb_rush_att": "qb_rush_att",
              "comp_pct": "comp_rate", "yds_per_cmp": "yds_per_cmp", "ypc": "ypc", "qb_ypc": "qb_ypc",
              "catch_pct": "catch_rate", "yds_per_rec": "yds_per_rec",
              "plays_home": "team_plays", "plays_away": "team_plays", "ppp_home": "pts_per_play", "ppp_away": "pts_per_play"}
BASELINES = bl.METHODS
MODEL_METHOD = "vol_x_eff"


def run_harness(data, seasons) -> dict:
    """Walk-forward runs through eval.backtest: the volume and efficiency models separately (their own quantities), the combined
    model, and the five baselines. Every model sees only games before the week's first kickoff."""
    from eval import backtest as bt
    from features import volume_features as vf
    from models import combine, efficiency, volume

    cap = max(seasons)
    tv = vf.build_team_volume(config.RAW_DUCKDB_PATH, cap)
    extras = volume.plays_extras(data.team_log, tv).join(efficiency.ppp_extras(data.team_log, tv), on="game_id")
    data.game_extras, data.player_extras = extras, efficiency.efficiency_player_actuals
    vp = volume.volume_predictors(vf.load_feature_tables(data))
    ep = efficiency.efficiency_predictors(efficiency.load_efficiency_tables(data))
    out = {"vol": bt.walk_forward("vol", data, *vp, seasons=seasons), "eff": bt.walk_forward("eff", data, *ep, seasons=seasons),
           MODEL_METHOD: bt.walk_forward(MODEL_METHOD, data, *combine.combined_predictors(vp, ep), seasons=seasons)}
    for m in BASELINES:
        out[m] = bt.walk_forward(m, data, *bt.baseline_predictors(m), seasons=seasons)
    return out


def build_walkforward_predictions(runs: dict, data, roles: pl.DataFrame, cutoffs: pl.DataFrame) -> pl.DataFrame:
    """One row per player-game / team-game per quantity. expected = the Phase 3 model's prediction (null where no model had
    enough earlier rows); actual is the realised value; residual = actual - expected."""
    raw = pl.concat([runs["vol"], runs["eff"]]).filter(pl.col("market").is_in(list(QUANTITIES)))
    pgame = data.player_log.select("player_id", "gameday", "game_id", "team")
    players = (raw.filter(pl.col("kind") == "player").join(pgame, left_on=["entity", "gameday"], right_on=["player_id", "gameday"], how="left")
               .with_columns(player_id=pl.col("entity"))
               .join(roles.select("game_id", "player_id", "expected_role"), on=["game_id", "player_id"], how="left")
               .with_columns(role=pl.col("expected_role").fill_null("other")).drop("expected_role"))
    tl = data.team_log.select("game_id", "team", "is_home")
    games = (raw.filter(pl.col("kind") == "game").with_columns(game_id=pl.col("entity"), is_home=pl.col("market").str.ends_with("_home"))
             .join(tl, on=["game_id", "is_home"], how="left").drop("is_home")
             .with_columns(player_id=pl.lit(None, dtype=pl.String), role=pl.lit("team")))
    d = pl.concat([players.select(games.columns), games])
    d = d.with_columns(quantity=pl.col("market").replace_strict(QUANTITIES), expected=pl.col("prediction")).with_columns(
        residual=pl.col("actual") - pl.col("expected"), model_version=pl.lit(MODEL_VERSION),
        trained_through_season=pl.when(pl.col("week") > 1).then(pl.col("season")).otherwise(pl.col("season") - 1),
        trained_through_week=pl.col("week") - 1)
    d = _timing(d, cutoffs)
    return (d.select("season", "week", "game_id", "player_id", "team", "role", "quantity", "expected", "actual", "residual", "model_version",
                     "trained_through_season", "trained_through_week", "cutoff_date", "outcome_known_from")
              .sort("season", "week", "game_id", "quantity", "team", "player_id", nulls_last=True))


def _expanding_best(wide: pl.DataFrame, market: str) -> pl.DataFrame:
    """Per (season, week): the baseline with the lowest loss on rows from EARLIER weeks only (common rows: all five baselines
    predicted), so the label is something a forecaster at the cutoff could have known. Null until a baseline has history."""
    pred_cols = [f"pred_{b}" for b in BASELINES]
    c = wide.drop_nulls(pred_cols)
    loss = lambda col: ((pl.col(col) - pl.col("actual")) ** 2) if market == "moneyline" else (pl.col(col) - pl.col("actual")).abs()
    wk = (c.group_by("season", "week").agg([loss(p).sum().alias(f"l_{p}") for p in pred_cols])
           .sort("season", "week").with_columns([pl.col(f"l_{p}").cum_sum().shift(1).alias(f"l_{p}") for p in pred_cols]))
    best = wk.select("season", "week", best_baseline=pl.concat_list([pl.col(f"l_{p}") for p in pred_cols]).list.arg_min().map_elements(
        lambda i: BASELINES[i] if i is not None else None, return_dtype=pl.String),
        _has=pl.col(pred_cols[0].replace("pred_", "l_pred_")).is_not_null())
    return best.with_columns(best_baseline=pl.when(pl.col("_has")).then(pl.col("best_baseline"))).drop("_has")


def build_market_predictions(runs: dict, data, cutoffs: pl.DataFrame, final_best: dict) -> pl.DataFrame:
    """One row per player-game / game per market: the combined model, the five baselines, which baseline was best, the actual.

    best_baseline        the best baseline on all earlier weeks' common rows at the row's cutoff (leak-free; null before any history)
    best_baseline_final  the best baseline over the WHOLE 2020-2024 comparison (Phase 3's yardstick; uses later weeks, so it is a
                         label for evaluation, never a pre-game input)
    Game markets: player_id is null and team is the home team.
    """
    keys = ["kind", "market", "season", "week", "gameday", "entity"]
    wide = runs[MODEL_METHOD].select(*keys, "actual", model_prediction="prediction")
    for b in BASELINES:
        wide = wide.join(runs[b].select(*keys, **{f"pred_{b}": "prediction"}), on=keys, how="left")
    pgame = data.player_log.select("player_id", "gameday", "game_id", "team")
    home = data.team_log.filter(pl.col("is_home")).select("game_id", home_team="team")
    wide = (wide.join(pgame, left_on=["entity", "gameday"], right_on=["player_id", "gameday"], how="left")
                .join(home, left_on="entity", right_on="game_id", how="left")
                .with_columns(player_id=pl.when(pl.col("kind") == "player").then(pl.col("entity")),
                              game_id=pl.when(pl.col("kind") == "game").then(pl.col("entity")).otherwise(pl.col("game_id")),
                              team=pl.when(pl.col("kind") == "game").then(pl.col("home_team")).otherwise(pl.col("team"))).drop("home_team"))
    parts = []
    for market in sorted(wide["market"].unique().to_list()):
        w = wide.filter(pl.col("market") == market)
        best = _expanding_best(w, market)
        parts.append(w.join(best, on=["season", "week"], how="left"))
    d = pl.concat(parts)
    fb = final_best
    d = d.with_columns(
        best_baseline_final=pl.col("market").replace_strict(fb, default=None),
        best_baseline_prediction=pl.concat_str([pl.lit("pred_"), pl.col("best_baseline")]),
    )
    d = d.with_columns(best_baseline_prediction=pl.coalesce(*[pl.when(pl.col("best_baseline") == b).then(pl.col(f"pred_{b}")) for b in BASELINES]),
                       model_version=pl.lit(MODEL_VERSION))
    d = _timing(d, cutoffs)
    return (d.select("season", "week", "game_id", "player_id", "team", "market", "model_prediction",
                     *[f"pred_{b}" for b in BASELINES], "best_baseline", "best_baseline_prediction", "best_baseline_final", "actual",
                     "model_version", "cutoff_date", "outcome_known_from")
              .sort("season", "week", "game_id", "market", "team", "player_id", nulls_last=True))


def final_best_baselines(runs: dict) -> dict:
    """Best baseline per market over everything in `runs`, judged on identical rows (eval.compare.best_baseline_per_market)."""
    from eval import compare
    t = compare.best_baseline_per_market(pl.concat([runs[b] for b in BASELINES]))
    return dict(t.filter(pl.col("is_best")).select("market", "method").iter_rows())


# ------------------------------------------------------------------ build + persist
TABLES = ("walkforward_predictions", "market_predictions", "player_game_roles", "team_game_rates")


def build_all(seasons=config.BACKTEST_SEASONS, raw_db=config.RAW_DUCKDB_PATH) -> dict:
    """The four tables for `seasons` (a subset of 2020-2024; earlier seasons still feed the models' history)."""
    from eval import backtest as bt

    seasons = tuple(seasons)
    if not set(seasons) <= set(config.BACKTEST_SEASONS):
        raise config.HoldoutError(f"seasons {seasons} are outside the backtest seasons {config.BACKTEST_SEASONS}")
    cap = max(seasons)
    data = bt.load_backtest_data(raw_db)
    cutoffs = week_cutoffs(raw_db, cap)
    roles = build_player_game_roles(raw_db, cap)
    runs = run_harness(data, seasons)
    tables = {
        "walkforward_predictions": build_walkforward_predictions(runs, data, roles, cutoffs),
        "market_predictions": build_market_predictions(runs, data, cutoffs, final_best_baselines(runs)),
        "player_game_roles": roles.filter(pl.col("season").is_in(seasons)),
        "team_game_rates": build_team_game_rates(raw_db, cap).filter(pl.col("season").is_in(seasons)),
    }
    for name, t in tables.items():
        if t.height and t["season"].max() >= config.HOLDOUT_SEASON:
            raise config.HoldoutError(f"{name} contains a {config.HOLDOUT_SEASON}+ row")
    return tables


def persist(tables: dict, db_path=config.DUCKDB_PATH, out_dir=config.PROCESSED_DIR) -> None:
    """Writes each table to <out_dir>/<name>.parquet (no timestamps, so a rebuild is byte-identical) and to DuckDB
    (CREATE OR REPLACE, like the other derived tables)."""
    config.ensure_data_dirs()
    con = duckdb.connect(str(db_path))
    try:
        for name, t in tables.items():
            t.write_parquet(out_dir / f"{name}.parquet")
            con.register("_t", t.to_arrow())
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
            con.unregister("_t")
    finally:
        con.close()

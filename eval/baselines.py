"""Five baseline predictors for 8 player markets and 3 game markets.

Every predictor is a pure function of (history frame, target, cutoff): it sees only
rows with gameday strictly BEFORE `cutoff`, and cutoff may not be later than the
target game's own date. Nothing here reads the database or the full season; the
`build_*` loaders at the bottom make the history frames once, up front.

Conventions
-----------
* History holds only games a player actually appeared in (player_stats has no row for
  a DNP), so averages are per game played. Regular season only.
* "Team game" numbers are 1-based regular-season game counts per team (byes skipped).
* Every predictor returns None for a market when there is no usable history.
* Method names: last3, season_avg, recency, blend_70_30, role_avg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import polars as pl

import config
from features.weights import games_elapsed, recency_weight

# market -> player_stats column
PLAYER_MARKETS = {m: spec["stat"] for m, spec in config.MARKETS.items() if spec["kind"] == "player"}  # market -> player_stats column
GAME_MARKETS = tuple(m for m, spec in config.MARKETS.items() if spec["kind"] == "game")
METHODS = ("last3", "season_avg", "recency", "blend_70_30", "role_avg")
FAMILIES = ("QB", "RB", "WR", "TE")
_POSITION_FAMILY = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}


@dataclass(frozen=True)
class PlayerTarget:
    player_id: str          # gsis_id
    gameday: date
    season: int
    team_game_num: int      # this player's team's regular-season game number
    opponent: str
    family: str             # QB / RB / WR / TE
    slot: int               # depth-chart slot bucket known pregame: 1, 2, 3 (3 = 3rd or lower), 0 = unlisted
    team: str | None = None  # the player's team in the target game (needed to handle mid-season trades)
    week: int | None = None  # regular-season week of the target game


@dataclass(frozen=True)
class GameTarget:
    gameday: date
    season: int
    home_team: str
    away_team: str
    home_game_num: int
    away_game_num: int


def _check_cutoff(target, cutoff: date):
    if cutoff > target.gameday:
        raise ValueError(f"cutoff {cutoff} is after the target game date {target.gameday}")


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------- player baselines
def _player_hist(log: pl.DataFrame, target: PlayerTarget, cutoff: date) -> pl.DataFrame:
    _check_cutoff(target, cutoff)
    return log.filter((pl.col("player_id") == target.player_id) & (pl.col("gameday") < cutoff)).sort("gameday")


def _col_means(df: pl.DataFrame):
    return {m: _mean(df[c].to_list()) for m, c in PLAYER_MARKETS.items()} if df.height else {m: None for m in PLAYER_MARKETS}


def player_last3(log, target, cutoff):
    return _col_means(_player_hist(log, target, cutoff).tail(3))


def player_season_avg(log, target, cutoff):
    h = _player_hist(log, target, cutoff)
    return _col_means(h.filter(pl.col("season") == target.season))


def _games_back(season, team_game_num, team, week, target) -> float:
    """Team games between a past game and the target. A past game for a DIFFERENT team in the same
    season (mid-season trade) has team-game numbers that aren't comparable, so the week difference
    is used instead (off by at most one game, when a bye falls in between)."""
    if season == target.season and target.team is not None and target.week is not None and team != target.team:
        return max(1.0, float(target.week - week))
    return games_elapsed(season, team_game_num, target.season, target.team_game_num)


def player_recency(log, target, cutoff):
    """Recency-weighted average over the player's whole prior history (decay in team games)."""
    h = _player_hist(log, target, cutoff)
    if not h.height:
        return {m: None for m in PLAYER_MARKETS}
    w = [recency_weight(_games_back(s, g, team, wk, target))
         for s, g, team, wk in zip(h["season"].to_list(), h["team_game_num"].to_list(),
                                   h["team"].to_list(), h["week"].to_list())]
    out = {}
    for m, c in PLAYER_MARKETS.items():
        pairs = [(wi, v) for wi, v in zip(w, h[c].to_list()) if v is not None]
        tot = sum(wi for wi, _ in pairs)
        out[m] = sum(wi * v for wi, v in pairs) / tot if tot else None
    return out


def _window(log, cutoff):
    return log.filter((pl.col("gameday") < cutoff) & (pl.col("gameday") >= cutoff - timedelta(days=config.ROLE_WINDOW_DAYS)))


def player_role_avg(log, target, cutoff):
    """Mean over everyone in the same position family + depth-slot bucket (trailing window)."""
    _check_cutoff(target, cutoff)
    g = _window(log, cutoff).filter((pl.col("family") == target.family) & (pl.col("slot") == target.slot))
    return _col_means(g)


def _opp_allowed(log, target, cutoff):
    """Mean of same role group's output against this opponent (trailing window)."""
    g = _window(log, cutoff).filter(
        (pl.col("opponent") == target.opponent) & (pl.col("family") == target.family) & (pl.col("slot") == target.slot))
    return _col_means(g)


def player_blend(log, target, cutoff):
    """70% the player's own average + 30% what the opponent allows that role group.

    Own average = season-to-date, falling back to last-3 (so week 1 isn't empty). If the
    opponent-allowed average is missing for a market the blend falls back to the own average.
    """
    own, l3 = player_season_avg(log, target, cutoff), player_last3(log, target, cutoff)
    opp = _opp_allowed(log, target, cutoff)
    out = {}
    for m in PLAYER_MARKETS:
        o = own[m] if own[m] is not None else l3[m]
        if o is None:
            out[m] = None
        elif opp[m] is None:
            out[m] = o
        else:
            out[m] = config.BLEND_OWN_WEIGHT * o + (1 - config.BLEND_OWN_WEIGHT) * opp[m]
    return out


PLAYER_BASELINES = {"last3": player_last3, "season_avg": player_season_avg, "recency": player_recency,
                    "blend_70_30": player_blend, "role_avg": player_role_avg}


def predict_player(method: str, log: pl.DataFrame, target: PlayerTarget, cutoff: date | None = None) -> dict:
    """All 8 player-market predictions for one player-game. cutoff defaults to the game date."""
    return PLAYER_BASELINES[method](log, target, cutoff or target.gameday)


# ---------------------------------------------------------------- game baselines
# team_log columns: team, opponent, gameday, season, team_game_num, pf, pa
def _team_hist(tlog, team, cutoff):
    return tlog.filter((pl.col("team") == team) & (pl.col("gameday") < cutoff)).sort("gameday")


def _team_series(tlog, team, cutoff, season, game_num, method, col):
    """One method's estimate of a team's pf or pa (own-average methods only)."""
    h = _team_hist(tlog, team, cutoff)
    if not h.height:
        return None
    if method == "last3":
        return _mean(h.tail(3)[col].to_list())
    if method == "season_avg":
        return _mean(h.filter(pl.col("season") == season)[col].to_list())
    if method == "recency":
        w = [recency_weight(games_elapsed(s, g, season, game_num))
             for s, g in zip(h["season"].to_list(), h["team_game_num"].to_list())]
        return sum(wi * v for wi, v in zip(w, h[col].to_list())) / sum(w)
    raise ValueError(method)


def _team_points(tlog, team, opp, cutoff, season, game_num, opp_game_num, method):
    if method in ("last3", "season_avg", "recency"):
        return _team_series(tlog, team, cutoff, season, game_num, method, "pf")
    if method == "blend_70_30":
        own = _team_series(tlog, team, cutoff, season, game_num, "season_avg", "pf")
        if own is None:
            own = _team_series(tlog, team, cutoff, season, game_num, "last3", "pf")
        allowed = _team_series(tlog, opp, cutoff, season, opp_game_num, "season_avg", "pa")
        if allowed is None:
            allowed = _team_series(tlog, opp, cutoff, season, opp_game_num, "last3", "pa")
        if own is None:
            return None
        return own if allowed is None else config.BLEND_OWN_WEIGHT * own + (1 - config.BLEND_OWN_WEIGHT) * allowed
    if method == "role_avg":  # game-level analogue: league-average points per team-game
        g = _window(tlog, cutoff)
        return _mean(g["pf"].to_list()) if g.height else None
    raise ValueError(method)


def predict_game(method: str, tlog: pl.DataFrame, target: GameTarget, cutoff: date | None = None) -> dict:
    """Predicted home margin (`spread`, same sign as nflverse spread_line/result),
    P(home win) (`moneyline`, normal CDF of margin / GAME_MARGIN_SD), and `total`."""
    cutoff = cutoff or target.gameday
    _check_cutoff(target, cutoff)
    home = _team_points(tlog, target.home_team, target.away_team, cutoff, target.season,
                        target.home_game_num, target.away_game_num, method)
    away = _team_points(tlog, target.away_team, target.home_team, cutoff, target.season,
                        target.away_game_num, target.home_game_num, method)
    if home is None or away is None:
        return {"spread": None, "moneyline": None, "total": None}
    margin = home - away
    return {"spread": margin, "total": home + away,
            "moneyline": 0.5 * (1 + math.erf(margin / config.GAME_MARGIN_SD / math.sqrt(2)))}


# ---------------------------------------------------------------- history builders
def _latest(con, table, keys):
    return f"(SELECT DISTINCT ON ({keys}) * FROM {table} ORDER BY {keys}, pulled_at DESC)"


def _season_cap(max_season) -> str:
    """SQL season window: config.FEATURE_HISTORY_START .. cap. Raises config.HoldoutError for 2025+; None
    means the last backtest season."""
    return config.season_sql(max_season)


def build_team_game_log(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """One row per team per completed regular-season game: pf, pa, team game number.

    `max_season` is applied in SQL, so later seasons (e.g. the locked holdout) are never loaded; asking for
    the holdout season or later raises config.HoldoutError.
    """
    config.cap_season(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        sched = con.execute(
            f"SELECT game_id, season, week, gameday, home_team, away_team, home_score, away_score "
            f"FROM {_latest(con, 'schedules', 'game_id')} WHERE game_type = 'REG' "
            f"AND home_score IS NOT NULL AND away_score IS NOT NULL{_season_cap(max_season)}").pl()
    finally:
        con.close()
    sched = sched.with_columns(pl.col("gameday").str.to_date())
    home = sched.select("game_id", "season", "week", "gameday", team=pl.col("home_team"), opponent=pl.col("away_team"),
                        pf=pl.col("home_score"), pa=pl.col("away_score"), is_home=pl.lit(True))
    away = sched.select("game_id", "season", "week", "gameday", team=pl.col("away_team"), opponent=pl.col("home_team"),
                        pf=pl.col("away_score"), pa=pl.col("home_score"), is_home=pl.lit(False))
    return (pl.concat([home, away]).sort("team", "season", "gameday")
            .with_columns(team_game_num=pl.col("gameday").rank("ordinal").over("team", "season").cast(pl.Int32)))


def _slot_bucket(rank):
    return None if rank is None else min(max(int(rank), 1), 3)


def build_role_table(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Depth-chart roles as (weekly 2020-24 frame, timestamped 2025+ frame): family + slot bucket.

    2020-24 charts are weekly (season/week/depth_team); 2025+ charts are timestamped
    snapshots (dt/pos_rank). Both are mapped to (family, slot 1/2/3). The newer snapshots
    are matched to games as-of the day BEFORE the game, so they can't leak game-day moves.
    """
    config.cap_season(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        old = con.execute(
            "SELECT season, week, club_code AS team, gsis_id, position, depth_team FROM "
            "(SELECT DISTINCT ON (season, week, club_code, gsis_id, depth_team, position) * FROM depth_charts "
            f" WHERE season IS NOT NULL AND formation = 'Offense'{_season_cap(max_season)} ORDER BY season, week, club_code, gsis_id, "
            " depth_team, position, pulled_at DESC)").pl()
        # Timestamped snapshots only exist from 2025; skip them entirely when capped below that.
        new = (con.execute(
            "SELECT team, gsis_id, pos_abb AS position, pos_rank, dt FROM depth_charts WHERE season IS NULL "
            "AND pos_grp NOT IN ('Base 4-3 D', 'Base 3-4 D', 'Special Teams')").pl()
            if config.cap_season(max_season) >= config.HOLDOUT_SEASON else
            pl.DataFrame(schema={"team": pl.String, "gsis_id": pl.String, "position": pl.String,
                                 "pos_rank": pl.String, "dt": pl.String}))
    finally:
        con.close()
    old = (old.with_columns(family=pl.col("position").replace_strict(_POSITION_FAMILY, default=None),
                            slot=pl.col("depth_team").cast(pl.Int32, strict=False).clip(1, 3))
           .drop_nulls(["family", "slot"])
           .group_by("season", "week", "team", "gsis_id").agg(pl.col("family").first(), pl.col("slot").min()))
    new = (new.with_columns(family=pl.col("position").replace_strict(_POSITION_FAMILY, default=None),
                            slot=pl.col("pos_rank").cast(pl.Int32, strict=False).clip(1, 3),
                            snap=pl.col("dt").str.to_datetime(time_zone="UTC", strict=False).dt.replace_time_zone(None))
           .drop_nulls(["family", "slot", "snap"]))
    return old, new


def build_player_game_log(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """One row per player-game (regular season, QB/RB/WR/TE): stats + pregame role.

    `max_season` is applied in SQL, so later seasons are never loaded; the holdout season or later raises
    config.HoldoutError.
    """
    config.cap_season(max_season)
    team_log = build_team_game_log(raw_db, max_season)
    old_roles, new_roles = build_role_table(raw_db, max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        cols = ", ".join(PLAYER_MARKETS.values())
        ps = con.execute(
            f"SELECT player_id, season, week, game_id, team, opponent_team AS opponent, position, {cols} FROM "
            f"{_latest(con, 'player_stats', 'player_id, season, week')} WHERE season_type = 'REG' "
            f"AND position IN ('QB','RB','HB','FB','WR','TE'){_season_cap(max_season)}").pl()
    finally:
        con.close()
    ps = ps.with_columns(family=pl.col("position").replace_strict(_POSITION_FAMILY, default=None))
    ps = ps.join(team_log.select("game_id", "team", "gameday", "team_game_num"), on=["game_id", "team"], how="inner")
    # weekly (2020-24) roles
    ps = ps.join(old_roles.rename({"gsis_id": "player_id", "slot": "slot_old"}).drop("family"),
                 on=["season", "week", "team", "player_id"], how="left")
    # timestamped (2025+) roles: latest snapshot strictly before the game day
    new_sorted = new_roles.select("team", player_id=pl.col("gsis_id"), snap="snap", slot_new="slot").sort("snap")
    ps = ps.with_columns(game_start=pl.col("gameday").cast(pl.Datetime)).sort("game_start")
    ps = ps.join_asof(new_sorted, left_on="game_start", right_on="snap", by=["team", "player_id"],
                      strategy="backward", allow_exact_matches=False, check_sortedness=False)
    ps = ps.with_columns(slot=pl.coalesce("slot_old", "slot_new").fill_null(0).cast(pl.Int32)).drop(
        "slot_old", "slot_new", "snap", "game_start", "position")
    return ps.drop_nulls("family").sort("player_id", "gameday")

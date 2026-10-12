"""Targets of a live run: the games about to be played and the players eligible in them, as placeholder rows (no outcomes) for the feature builders.

The feature builders (features/volume_features.py, models/efficiency.py) already keep rows whose outcome is unknown and compute every feature from games
strictly before the week's cutoff, so a target is just a log row with its stats null. This module makes those rows:

  team rows    game_id, season, week, gameday, team, opponent, is_home, team_game_num (from the full schedule, byes do not count), pf / pa null
  player rows  every candidate on the team's offensive depth chart (live/features: features.depth_live.chart_as_of) plus anyone who played for the team in its
               last LOOKBACK_GAMES games; family from the chart position (fullbacks fold into RB), slot = best chart rank clipped to 1-3 (0 = unlisted),
               chart flags as eval/baselines.build_chart_flags, usage = mean of his previous up-to-4 games played; `elig` from eval/eligibility.add_eligibility,
               the same rule as the backtest. Stats null.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

import config
from eval import eligibility as elig

ET = ZoneInfo("America/New_York")
LOOKBACK_GAMES = 4
POSITION_FAMILY = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
STATS = ["attempts", "completions", "passing_yards", "carries", "rushing_yards", "targets", "receptions", "receiving_yards", "rush_att_ex_kneel", "rush_yds_ex_kneel"]


def kickoff_utc(gameday: str, gametime: str) -> datetime:
    """Kickoff as a UTC datetime from nflverse's Eastern-time gameday ('2026-10-18') and gametime ('13:00')."""
    h, m = (int(x) for x in gametime.split(":")[:2])
    d = date.fromisoformat(gameday)
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET).astimezone(timezone.utc)


def schedule_frame(sched: pl.DataFrame) -> pl.DataFrame:
    """The regular-season rows of a schedules table with a `gameday` date and a UTC `kickoff_time`; games without a kickoff time are dropped (never guessed)."""
    s = sched.filter((pl.col("game_type") == "REG") & pl.col("gametime").is_not_null() & pl.col("gameday").is_not_null())
    return s.with_columns(
        kickoff_time=pl.struct("gameday", "gametime").map_elements(lambda r: kickoff_utc(r["gameday"], r["gametime"]), return_dtype=pl.Datetime("us", "UTC")),
        gameday=pl.col("gameday").str.to_date())


def target_games(sched: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return schedule_frame(sched).filter((pl.col("season") == season) & (pl.col("week") == week)).sort("kickoff_time", "game_id")


def team_rows(sched: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Placeholder team-log rows for `games`; team_game_num counts the team's games of the season by date over the FULL schedule."""
    full = schedule_frame(sched)
    allrows = pl.concat([
        full.select("game_id", "season", "week", "gameday", team=pl.col("home_team"), opponent=pl.col("away_team"), is_home=pl.lit(True)),
        full.select("game_id", "season", "week", "gameday", team=pl.col("away_team"), opponent=pl.col("home_team"), is_home=pl.lit(False))])
    allrows = allrows.with_columns(team_game_num=pl.col("gameday").rank("ordinal").over("team", "season").cast(pl.Int32))
    return allrows.join(games.select("game_id"), on="game_id", how="inner").with_columns(
        pf=pl.lit(None, dtype=pl.Int32), pa=pl.lit(None, dtype=pl.Int32)).sort("team", "gameday")


def _chart_players(chart: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, player) from an offensive chart (features.depth_live.chart_as_of): family, slot, chart flags."""
    d = chart.filter(pl.col("position").is_in(["QB", "RB", "FB", "WR", "TE"])).with_columns(rank=pl.col("depth_team").cast(pl.Int32, strict=False)).drop_nulls("rank")
    rule = config.ELIGIBLE_PLAYER_RULE["depth_chart"]
    key = ["team"]
    rb = d.filter(pl.col("position") == "RB").with_columns(r=pl.col("rank").rank("dense").over(key))
    fb = d.filter(pl.col("position") == "FB").with_columns(r=pl.col("rank").rank("dense").over(key))
    flags = pl.concat([
        rb.filter(pl.col("r").is_in(list(rule["RB"]))).select("team", "gsis_id", flag=pl.lit("chart_rb")),
        fb.filter(pl.col("r").is_in(list(rule["FB"]))).select("team", "gsis_id", flag=pl.lit("chart_fb")),
        d.filter((pl.col("position") == "WR") & pl.col("rank").is_in(list(rule["WR"]))).select("team", "gsis_id", flag=pl.lit("chart_wr")),
        d.filter((pl.col("position") == "TE") & pl.col("rank").is_in(list(rule["TE"]))).select("team", "gsis_id", flag=pl.lit("chart_te"))]).unique()
    base = d.group_by("team", "gsis_id").agg(family=pl.col("position").first().replace_strict(POSITION_FAMILY, default=None), slot=pl.col("rank").min().clip(1, 3).cast(pl.Int32))
    for c in elig.CHART_COLUMNS:
        base = base.join(flags.filter(pl.col("flag") == c).select("team", "gsis_id", pl.lit(True).alias(c)), on=["team", "gsis_id"], how="left")
    return base.with_columns([pl.col(c).fill_null(False) for c in elig.CHART_COLUMNS]).rename({"gsis_id": "player_id"})


def _recent_players(player_log: pl.DataFrame, team_log: pl.DataFrame, teams: list, n_games: int = LOOKBACK_GAMES) -> pl.DataFrame:
    """Players who played for a team in its last n_games (history), for usage-eligible players who are not on the chart."""
    last = (team_log.filter(pl.col("team").is_in(teams)).sort("team", "gameday").group_by("team").agg(pl.col("game_id").tail(n_games).alias("g")).explode("g"))
    return (player_log.join(last.rename({"g": "game_id"}), on=["team", "game_id"], how="inner")
            .group_by("team", "player_id").agg(family=pl.col("family").last()).with_columns(slot=pl.lit(0, dtype=pl.Int32)))


def usage_prev(player_log: pl.DataFrame, ids: list) -> pl.DataFrame:
    """Per player: the mean of his previous up-to-4 games played, as eval/eligibility.add_usage_averages gives a game right after the last one in the log."""
    n = config.ELIGIBLE_PLAYER_RULE["usage"]["window_games"]
    c, t, q = elig.usage_columns()
    h = player_log.filter(pl.col("player_id").is_in(ids)).sort("player_id", "gameday")
    return h.group_by("player_id", maintain_order=True).agg(
        pl.col("carries").cast(pl.Float64).tail(n).mean().alias(c), pl.col("targets").cast(pl.Float64).tail(n).mean().alias(t),
        pl.col("rush_att_ex_kneel").cast(pl.Float64).tail(n).mean().alias(q))


def player_rows(games: pl.DataFrame, chart: pl.DataFrame, history_players: pl.DataFrame, history_teams: pl.DataFrame, trows: pl.DataFrame) -> pl.DataFrame:
    """Placeholder player-log rows (stats null, `elig` set) for every candidate of every team in `games`."""
    teams = trows["team"].unique().to_list()
    on_chart = _chart_players(chart.filter(pl.col("team").is_in(teams)))
    recent = _recent_players(history_players, history_teams, teams)
    cand = pl.concat([on_chart.select("team", "player_id", "family", "slot", *elig.CHART_COLUMNS),
                      recent.join(on_chart.select("player_id").unique(), on="player_id", how="anti").select(           # a player on any team's chart is not also a "recent player" of his old team
                          "team", "player_id", "family", "slot", *[pl.lit(False).alias(c) for c in elig.CHART_COLUMNS])]).drop_nulls("family")
    cand = cand.join(usage_prev(history_players, cand["player_id"].to_list()), on="player_id", how="left")
    for c in elig.usage_columns():
        if c not in cand.columns:
            cand = cand.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    rows = cand.join(trows.select("game_id", "season", "week", "gameday", "team", "opponent", "team_game_num"), on="team", how="inner")
    rows = elig.add_eligibility(rows)
    return rows.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in STATS]).sort("player_id", "gameday")

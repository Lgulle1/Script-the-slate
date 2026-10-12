"""Injury features for the volume and efficiency models (4a.4): built week by week so no row sees its own week or later.

For every regular-season week W of the backtest seasons, with W's first kickoff as the cutoff:
  * the status model (4a.1), the role-shift table (4a.2) and the early-exit model (4a.3) are fit on games before the cutoff;
  * each team-game is run through models.injuries.game_expected_shares with the statuses posted before the game's calendar day
    (models.injuries.main_run_as_of).

Player features (keyed by player_id + game_id; null for a player the pre-game team list does not contain):
  inj_exp_snap_share   P(play) * snap_share_given_play for his status (4a.1)
  inj_p_out            his probability of being out beyond the healthy baseline (0 for no designation, 1 for IR / PUP / suspended)
  inj_p_exit           P(early exit) (4a.3)
  inj_exp_carry_share / inj_exp_target_share / inj_exp_dropback_share / inj_exp_snap   expected shares after redistribution (4a.2)
  inj_d_carry / inj_d_target / inj_d_dropback  expected minus baseline: how far this game's injuries move his share
Team features (keyed by game_id + team; the opponent's are joined as opp_*):
  inj_team_qb_out, inj_team_skill_out (sum of p_out over WR1-3, RB1, TE1), inj_team_ol_out (sum over the usual OL starters)
"""
from __future__ import annotations

import polars as pl

import config
from models import injuries as ij

PLAYER_COLUMNS = ["inj_exp_snap_share", "inj_p_out", "inj_p_exit", "inj_exp_carry_share", "inj_exp_target_share", "inj_exp_dropback_share",
                  "inj_exp_snap", "inj_d_carry", "inj_d_target", "inj_d_dropback"]
TEAM_COLUMNS = ["inj_team_qb_out", "inj_team_skill_out", "inj_team_ol_out"]
SKILL_ROLES = ("WR1", "WR2", "WR3", "RB1", "TE1")


def _prepare(pw, frame, injuries, ros):
    """The lookups every week of the build shares."""
    status_fn = ij.make_status_lookup(injuries)
    blocked = {k[:2]: set(g["gsis_id"].to_list()) for k, g in
               ros.filter(pl.col("roster_status").is_in(list(ij.BLOCKED_ROSTER_STATUSES))).partition_by("season", "week", as_dict=True).items()}
    groups = dict(pw.sort("gameday").group_by("gsis_id", maintain_order=True).agg(pl.col("group").last()).iter_rows())
    by_team = {k[0]: g for k, g in frame.partition_by("team", as_dict=True).items()}
    return status_fn, blocked, groups, by_team


def _week_rows(trio, lookups, games: pl.DataFrame, season: int, week: int, cutoff, as_of_of) -> tuple:
    """Player and team feature rows of the given team-games (one week), models in `trio` = (status, shifts, exit), `as_of_of(gameday)` the status clock."""
    model, shifts, exit_model = trio
    status_fn, blocked, groups, by_team = lookups
    prow, trow = [], []
    for g in games.iter_rows(named=True):
        out = ij.game_expected_shares(by_team[g["team"]], shifts, model, None, None, g["game_id"], g["team"], season, week, g["gameday"],
                                      as_of_of(g["gameday"]), cutoff, exit_model=exit_model, status_fn=status_fn,
                                      blocked_ids=blocked.get((season, week), set()), player_groups=groups, with_eff=False)
        listed = out.filter(pl.col("player_id") != "rest")
        for r in listed.iter_rows(named=True):
            prow.append(dict(player_id=r["player_id"], game_id=g["game_id"], inj_exp_snap_share=r["exp_snap_share"], inj_p_out=r["p_out"],
                             inj_p_exit=r["p_exit"], inj_exp_carry_share=r["exp_carry"], inj_exp_target_share=r["exp_target"],
                             inj_exp_dropback_share=r["exp_dropback"], inj_exp_snap=r["exp_snap"],
                             inj_d_carry=r["exp_carry"] - r["base_carry"], inj_d_target=r["exp_target"] - r["base_target"],
                             inj_d_dropback=r["exp_dropback"] - r["base_dropback"], role=r["role"]))
        roles = dict(zip(listed["player_id"], listed["role"]))
        po = dict(zip(listed["player_id"], listed["p_out"]))
        trow.append(dict(game_id=g["game_id"], team=g["team"],
                         inj_team_qb_out=sum(v for k, v in po.items() if roles[k] == "QB"),
                         inj_team_skill_out=sum(v for k, v in po.items() if roles[k] in SKILL_ROLES),
                         inj_team_ol_out=sum(v for k, v in po.items() if roles[k] == "OL")))
    return prow, trow


def _assemble(prow, trow) -> dict:
    players = pl.DataFrame(prow).drop("role").unique(["player_id", "game_id"], keep="first")
    return {"players": players, "teams": pl.DataFrame(trow)}


def build_injury_features(data, raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> dict:
    """{'players': DataFrame, 'teams': DataFrame} for every team-game of `data` (a backtest.BacktestData)."""
    cap = config.cap_season(max_season)
    pw = ij.build_player_weeks(raw_db, cap)
    frame = ij.build_role_frame(raw_db, cap)
    lookups = _prepare(pw, frame, ij.load_injury_rows(raw_db, cap), ij.load_roster_status(raw_db, cap))
    prow, trow = [], []
    for season, week, cutoff in data.weeks:
        trio = (ij.fit_status_model(pw, cutoff), ij.fit_shifts(frame, cutoff), ij.fit_exit_model(pw, cutoff))
        games = data.team_log.filter((pl.col("season") == season) & (pl.col("week") == week))
        p, t = _week_rows(trio, lookups, games, season, week, cutoff, ij.main_run_as_of)
        prow += p
        trow += t
    return _assemble(prow, trow)


def build_live_injury_features(pw, frame, injuries, ros, games: pl.DataFrame, season: int, week: int, cutoff, fit_cutoff, now) -> dict:
    """Injury features of the games about to be played (`games`: game_id, team, gameday), from frames that already hold the target rows (live/history.py).

    The status, shift and early-exit models are fit on games before `fit_cutoff` (2020-2024 rows only, plan P5.4); the team players, baselines and statuses come from the
    history before `cutoff` and the injury reports posted by min(now, the main run of the game's day)."""
    trio = (ij.fit_status_model(pw, fit_cutoff), ij.fit_shifts(frame, fit_cutoff), ij.fit_exit_model(pw, fit_cutoff))
    clock = lambda gameday: min(ij._utc(now), ij.main_run_as_of(gameday))
    p, t = _week_rows(trio, _prepare(pw, frame, injuries, ros), games, season, week, cutoff, clock)
    return _assemble(p, t)


def attach(tables, feats: dict, opponents: pl.DataFrame):
    """FeatureTables with the injury columns joined on: players on (player_id, game_id); teams on (game_id, team) plus the
    opponent's team features as opp_* (`opponents`: game_id, team, opponent from the team log)."""
    from features.volume_features import FeatureTables

    team = feats["teams"]
    opp = (opponents.select("game_id", "team", "opponent").join(team.rename({c: f"opp_{c}" for c in TEAM_COLUMNS}).rename({"team": "opponent"}),
                                                                  on=["game_id", "opponent"], how="left").drop("opponent"))
    team_feats = team.join(opp, on=["game_id", "team"], how="left")

    def add_team(df):
        return df.join(team_feats, on=["game_id", "team"], how="left")

    players = {q: add_team(df.join(feats["players"], on=["player_id", "game_id"], how="left")) for q, df in tables.players.items()}
    return FeatureTables(players, add_team(tables.teams))

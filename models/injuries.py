"""Injury layer (Phase 4a): status-to-probability model (4a.1); role redistribution is added below it (4a.2).

4a.1  From past games, what does a player's injury-report status say about (a) whether he plays and (b) how much?

  P(play)               share of player-games in the group in which the player took any snap (offense, defense or special teams)
  snap_share_given_play his snap share that game / his own trailing NORMAL snap share (0.6 = he played at 60% of his usual load)

Groups are position group x report status x practice status. Thin groups are shrunk toward their parent with n / (n + k):
  (position, report, practice)  ->  (position, report)  ->  (report)          k = SHRINK_K = 10
so a group with fewer than 10 observations never carries more than half the weight of its own raw rate.

What the data can and cannot do (recorded so nobody assumes more):
* The nflverse injuries table has ONE row per player-week: the report status and the practice status last posted, with its
  `date_modified` (about 90% of rows are last modified Friday). It does not keep the Wednesday / Thursday / Friday practice
  trajectory, so the group key uses the practice status of that final row, not the whole week's sequence. A trajectory needs
  a source that stores each day's report; none of the free pulls does.
* There is no separate Sunday inactives snapshot. `late_run_as_of` still exists (statuses posted up to 12:00 UTC on game day), but
  only the handful of rows modified on game day differ from the main run.
* Population for fitting: every player on a team's weekly roster with status ACT or INA (active or inactive for the game) in a
  regular-season week that team played. IR / PUP / suspended / NFI players (RES, PUP, SUS, NWT, RSN, EXE) are not modelled:
  they get P(play) = 0 directly (BLOCKED_ROSTER_STATUSES).

Timing. The fit for week W uses only games dated before W's first kickoff (the walk-forward cutoff), and each past game's
status is the one posted by that game's own `main_run_as_of` (so a status posted after the cutoff is never seen). A prediction
takes an explicit `as_of`; injury rows with date_modified > as_of are treated as not yet posted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

import duckdb
import numpy as np
import polars as pl

import config
from eval import baselines as bl

SHRINK_K = 10
MIN_OBS = 10                       # a group below this never carries half the weight of its own rate
NORMAL_WINDOW = 6                  # games in the trailing "normal" snap share
MIN_NORMAL_GAMES = 2
MIN_NORMAL_PCT = 0.10             # only players with a regular workload are in the fit (see fit_status_model)
RATIO_CAP = 2.0                    # a single game above twice his usual load is clipped
BLOCKED_ROSTER_STATUSES = ("RES", "PUP", "SUS", "NWT", "RSN", "EXE")
NONE = "none"
PRACTICE_CODE = {"Full Participation in Practice": "FP", "Limited Participation in Practice": "LP",
                 "Did Not Participate In Practice": "DNP"}
OFFENSE_GROUPS = {"QB", "RB", "FB", "WR", "TE", "OL", "T", "G", "C"}
SPECIAL_GROUPS = {"K", "P", "LS"}


# ------------------------------------------------------------------ timing helpers
def _utc(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime.combine(d, time(0, 0), tzinfo=timezone.utc)


def main_run_as_of(gameday: date) -> datetime:
    """The main run: everything posted before the game's calendar day (the Friday report for a Sunday game, plus Saturday
    updates). Nothing posted on game day or later."""
    return _utc(gameday) - timedelta(seconds=1)


def late_run_as_of(gameday: date) -> datetime:
    """The late run: also statuses posted on game day up to 12:00 UTC (before any US kickoff)."""
    return _utc(gameday) + timedelta(hours=12)


def clean_practice(s) -> str:
    if s is None:
        return NONE
    return PRACTICE_CODE.get(str(s).strip(), NONE)


def clean_report(s) -> str:
    s = None if s is None else str(s).strip()
    return s if s else NONE


def pos_group(position: str | None) -> str:
    """Position group used for the groups: QB, RB, WR, TE, OL, DL, LB, DB, K, P, LS (FB folds into RB)."""
    p = (position or "").upper()
    return {"FB": "RB", "HB": "RB", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "DE": "DL", "DT": "DL", "NT": "DL",
            "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "ILB": "LB", "OLB": "LB", "MLB": "LB"}.get(p, p or "other")


def snap_side_pct(group: str):
    """Which snap percentage is the player's workload: offense for offensive positions, special teams for K/P/LS, else defense."""
    return "offense_pct" if group in OFFENSE_GROUPS else ("st_pct" if group in SPECIAL_GROUPS else "defense_pct")


# ------------------------------------------------------------------ data
def _trailing_normal(pw: pl.DataFrame) -> pl.DataFrame:
    """Adds normal_snap_pct (his usual snap percentage BEFORE the game) and ratio (snap_pct / normal for games he played).

    Normal = mean snap_pct over his previous NORMAL_WINDOW games played with no injury designation; if fewer than
    MIN_NORMAL_GAMES such games, over his previous NORMAL_WINDOW games played; otherwise null. Only earlier games count.
    """
    pw = pw.sort("gsis_id", "gameday")
    normal = np.full(pw.height, np.nan)
    gid, healthy_flag = pw["gsis_id"].to_numpy(), (pw["report_status"] == NONE).to_numpy()
    played, pct = pw["played"].to_numpy(), pw["snap_pct"].fill_null(0.0).to_numpy()
    bounds = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        hist_all: list = []
        hist_ok: list = []
        for i in range(a, b):
            ok = hist_ok[-NORMAL_WINDOW:]
            al = hist_all[-NORMAL_WINDOW:]
            if len(ok) >= MIN_NORMAL_GAMES:
                normal[i] = float(np.mean(ok))
            elif len(al) >= MIN_NORMAL_GAMES:
                normal[i] = float(np.mean(al))
            if played[i]:
                hist_all.append(pct[i])
                if healthy_flag[i]:
                    hist_ok.append(pct[i])
    pw = pw.with_columns(normal_snap_pct=pl.Series(normal).fill_nan(None))
    return pw.with_columns(ratio=pl.when(pl.col("played") & (pl.col("normal_snap_pct") > 0))
                           .then((pl.col("snap_pct") / pl.col("normal_snap_pct")).clip(0.0, RATIO_CAP)))


def build_player_weeks(raw_db=config.RAW_DUCKDB_PATH, max_season=None, as_of_fn=main_run_as_of) -> pl.DataFrame:
    """One row per rostered (ACT / INA) player-game, regular season, FEATURE_HISTORY_START..max_season.

    Columns: season, week, team, gsis_id, position, group, gameday, report_status, practice_status (as posted by
    as_of_fn(gameday); 'none' when nothing had been posted), played (any snap), snap_pct (his side's percentage; 0 if no
    snap row), normal_snap_pct, ratio.
    """
    from ingest import ids

    config.cap_season(max_season)
    cap = bl._season_cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        ro = con.execute(
            "SELECT season, week, team, gsis_id, position FROM (SELECT DISTINCT ON (season, week, team, gsis_id) * FROM rosters_weekly "
            f"ORDER BY season, week, team, gsis_id, pulled_at DESC) WHERE game_type = 'REG' AND gsis_id IS NOT NULL "
            f"AND status IN ('ACT', 'INA'){cap}").pl()
        inj = con.execute(
            "SELECT CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week, team, gsis_id, report_status, practice_status, date_modified "
            "FROM (SELECT DISTINCT ON (season, week, team, gsis_id) * FROM injuries ORDER BY season, week, team, gsis_id, pulled_at DESC) "
            f"WHERE game_type = 'REG' AND gsis_id IS NOT NULL{cap}").pl()
        sn = con.execute(
            "SELECT game_id, pfr_player_id, team, offense_snaps, defense_snaps, st_snaps, offense_pct, defense_pct, st_pct FROM "
            f"{bl._latest(con, 'snap_counts', 'game_id, pfr_player_id')} WHERE game_type = 'REG'{cap}").pl()
    finally:
        con.close()
    games = bl.build_team_game_log(raw_db, max_season).select("season", "week", "team", "game_id", "gameday")
    sn = ids.add_canonical_gsis_id(sn).drop_nulls("gsis_id")
    sn = sn.with_columns(any_snap=(pl.col("offense_snaps").fill_null(0) + pl.col("defense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)) > 0)
    pw = ro.join(games, on=["season", "week", "team"], how="inner")
    pw = pw.join(sn.select("game_id", "team", "gsis_id", "offense_pct", "defense_pct", "st_pct", "any_snap"),
                 on=["game_id", "team", "gsis_id"], how="left")
    pw = pw.with_columns(group=pl.col("position").map_elements(pos_group, return_dtype=pl.String), played=pl.col("any_snap").fill_null(False))
    pw = pw.with_columns(snap_pct=pl.when(pl.col("group").is_in(list(OFFENSE_GROUPS))).then(pl.col("offense_pct"))
                         .when(pl.col("group").is_in(list(SPECIAL_GROUPS))).then(pl.col("st_pct")).otherwise(pl.col("defense_pct")))
    inj = inj.with_columns(date_modified=pl.col("date_modified").dt.convert_time_zone("UTC"))
    pw = pw.join(inj, on=["season", "week", "team", "gsis_id"], how="left")
    asof = pl.Series([None if g is None else as_of_fn(g) for g in pw["gameday"].to_list()], dtype=pl.Datetime("us", "UTC"))
    posted = (pl.col("date_modified") <= asof)
    pw = pw.with_columns(report_status=pl.when(posted).then(pl.col("report_status")).otherwise(None),
                         practice_status=pl.when(posted).then(pl.col("practice_status")).otherwise(None))
    pw = pw.with_columns(report_status=pl.col("report_status").map_elements(clean_report, return_dtype=pl.String, skip_nulls=False),
                         practice_status=pl.col("practice_status").map_elements(clean_practice, return_dtype=pl.String, skip_nulls=False))
    pw = _trailing_normal(pw)
    return pw.select("season", "week", "team", "gsis_id", "position", "group", "gameday", "report_status", "practice_status",
                     "played", "snap_pct", "normal_snap_pct", "ratio").sort("season", "week", "team", "gsis_id")


def load_injury_rows(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """The injuries table as the predictor sees it: (season, week, gsis_id, report_status, practice_status, date_modified)."""
    cap = bl._season_cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week, team, gsis_id, report_status, practice_status, date_modified "
            "FROM (SELECT DISTINCT ON (season, week, team, gsis_id) * FROM injuries ORDER BY season, week, team, gsis_id, pulled_at DESC) "
            f"WHERE game_type = 'REG' AND gsis_id IS NOT NULL{cap}").pl()
    finally:
        con.close()
    return d.with_columns(pl.col("date_modified").dt.convert_time_zone("UTC"))


def load_roster_status(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """(season, week, gsis_id, roster_status) from the weekly rosters."""
    cap = bl._season_cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute(
            "SELECT season, week, gsis_id, status AS roster_status FROM (SELECT DISTINCT ON (season, week, team, gsis_id) * FROM rosters_weekly "
            f"ORDER BY season, week, team, gsis_id, pulled_at DESC) WHERE game_type = 'REG' AND gsis_id IS NOT NULL{cap}").pl()
    finally:
        con.close()


# ------------------------------------------------------------------ the fit
@dataclass
class StatusModel:
    """Shrunk P(play) and snap_share_given_play per (position group, report status, practice status)."""
    cutoff: date
    k: float
    n_games: int
    play: dict = field(default_factory=dict)     # key -> (n, raw rate)
    ratio: dict = field(default_factory=dict)    # key -> (n, raw mean)

    @staticmethod
    def _chain(group, report, practice):
        return [(group, report, practice), (group, report), (report,)]

    def _shrunk(self, table: dict, chain, root: float) -> tuple[float, int, str]:
        est, level = root, "root"
        for key in reversed(chain):
            n, raw = table.get(key, (0, root))
            est = (n * raw + self.k * est) / (n + self.k) if n else est
            level = "x".join(key) if n else level
        n_own = table.get(chain[0], (0, 0))[0]
        return est, n_own, level

    def rates(self, group: str, report: str = NONE, practice: str = NONE) -> dict:
        chain = self._chain(group, report, practice)
        p, n_p, _ = self._shrunk(self.play, chain, self.play.get((), (0, 0.9))[1])
        r, n_r, _ = self._shrunk(self.ratio, chain, self.ratio.get((), (0, 1.0))[1])
        return dict(p_play=p, snap_share_given_play=r, n_play=n_p, n_ratio=n_r,
                    raw_p_play=self.play.get(chain[0], (0, None))[1], raw_ratio=self.ratio.get(chain[0], (0, None))[1])


def fit_status_model(pw: pl.DataFrame, cutoff: date, k: float = SHRINK_K) -> StatusModel:
    """Fit on player-games dated strictly before `cutoff` (the week's first kickoff). Later rows are never read.

    Only players with a regular workload enter the fit: a normal snap percentage of at least MIN_NORMAL_PCT before the game
    (computed from earlier games only). snap_share_given_play is defined against a normal, so a player without one (a practice-
    squad call-up, a third-string QB) has nothing to be a fraction of, and counting the roster's never-used depth would pull
    every P(play) toward the bench's ~50%.
    """
    h = pw.filter((pl.col("gameday") < cutoff) & (pl.col("normal_snap_pct") >= MIN_NORMAL_PCT))
    model = StatusModel(cutoff=cutoff, k=k, n_games=h.height)
    levels = {3: ["group", "report_status", "practice_status"], 2: ["group", "report_status"], 1: ["report_status"], 0: []}
    for depth, cols in levels.items():
        agg = (h.group_by(cols).agg(n=pl.len(), p=pl.col("played").mean(), n_r=pl.col("ratio").drop_nulls().len(), r=pl.col("ratio").mean())
               if cols else h.select(n=pl.len(), p=pl.col("played").mean(), n_r=pl.col("ratio").drop_nulls().len(), r=pl.col("ratio").mean()))
        for row in agg.iter_rows(named=True):
            key = tuple(row[c] for c in cols)
            model.play[key] = (row["n"], row["p"])
            if row["n_r"]:
                model.ratio[key] = (row["n_r"], row["r"])
    return model


# ------------------------------------------------------------------ the prediction
def status_as_of(injuries: pl.DataFrame, gsis_id: str, season: int, week: int, as_of: datetime) -> tuple[str, str]:
    """(report status, practice status) of the latest injury row posted at or before `as_of`; ('none', 'none') if none yet."""
    r = injuries.filter((pl.col("gsis_id") == gsis_id) & (pl.col("season") == season) & (pl.col("week") == week)
                        & (pl.col("date_modified") <= _utc(as_of))).sort("date_modified")
    if not r.height:
        return NONE, NONE
    last = r.row(r.height - 1, named=True)
    return clean_report(last["report_status"]), clean_practice(last["practice_status"])


def expected_snap_share(model: StatusModel, injuries: pl.DataFrame, rosters: pl.DataFrame, gsis_id: str, season: int, week: int,
                        group: str, as_of: datetime, normal_snap_pct: float | None = None) -> dict:
    """expected_snap_share = P(play) * snap_share_given_play (a fraction of his normal workload), plus the two parts.

    A player whose roster status that week is IR / PUP / suspended / NFI (BLOCKED_ROSTER_STATUSES) gets P(play) = 0 without going
    through the model. `normal_snap_pct`, when given (his trailing normal percentage), also yields expected_snap_pct.
    """
    rs = rosters.filter((pl.col("gsis_id") == gsis_id) & (pl.col("season") == season) & (pl.col("week") == week))["roster_status"].to_list()
    blocked = any(s in BLOCKED_ROSTER_STATUSES for s in rs)
    report, practice = status_as_of(injuries, gsis_id, season, week, as_of)
    if blocked:
        out = dict(p_play=0.0, snap_share_given_play=0.0, n_play=0, n_ratio=0, raw_p_play=None, raw_ratio=None)
    else:
        out = model.rates(group, report, practice)
    out.update(expected_snap_share=out["p_play"] * out["snap_share_given_play"], report_status=report, practice_status=practice, blocked=blocked)
    if normal_snap_pct is not None:
        out["expected_snap_pct"] = out["expected_snap_share"] * normal_snap_pct
    return out

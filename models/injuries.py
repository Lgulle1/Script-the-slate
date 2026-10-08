"""Injury layer (Phase 4a): status-to-probability model (4a.1), role redistribution (4a.2), in-game exits (4a.3).

EARLY-EXIT GAME (4a.3) -- the one rule, used everywhere in this file:
    a player-game is an early-exit game when the player (1) took snaps but fewer than half of his usual snaps (his snap
    percentage that game < EXIT_FRACTION = 0.5 x his trailing normal snap percentage), AND (2) is on the NEXT week's injury report
    (his team's next game week, same season) with an injury listed as the primary injury (not "illness", "not injury related",
    "resting player", "personal matter"). The injuries table does not say which game an injury came from, so "tied to that game"
    is read as "listed with a real injury on the very next report". Games that fail the rule are ordinary games, however
    few snaps they were. Early-exit games are excluded from every workload baseline (the trailing normal snap percentage, the
    snap_share_given_play ratios, the 4a.2 trailing share baselines) so the same tail event is never in both the baseline and
    the exit draw.

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
EXIT_FRACTION = 0.5               # early exit: played less than this fraction of his usual snaps (rule in the header)
EXIT_K = 15                       # shrinkage of a player's own exit rate toward his position's
NOT_AN_INJURY = ("illness", "not injury", "rest", "personal")
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
    """Adds normal_snap_pct (his usual snap percentage BEFORE the game), early_exit (the 4a.3 rule), work (snap_pct / normal for
    games he played) and ratio (work, for games that are NOT early exits).

    Normal = mean snap_pct over his previous NORMAL_WINDOW games played, not early exits, with no injury designation; if fewer
    than MIN_NORMAL_GAMES such games, over his previous NORMAL_WINDOW non-exit games played; otherwise null. Only earlier games
    count, and an early-exit game never enters the history.
    """
    pw = pw.sort("gsis_id", "gameday")
    normal = np.full(pw.height, np.nan)
    exit_ = np.zeros(pw.height, dtype=bool)
    gid, healthy_flag = pw["gsis_id"].to_numpy(), (pw["report_status"] == NONE).to_numpy()
    played, pct = pw["played"].to_numpy(), pw["snap_pct"].fill_null(0.0).to_numpy()
    cand = pw["next_week_injury"].to_numpy()
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
            if not played[i]:
                continue
            if cand[i] and normal[i] > 0 and pct[i] < EXIT_FRACTION * normal[i]:
                exit_[i] = True                       # early-exit game: kept out of every baseline
                continue
            hist_all.append(pct[i])
            if healthy_flag[i]:
                hist_ok.append(pct[i])
    pw = pw.with_columns(normal_snap_pct=pl.Series(normal).fill_nan(None), early_exit=pl.Series(exit_))
    pw = pw.with_columns(work=pl.when(pl.col("played") & (pl.col("normal_snap_pct") > 0))
                         .then((pl.col("snap_pct") / pl.col("normal_snap_pct")).clip(0.0, RATIO_CAP)))
    return pw.with_columns(ratio=pl.when(~pl.col("early_exit")).then(pl.col("work")))


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
            "SELECT CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week, team, gsis_id, report_status, practice_status, date_modified, "
            "report_primary_injury, practice_primary_injury "
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
    # 4a.3: is he on the NEXT report (his team's next game week) with a real primary injury?
    nxt_week = games.select("season", "team", "week").unique().sort("season", "team", "week").with_columns(
        next_week=pl.col("week").shift(-1).over("season", "team"))
    text = pl.concat_str([pl.col("report_primary_injury").fill_null(""), pl.lit(" "), pl.col("practice_primary_injury").fill_null("")]).str.to_lowercase()
    real = (text.str.strip_chars() != "") & ~pl.any_horizontal([text.str.contains(w) for w in NOT_AN_INJURY])
    nx = (inj.select("season", next_week=pl.col("week"), gsis_id=pl.col("gsis_id"), next_week_injury=real)
             .filter(pl.col("next_week_injury")).unique(["season", "next_week", "gsis_id"]))
    pw = (pw.join(nxt_week, on=["season", "team", "week"], how="left").join(nx, on=["season", "next_week", "gsis_id"], how="left")
            .with_columns(next_week_injury=pl.col("next_week_injury").fill_null(False)).drop("next_week"))
    pw = pw.join(inj.drop("report_primary_injury", "practice_primary_injury"), on=["season", "week", "team", "gsis_id"], how="left")
    asof = pl.Series([None if g is None else as_of_fn(g) for g in pw["gameday"].to_list()], dtype=pl.Datetime("us", "UTC"))
    posted = (pl.col("date_modified") <= asof)
    pw = pw.with_columns(report_status=pl.when(posted).then(pl.col("report_status")).otherwise(None),
                         practice_status=pl.when(posted).then(pl.col("practice_status")).otherwise(None))
    pw = pw.with_columns(report_status=pl.col("report_status").map_elements(clean_report, return_dtype=pl.String, skip_nulls=False),
                         practice_status=pl.col("practice_status").map_elements(clean_practice, return_dtype=pl.String, skip_nulls=False))
    pw = _trailing_normal(pw)
    return pw.select("season", "week", "team", "gsis_id", "position", "group", "gameday", "report_status", "practice_status",
                     "played", "snap_pct", "normal_snap_pct", "ratio", "work", "early_exit", "next_week_injury").sort("season", "week", "team", "gsis_id")


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
            if not row["n"]:
                continue                       # nothing before the cutoff: the defaults in rates() apply
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


# ====================================================================== 4a.2 role redistribution
"""When a teammate is likely out, move his workload to the right players.

Roles (ROLES6): QB, RB1, RB2FB (RB2 and fullback), WR1, WR2, WR3, TE1, TE2, OL (the group of usual starters), other. `slot` is
not a role here: no free source says who played the slot (see features/phase4_inputs.py). Roles come from the pre-game depth chart
(player_game_roles.expected_role); OL starters are linemen whose trailing normal snap percentage is at least OL_STARTER_PCT.

Measured shifts. For every past game before the cutoff in which EXACTLY ONE role r was missing (nobody in that role took an
offensive snap; for OL, at least one starter missing) and the team's other roles were all present, each remaining role j's
share of carries / targets / dropbacks / snaps is compared with that same player's own trailing baseline (the mean of his
previous NORMAL_WINDOW games played). The team's average shift per (r, j, stat) is shrunk toward the league-wide shift with
n / (n + SHIFT_K), n = the team's games of that kind (k = 4), so one past game never drives it. Players in `other` are pooled
for measuring and spread over the other players in proportion to their baselines.

Applying. A player's absence probability q = 1 - P(play | status) / P(play | healthy) from the status model (0 for a player with
no designation; 1 for IR / PUP / suspended), and his workload factor s = snap_share_given_play / healthy value (<= 1). Expected raw
share of a remaining player j = baseline_j + sum_i q_i * shift(role_i, role_j); of the player i himself = baseline_i * (1 - q_i) * s_i.
Shares are clipped at 0 and rescaled so each team's carry, target and dropback shares sum to 1 (snap shares are clipped to
[0, 1], they do not sum to 1). With nobody out (q = 0, s = 1) the output IS the normalised baseline.
Replacement quality: each player carries his OWN trailing efficiency (yards per carry, yards per target, catch rate; league mean
for his role only when he has no history), never the starter's.
"""
ROLES6 = ("QB", "RB1", "RB2FB", "WR1", "WR2", "WR3", "TE1", "TE2", "OL", "other")
ROLE_OF_EXPECTED = {"QB1": "QB", "RB1": "RB1", "RB2": "RB2FB", "FB": "RB2FB", "WR1": "WR1", "WR2": "WR2", "WR3": "WR3", "TE1": "TE1",
                    "TE2": "TE2"}
SHARE_STATS = ("carry", "target", "dropback", "snap")
NORMALISED = ("carry", "target", "dropback")
SHIFT_K = 4
ELIGIBLE_GROUPS = {"carry": {"RB", "QB"}, "target": {"WR", "TE", "RB"}, "dropback": {"QB"}}   # who can absorb freed share of each kind
OL_STARTER_PCT = 0.70


def _trailing_mean_shares(frame: pl.DataFrame) -> pl.DataFrame:
    """Adds base_<stat>: the player's mean share over his previous NORMAL_WINDOW games PLAYED (null with no earlier game)."""
    frame = frame.sort("player_id", "gameday")
    cols = {s: frame[f"{s}_share"].fill_null(0.0).to_numpy() for s in SHARE_STATS}
    played, pid = frame["played"].to_numpy(), frame["player_id"].to_numpy()
    exit_ = frame["early_exit"].to_numpy() if "early_exit" in frame.columns else np.zeros(frame.height, dtype=bool)
    out = {s: np.full(frame.height, np.nan) for s in SHARE_STATS}
    bounds = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        hist = {s: [] for s in SHARE_STATS}
        for i in range(a, b):
            if hist["carry"]:
                for s in SHARE_STATS:
                    out[s][i] = float(np.mean(hist[s][-NORMAL_WINDOW:]))
            if played[i] and not exit_[i]:          # an early-exit game is never part of a baseline (4a.3)
                for s in SHARE_STATS:
                    hist[s].append(cols[s][i])
    return frame.with_columns([pl.Series(f"base_{s}", out[s]).fill_nan(None) for s in SHARE_STATS])


def build_role_frame(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """One row per (team-game, player) in a role: everyone who took an offensive snap, plus the depth-chart player of every
    role who did not (played = False), plus the usual OL starters. Columns: season, week, game_id, team, gameday, player_id, role,
    played, carry_share, target_share, dropback_share, snap_share, carries, rush_yds, targets, rec_yds, receptions, base_*."""
    from features import phase4_inputs as p4

    cutoffs = p4.week_cutoffs(raw_db, max_season)
    roles = p4.build_player_game_roles(raw_db, max_season)
    chart = p4._expected_roles(raw_db, max_season, cutoffs)
    log = bl.build_player_game_log(raw_db, max_season).select("game_id", "player_id", "rush_yds_ex_kneel", "rushing_yards", "carries",
                                                               "targets", "receiving_yards", "receptions")
    games = bl.build_team_game_log(raw_db, max_season).select("season", "week", "team", "game_id", "gameday")
    played = roles.select("season", "week", "game_id", "team", "player_id", role=pl.col("expected_role").replace_strict(ROLE_OF_EXPECTED, default="other"),
                          played=pl.lit(True), carry_share=pl.col("carry_share").fill_null(0.0), target_share=pl.col("target_share").fill_null(0.0),
                          dropback_share=pl.col("dropback_share").fill_null(0.0), snap_share=pl.col("snap_share"))
    played = played.join(log, on=["game_id", "player_id"], how="left")
    missing = (chart.filter(pl.col("expected_role").is_in(list(ROLE_OF_EXPECTED))).join(games, on=["season", "week", "team"], how="inner")
               .join(played.select("game_id", "player_id"), on=["game_id", "player_id"], how="anti")
               .select("season", "week", "game_id", "team", "player_id", role=pl.col("expected_role").replace_strict(ROLE_OF_EXPECTED),
                       played=pl.lit(False), carry_share=pl.lit(0.0), target_share=pl.lit(0.0), dropback_share=pl.lit(0.0), snap_share=pl.lit(0.0)))
    pw = build_player_weeks(raw_db, max_season)
    ol = (pw.filter((pl.col("group") == "OL") & (pl.col("normal_snap_pct") >= OL_STARTER_PCT))
          .join(games, on=["season", "week", "team"], how="inner")
          .select("season", "week", "game_id", "team", player_id="gsis_id", role=pl.lit("OL"), played="played", carry_share=pl.lit(0.0),
                  target_share=pl.lit(0.0), dropback_share=pl.lit(0.0), snap_share=pl.col("snap_pct").fill_null(0.0)))
    frame = pl.concat([played, missing, ol], how="diagonal_relaxed").join(games.select("game_id", "team", "gameday"), on=["game_id", "team"], how="left")
    exits = pw.filter(pl.col("early_exit")).join(games, on=["season", "week", "team"], how="inner").select("game_id", "team", player_id="gsis_id",
                                                                                                         early_exit=pl.lit(True))
    frame = frame.join(exits, on=["game_id", "team", "player_id"], how="left").with_columns(early_exit=pl.col("early_exit").fill_null(False))
    for c in ("rush_yds_ex_kneel", "rushing_yards", "carries", "targets", "receiving_yards", "receptions"):
        frame = frame.with_columns(pl.col(c).fill_null(0.0).cast(pl.Float64))
    frame = frame.rename({"rush_yds_ex_kneel": "rush_yds_ex", "rushing_yards": "rush_yds", "receiving_yards": "rec_yds"})
    return _trailing_mean_shares(frame).sort("season", "week", "game_id", "team", "role", "player_id")


# ---------------------------------------------------------------- measuring the shifts
def _absent_role(g: pl.DataFrame):
    """The single role missing from a team-game, or None (zero or several missing)."""
    roles, played = g["role"].to_list(), g["played"].to_list()
    miss = set()
    for r in set(roles) - {"other"}:
        flags = [p for rr, p in zip(roles, played) if rr == r]
        if (r == "OL" and not all(flags)) or (r != "OL" and not any(flags)):   # OL: any starter missing; others: nobody in the role played
            miss.add(r)
    return next(iter(miss)) if len(miss) == 1 else None


def measure_shifts(frame: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Per (team, absent role r, remaining role j, stat): n games, mean delta (share - own baseline). Only games dated before
    `cutoff` are read. Players in `other` are pooled into one entity (shares and baselines summed)."""
    h = frame.filter(pl.col("gameday") < cutoff)
    rows = []
    for (game_id, team), g in h.group_by(["game_id", "team"], maintain_order=True):
        r = _absent_role(g)
        if r is None:
            continue
        here = g.filter(pl.col("played") & (pl.col("role") != r))
        if "early_exit" in g.columns:
            here = here.filter(~pl.col("early_exit"))      # a player who left hurt is the exit draw's business, not a redistribution
        for role_j, gj in here.group_by("role", maintain_order=True):
            role_j = role_j[0]
            for s in SHARE_STATS:
                base = gj[f"base_{s}"]
                if base.null_count():
                    continue
                if role_j == "other":
                    delta = float(gj[f"{s}_share"].sum() - base.sum())
                else:
                    delta = float((gj[f"{s}_share"] - base).sum())    # one player per role by construction
                rows.append((team, r, role_j, s, game_id, delta))
    return pl.DataFrame(rows, schema={"team": pl.String, "absent_role": pl.String, "role": pl.String, "stat": pl.String, "game_id": pl.String,
                                      "delta": pl.Float64}, orient="row")


class Shifts:
    """Team shifts shrunk toward the league shifts with n / (n + k)."""

    def __init__(self, measured: pl.DataFrame, k: float = SHIFT_K):
        self.k = k
        self.league = {key[:3]: v for key, v in
                       ((tuple(r[c] for c in ("absent_role", "role", "stat")), r["m"]) for r in
                        measured.group_by("absent_role", "role", "stat").agg(m=pl.col("delta").mean()).iter_rows(named=True))}
        self.team = {(r["team"], r["absent_role"], r["role"], r["stat"]): (r["n"], r["m"]) for r in
                     measured.group_by("team", "absent_role", "role", "stat").agg(n=pl.len(), m=pl.col("delta").mean()).iter_rows(named=True)}

    def get(self, team: str, absent_role: str, role: str, stat: str) -> float:
        lg = self.league.get((absent_role, role, stat), 0.0)
        n, m = self.team.get((team, absent_role, role, stat), (0, lg))
        return (n * m + self.k * lg) / (n + self.k)


def fit_shifts(frame: pl.DataFrame, cutoff: date, k: float = SHIFT_K) -> Shifts:
    return Shifts(measure_shifts(frame, cutoff), k)


# ---------------------------------------------------------------- trailing efficiency (replacement quality)
def trailing_efficiency(frame: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Per player, from his own previous NORMAL_WINDOW games played before the cutoff: yards per carry, yards per target, catch
    rate (null when he has no carries / targets)."""
    h = frame.filter((pl.col("gameday") < cutoff) & pl.col("played")).sort("player_id", "gameday")
    h = h.group_by("player_id", maintain_order=True).tail(NORMAL_WINDOW)
    return (h.group_by("player_id").agg(c=pl.col("carries").sum(), ry=pl.col("rush_yds_ex").sum(), t=pl.col("targets").sum(),
                                        ty=pl.col("rec_yds").sum(), rec=pl.col("receptions").sum())
            .select("player_id", eff_ypc=pl.when(pl.col("c") > 0).then(pl.col("ry") / pl.col("c")),
                    eff_yds_per_target=pl.when(pl.col("t") > 0).then(pl.col("ty") / pl.col("t")),
                    eff_catch_rate=pl.when(pl.col("t") > 0).then(pl.col("rec") / pl.col("t"))))


# ---------------------------------------------------------------- applying them
def absence_inputs(model: StatusModel, group: str, report: str, practice: str, blocked: bool = False) -> tuple[float, float]:
    """(q, s): probability the player is out beyond his healthy baseline, and his workload factor if he plays (<= 1)."""
    if blocked:
        return 1.0, 1.0
    r, h = model.rates(group, report, practice), model.rates(group, NONE, NONE)
    q = min(1.0, max(0.0, 1.0 - r["p_play"] / h["p_play"])) if h["p_play"] > 0 else 0.0
    s = min(1.0, r["snap_share_given_play"] / h["snap_share_given_play"]) if h["snap_share_given_play"] > 0 else 1.0
    return q, s


def redistribute(shifts: Shifts, team: str, players: pl.DataFrame, q: dict, s: dict | None = None) -> pl.DataFrame:
    """Expected shares for one team-game.

    players: player_id, role, base_carry, base_target, base_dropback, base_snap (the players' trailing baselines; a player with
    a null baseline counts as 0). q[player_id] = probability he is out (default 0); s[player_id] = workload factor if he plays
    (default 1). Returns player_id, role, base_<stat> (normalised), exp_<stat>, plus a 'rest' row (unlisted players) for the
    three normalised stats.
    """
    s = s or {}
    grp_arr = players["group"].to_list() if "group" in players.columns else None
    p = players.select("player_id", "role", *[pl.col(f"base_{x}").fill_null(0.0) for x in SHARE_STATS])
    ids_, roles_ = p["player_id"].to_list(), p["role"].to_list()
    qv = np.array([float(q.get(i, 0.0)) for i in ids_])
    sv = np.array([float(s.get(i, 1.0)) for i in ids_])
    out = {"player_id": ids_ + ["rest"], "role": roles_ + ["rest"]}
    for st in SHARE_STATS:
        b = p[f"base_{st}"].to_numpy().astype(float)
        if st in NORMALISED:
            rest = max(0.0, 1.0 - b.sum())
            b, rest = (b / b.sum(), 0.0) if b.sum() > 1.0 else (b, rest)
        else:
            rest = 0.0
        raw = b * (1.0 - qv) * sv
        add = np.zeros(len(b))
        is_other = np.array([r == "other" for r in roles_])
        snap_b = p["base_snap"].to_numpy().astype(float)
        members = {r: [k for k, rr in enumerate(roles_) if rr == r] for r in set(roles_)}

        def weight(k):       # a player's part of his role: by baseline snaps (equal split when none)
            m = members[roles_[k]]
            tot = snap_b[m].sum()
            return snap_b[k] / tot if tot > 0 else 1.0 / len(m)

        for i in np.flatnonzero(qv > 0):
            wi = weight(i)
            for j, rj in enumerate(roles_):
                if j == i or rj == "other" or rj == roles_[i]:
                    continue
                add[j] += qv[i] * wi * shifts.get(team, roles_[i], rj, st) * weight(j) * (1.0 - qv[j])
            pool = shifts.get(team, roles_[i], "other", st) * qv[i] * wi
            w = np.where(is_other, b, 0.0)
            if w.sum() > 0:
                add += pool * w / w.sum()
        raw = np.clip(raw + add, 0.0, None)
        if st in NORMALISED:
            full = np.append(raw, rest)
            tot = full.sum()
            if tot < 1.0 - 1e-12:      # freed share the measured shifts did not hand on: to players who can take it, else the unlisted
                if grp_arr is not None:     # position-aware: carries -> RB/QB, targets -> WR/TE/RB, dropbacks -> QB; in proportion to expected share
                    mask = np.array([g in ELIGIBLE_GROUPS[st] for g in grp_arr]) & (qv < 0.999)
                    w = np.where(mask, raw, 0.0)
                    w = w if w.sum() > 0 else mask.astype(float)
                else:
                    w = np.where(is_other, np.maximum(b, 0.0), 0.0)
                    w = w if w.sum() > 0 else is_other.astype(float)
                if w.sum() > 0:
                    full[:-1] += (1.0 - tot) * w / w.sum()
                else:
                    full[-1] += 1.0 - tot
            else:
                full = full / tot
            out[f"base_{st}"] = list(np.append(b, rest))
            out[f"exp_{st}"] = list(full)
        else:
            out[f"base_{st}"] = list(np.append(b, 0.0))
            out[f"exp_{st}"] = list(np.append(np.clip(raw, 0.0, 1.0), 0.0))
    return pl.DataFrame(out)


# ---------------------------------------------------------------- one team-game, end to end
def _team_players(frame: pl.DataFrame, game_id: str, team: str, cutoff: date) -> pl.DataFrame:
    """The team's players for a pre-game prediction: the depth-chart role holders of this game (baselines are trailing means that
    exclude it), plus the `other` players seen in the team's previous 4 games (baselines = mean share over their last NORMAL_WINDOW
    games played before the cutoff). Who actually played in this game is NOT used."""
    g = frame.filter((pl.col("game_id") == game_id) & (pl.col("team") == team) & (pl.col("role") != "other"))
    prev = frame.filter((pl.col("team") == team) & (pl.col("gameday") < cutoff))
    recent = sorted(prev["gameday"].unique().to_list())[-4:]
    others = (prev.filter((pl.col("role") == "other") & pl.col("gameday").is_in(recent)).select("player_id").unique()
              .join(g.select("player_id"), on="player_id", how="anti"))      # a role holder this game is not also an `other`
    hist = (frame.filter((pl.col("gameday") < cutoff) & pl.col("played")).join(others, on="player_id", how="inner")
            .sort("player_id", "gameday").group_by("player_id", maintain_order=True).tail(NORMAL_WINDOW))
    o = hist.group_by("player_id").agg(**{f"base_{s}": pl.col(f"{s}_share").mean() for s in SHARE_STATS}).with_columns(role=pl.lit("other"))
    keep = ["player_id", "role", *[f"base_{s}" for s in SHARE_STATS]]
    return pl.concat([g.select(keep), o.select(keep)], how="vertical_relaxed")


def make_status_lookup(injuries: pl.DataFrame):
    """Fast `status_fn(gsis_id, season, week, as_of) -> (report, practice)` over the injuries rows (same rule as status_as_of)."""
    rows: dict = {}
    for r in injuries.sort("date_modified").iter_rows(named=True):
        rows.setdefault((r["gsis_id"], r["season"], r["week"]), []).append((r["date_modified"], r["report_status"], r["practice_status"]))

    def status_fn(gsis_id, season, week, as_of):
        last = None
        for dm, rep, pra in rows.get((gsis_id, season, week), ()):
            if dm <= _utc(as_of):
                last = (clean_report(rep), clean_practice(pra))
        return last or (NONE, NONE)
    return status_fn


def game_expected_shares(frame: pl.DataFrame, shifts: Shifts, model: StatusModel, injuries: pl.DataFrame | None, rosters: pl.DataFrame | None,
                         game_id: str, team: str, season: int, week: int, gameday: date, as_of: datetime, cutoff: date,
                         groups: dict | None = None, *, exit_model: "ExitModel | None" = None, status_fn=None, blocked_ids: set | None = None,
                         player_groups: dict | None = None, with_eff: bool = True) -> pl.DataFrame:
    """Expected carry / target / dropback / snap shares for one team-game before it is played: statuses as of `as_of`
    (status model -> q, s per player; IR / PUP / suspended -> out), shifts from `shifts` (fit on games before `cutoff`), each
    player's trailing efficiency from games before the cutoff. `groups` maps role -> status-model position group;
    `player_groups` (gsis_id -> group) overrides it per player. With `exit_model`, also p_exit. exp_snap_share is the 4a.1 output
    P(play) * snap_share_given_play for the player's status. `status_fn`/`blocked_ids` are the fast paths used by the feature build."""
    groups = groups or {"QB": "QB", "RB1": "RB", "RB2FB": "RB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "TE1": "TE", "TE2": "TE", "OL": "OL",
                        "other": "WR"}
    players = _team_players(frame, game_id, team, cutoff)
    if blocked_ids is None:
        blocked_ids = set(rosters.filter((pl.col("season") == season) & (pl.col("week") == week)
                                         & pl.col("roster_status").is_in(list(BLOCKED_ROSTER_STATUSES)))["gsis_id"].to_list())
    status_fn = status_fn or (lambda pid, se, wk, ao: status_as_of(injuries, pid, se, wk, ao))
    q, s, snap, pex = {}, {}, {}, {}
    for pid, role in zip(players["player_id"].to_list(), players["role"].to_list()):
        grp = (player_groups or {}).get(pid) or groups.get(role, "WR")
        report, practice = status_fn(pid, season, week, as_of)
        blocked = pid in blocked_ids
        q[pid], s[pid] = absence_inputs(model, grp, report, practice, blocked=blocked)
        r = model.rates(grp, report, practice)
        snap[pid] = 0.0 if blocked else r["p_play"] * r["snap_share_given_play"]
        if exit_model is not None:
            pex[pid] = exit_model.p_exit(pid, grp)
    players = players.with_columns(group=pl.struct("player_id", "role").map_elements(
        lambda r: (player_groups or {}).get(r["player_id"]) or groups.get(r["role"], "WR"), return_dtype=pl.String))
    out = redistribute(shifts, team, players, q, s)
    if with_eff:
        out = out.join(trailing_efficiency(frame, cutoff), on="player_id", how="left")
    cols = [pl.col("player_id").map_elements(lambda i: q.get(i, 0.0), return_dtype=pl.Float64).alias("p_out"),
            pl.col("player_id").map_elements(lambda i: snap.get(i), return_dtype=pl.Float64).alias("exp_snap_share")]
    if exit_model is not None:
        cols.append(pl.col("player_id").map_elements(lambda i: pex.get(i), return_dtype=pl.Float64).alias("p_exit"))
    return out.with_columns(*cols, game_id=pl.lit(game_id), team=pl.lit(team))


def role_prior_shares(frame: pl.DataFrame, cutoff: date) -> dict:
    """Mean share of carries / targets / dropbacks / snaps per depth-chart role (`other` = everyone not in a listed role), over every
    earlier game the role holder played (exit games excluded): what a role holder with no history of his own (a rookie, a new starter) is expected to get."""
    h = frame.filter((pl.col("gameday") < cutoff) & pl.col("played"))
    if "early_exit" in h.columns:
        h = h.filter(~pl.col("early_exit"))
    g = h.group_by("role").agg([pl.col(f"{s}_share").mean().alias(s) for s in SHARE_STATS])
    return {r["role"]: {s: r[s] for s in SHARE_STATS} for r in g.iter_rows(named=True)}


ROLE_K = 1.0     # games' worth of weight on the role's league-average share in a role holder's baseline (chosen once, not tuned)


def role_history(frame: pl.DataFrame, cutoff: date) -> dict:
    """(player_id, role) -> (n, mean shares) over the player's last NORMAL_WINDOW games played in that depth-chart role (`other` is a role:
    a benched former starter's baseline is his usage as a backup), any team, before the cutoff, exit games excluded."""
    h = frame.filter((pl.col("gameday") < cutoff) & pl.col("played"))
    if "early_exit" in h.columns:
        h = h.filter(~pl.col("early_exit"))
    h = h.sort("gameday").group_by("player_id", "role", maintain_order=True).tail(NORMAL_WINDOW)
    g = h.group_by("player_id", "role").agg(n=pl.len(), **{s: pl.col(f"{s}_share").mean() for s in SHARE_STATS})
    return {(r["player_id"], r["role"]): (r["n"], {s: r[s] for s in SHARE_STATS}) for r in g.iter_rows(named=True)}


def game_share_scenarios(frame: pl.DataFrame, shifts: Shifts, model: StatusModel, game_id: str, team: str, season: int, week: int,
                         as_of: datetime, cutoff: date, *, status_fn, blocked_ids: set, player_groups: dict, min_q: float = 0.001,
                         role_prior: dict | None = None, role_hist: dict | None = None) -> tuple:
    """The two ends of 4a.2's probability-weighted mixture, for a simulation that draws availability game by game.

    Returns (play, scenarios, q):
      play       player_id, role, group, p_out (q) and the expected shares IF EVERY PLAYER PLAYS (workload factors applied),
                 b_carry / b_target / b_dropback, with a final `rest` row (shares of the unlisted players)
      scenarios  for each player i with q_i > min_q: his shares in the world where ONLY i is out (everything else as in `play`)
      q          player_id -> probability he is out
    A simulated game draws each uncertain player out with probability q_i and uses  play + sum over out players (scenario_i - play).
    The probability-weighted average over those draws equals the mixture of game_expected_shares to first order (the redistribution is
    linear in q before clipping), which is what makes the simulated shares a distribution around the same expectation.

    role_prior / role_hist (role_prior_shares, role_history): every player's baseline (`other` is a role too) is the average of his last six games
    played IN THAT ROLE (on any team), shrunk toward the role's league average with n / (n + ROLE_K), n = those games. His all-games
    trailing baseline (4a.2) mixes jobs: a first-time starting quarterback's earlier backup games, a promoted backup, a rookie (n = 0 ->
    the role's average). (game_expected_shares, the 4a.4 feature builder, keeps its original behaviour: its results are frozen.)
    """
    groups = {"QB": "QB", "RB1": "RB", "RB2FB": "RB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "TE1": "TE", "TE2": "TE", "OL": "OL", "other": "WR"}
    players = _team_players(frame, game_id, team, cutoff)
    if role_prior:
        rh = role_hist if role_hist is not None else role_history(frame, cutoff)

        def role_base(pid, role, st, current):
            if role not in role_prior:
                return current
            n, mean = rh.get((pid, role), (0, None))
            own = mean[st] if (n and mean[st] is not None) else role_prior[role][st]     # no value in those games (e.g. no snap row): the role's
            return (n * own + ROLE_K * role_prior[role][st]) / (n + ROLE_K)
        players = players.with_columns([
            pl.struct("player_id", "role", f"base_{st}").map_elements(
                lambda r, st=st: role_base(r["player_id"], r["role"], st, r[f"base_{st}"]), return_dtype=pl.Float64).alias(f"base_{st}")
            for st in SHARE_STATS])
    q, s = {}, {}
    for pid, role in zip(players["player_id"].to_list(), players["role"].to_list()):
        grp = (player_groups or {}).get(pid) or groups.get(role, "WR")
        report, practice = status_fn(pid, season, week, as_of)
        q[pid], s[pid] = absence_inputs(model, grp, report, practice, blocked=pid in blocked_ids)
    players = players.with_columns(group=pl.struct("player_id", "role").map_elements(
        lambda r: (player_groups or {}).get(r["player_id"]) or groups.get(r["role"], "WR"), return_dtype=pl.String))
    play = redistribute(shifts, team, players, {}, s).join(players.select("player_id", "group"), on="player_id", how="left")
    play = play.with_columns(p_out=pl.col("player_id").map_elements(lambda i: q.get(i, 0.0), return_dtype=pl.Float64))
    scen = []
    for pid, qi in q.items():
        if qi > min_q:
            sc = redistribute(shifts, team, players, {pid: 1.0}, s)
            scen.append(sc.select("player_id", exp_carry="exp_carry", exp_target="exp_target", exp_dropback="exp_dropback").with_columns(out_player=pl.lit(pid)))
    scenarios = pl.concat(scen) if scen else pl.DataFrame(schema={"player_id": pl.String, "exp_carry": pl.Float64, "exp_target": pl.Float64,
                                                                  "exp_dropback": pl.Float64, "out_player": pl.String})
    return play, scenarios, q


# ====================================================================== 4a.3 in-game injury exits
@dataclass
class ExitModel:
    """P(early exit) by position, adjusted by the player's own history, and the share of the game he completes when he exits.

    p_exit(player) = (own exits + k * position rate) / (own games + k),   k = EXIT_K = 15   (n / (n + k) shrinkage toward the position)
    Share completed = his snap percentage that game / his normal, for his position's early-exit games (an empirical distribution).
    Fit on player-games dated before the cutoff, among players with a regular workload (normal >= MIN_NORMAL_PCT).
    """
    cutoff: date
    k: float
    position: dict = field(default_factory=dict)     # group -> (games, exits)
    player: dict = field(default_factory=dict)       # gsis_id -> (games, exits)
    shares: dict = field(default_factory=dict)       # group -> sorted array of share completed (0 <= x < EXIT_FRACTION)
    all_shares: np.ndarray = field(default_factory=lambda: np.array([0.25]))
    mean_ne: dict = field(default_factory=dict)      # group -> mean workload of NON-exit played games (the baseline's side)
    mean_exit: dict = field(default_factory=dict)    # group -> mean share completed

    def position_rate(self, group: str) -> float:
        n, e = self.position.get(group, (0, 0))
        tn = sum(v[0] for v in self.position.values())
        te = sum(v[1] for v in self.position.values())
        return (e + 0.0) / n if n else (te / tn if tn else 0.0)

    def p_exit(self, player_id: str | None, group: str) -> float:
        n, e = self.player.get(player_id, (0, 0))
        return (e + self.k * self.position_rate(group)) / (n + self.k)

    def sampler(self, group: str):
        """A function (rng, size=None) -> share of the game completed given an early exit, drawn from his position's history."""
        pool = self.shares.get(group)
        pool = pool if pool is not None and len(pool) >= 20 else self.all_shares
        return lambda rng, size=None: rng.choice(pool, size=size, replace=True)


def fit_exit_model(pw: pl.DataFrame, cutoff: date, k: float = EXIT_K) -> ExitModel:
    """Fit on played games before `cutoff` (the week's first kickoff); later rows are never read."""
    h = pw.filter((pl.col("gameday") < cutoff) & pl.col("played") & (pl.col("normal_snap_pct") >= MIN_NORMAL_PCT))
    m = ExitModel(cutoff=cutoff, k=k)
    for r in h.group_by("group").agg(n=pl.len(), e=pl.col("early_exit").sum()).iter_rows(named=True):
        m.position[r["group"]] = (r["n"], r["e"])
    for r in h.group_by("gsis_id").agg(n=pl.len(), e=pl.col("early_exit").sum()).iter_rows(named=True):
        m.player[r["gsis_id"]] = (r["n"], r["e"])
    ex = h.filter(pl.col("early_exit"))
    for g, v in ex.group_by("group").agg(v=pl.col("work")).iter_rows():
        m.shares[g] = np.sort(np.clip(np.array(v, dtype=float), 0.0, EXIT_FRACTION))
        m.mean_exit[g] = float(np.mean(m.shares[g]))
    if ex.height:
        m.all_shares = np.sort(np.clip(ex["work"].to_numpy().astype(float), 0.0, EXIT_FRACTION))
    for r in h.filter(~pl.col("early_exit")).group_by("group").agg(w=pl.col("work").mean()).iter_rows(named=True):
        m.mean_ne[r["group"]] = r["w"]
    return m


def early_exit_for(model: ExitModel, player_id: str | None, group: str):
    """(P(early exit), sampler) for a player in a game: the sampler draws the share of the game he completes when he exits, for
    the 4b simulation (`sampler(rng, size=None)`)."""
    return model.p_exit(player_id, group), model.sampler(group)

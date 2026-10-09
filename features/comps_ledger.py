"""4c.1 ledgers: per entity-game numerators and denominators for every feature in models/comps_spec.py.

A window value is a weighted pooled rate  sum(w * n) / sum(w * d)  over an entity's earlier games, so every feature is stored as a pair of
columns  "<unit>.<feature>|n"  and  "<unit>.<feature>|d"  per (game, entity). Nothing here knows about windows, cutoffs or lineups.

Entities
  team_ledger    one row per (game_id, team): the OFFENSE units for the team's own plays and the DEFENSE units for the plays it faced
  player_ledger  one row per (game_id, team, player_id): the archetype features, plus the per-player pieces (carry, dropback, receiving rates)
                 that models/comps.py uses to move a team's unit vector when the lineup changes
Play definitions (checked against pbp): a designed rush is rush == 1 (kneels and spikes carry rush = pass = 0, scrambles are pass = 1 and
qb_dropback = 1); a dropback is qb_dropback == 1 (sacks and scrambles included; a scramble has no passer id, so the QB is the
rusher and the dropback is credited to him); an attempt is pass == 1 with no sack and no scramble; a team
play is a rush or a pass. Two-point tries and deleted plays are out.

Every loader is capped at the last backtest season (config.cap_season): 2025+ is the locked holdout and is never read.
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from ingest import ids
from models import comps_spec as cs

START = cs.BASE_START
GROUPS = ("wr", "te", "rb")                   # receiver position groups the usage / coverage features split on
_POS_GROUP = {"WR": "wr", "TE": "te", "RB": "rb", "FB": "rb", "HB": "rb"}
CLOSED_SHELLS = ("COVER_1", "COVER_3")                          # middle of the field closed (single-high)
OPEN_SHELLS = ("COVER_2", "2_MAN", "COVER_4", "COVER_6")        # middle open (two-high)

FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}     # pbp carries today's code for a moved franchise; the other tables keep the code of the day


def canon(col: str) -> pl.Expr:
    """A team-code column mapped to the franchise's current code, so tables with historical codes join to pbp (game_id keeps the historical code)."""
    return pl.col(col).replace(FRANCHISE)


PBP_COLUMNS = ("season", "week", "game_id", "play_id", "posteam", "defteam", "rush", "pass", "sack", "qb_scramble", "qb_dropback", "epa", "success",
               "yards_gained", "cpoe", "cp", "air_yards", "yards_after_catch", "shotgun", "no_huddle", "pass_oe", "run_location", "run_gap", "qb_hit",
               "complete_pass", "yardline_100", "rusher_player_id", "receiver_player_id", "passer_player_id", "two_point_attempt", "play_deleted")


# ------------------------------------------------------------------------------------------------------------------------------------ loaders
def _window(max_season, prefix: str = "") -> str:
    return f"{prefix}season >= {START} AND {prefix}season <= {config.cap_season(max_season)}"


def load_games(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """One row per team per completed regular-season game, with the team's game number in its season (byes skipped)."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        s = con.execute("SELECT game_id, season, week, gameday, home_team, away_team, home_score, away_score FROM "
                        "(SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) "
                        f"WHERE game_type = 'REG' AND home_score IS NOT NULL AND away_score IS NOT NULL AND {_window(max_season)}").pl()
    finally:
        con.close()
    s = s.with_columns(pl.col("gameday").str.to_date())
    home = s.select("game_id", "season", "week", "gameday", team=canon("home_team"), opponent=canon("away_team"), is_home=pl.lit(True))
    away = s.select("game_id", "season", "week", "gameday", team=canon("away_team"), opponent=canon("home_team"), is_home=pl.lit(False))
    return (pl.concat([home, away]).sort("team", "season", "gameday")
            .with_columns(team_game_num=pl.col("gameday").rank("ordinal").over("team", "season").cast(pl.Int32))
            .sort("season", "week", "game_id", "team"))


def load_plays(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """Regular-season plays with the FTN charting columns and the participation coverage columns joined on (game_id, play_id)."""
    cols = ", ".join(f"p.{c}" for c in PBP_COLUMNS)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        plays = con.execute(
            f"SELECT {cols}, f.is_motion, f.is_play_action, f.is_screen_pass, f.n_blitzers, f.n_defense_box, "
            "q.defense_man_zone_type AS man_zone, q.defense_coverage_type AS coverage_type FROM "
            "(SELECT DISTINCT ON (game_id, play_id) * FROM pbp WHERE season_type = 'REG' ORDER BY game_id, play_id, pulled_at DESC) p "
            "LEFT JOIN (SELECT DISTINCT ON (nflverse_game_id, nflverse_play_id) * FROM ftn_charting ORDER BY nflverse_game_id, nflverse_play_id, pulled_at DESC) f "
            "  ON f.nflverse_game_id = p.game_id AND f.nflverse_play_id = p.play_id "
            "LEFT JOIN (SELECT DISTINCT ON (nflverse_game_id, play_id) * FROM participation ORDER BY nflverse_game_id, play_id, pulled_at DESC) q "
            "  ON q.nflverse_game_id = p.game_id AND q.play_id = p.play_id "
            f"WHERE {_window(max_season, 'p.')}").pl()
    finally:
        con.close()
    plays = plays.filter((pl.col("play_deleted").fill_null(0) == 0) & (pl.col("two_point_attempt").fill_null(0) == 0))
    one = lambda c: (pl.col(c).fill_null(0) == 1)
    return plays.with_columns(
        f_rush=one("rush"), f_db=one("qb_dropback"),
        f_att=one("pass") & ~one("sack") & ~one("qb_scramble"),
        f_play=one("rush") | one("pass")).with_columns(
        f_target=pl.col("f_att") & pl.col("receiver_player_id").is_not_null(),
        ftn=pl.col("is_play_action").is_not_null()).with_columns(
        qb_id=pl.when(pl.col("f_db")).then(pl.coalesce("passer_player_id", "rusher_player_id")))      # scrambles carry no passer id: the QB is the rusher


def load_positions(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """(gsis_id, season, week) -> position / height / weight from the weekly rosters (regular season, latest pull per row)."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        r = con.execute("SELECT gsis_id, season, week, position, height, NULLIF(weight, 0) AS weight FROM "
                        "(SELECT DISTINCT ON (gsis_id, season, week) * FROM rosters_weekly WHERE game_type = 'REG' AND gsis_id IS NOT NULL "
                        f" AND {_window(max_season)} ORDER BY gsis_id, season, week, pulled_at DESC)").pl()
    finally:
        con.close()
    return r.with_columns(key=pl.col("season") * 100 + pl.col("week")).sort("key")


def attach_position(df: pl.DataFrame, id_col: str, positions: pl.DataFrame, extra=("height", "weight")) -> pl.DataFrame:
    """Position (and height / weight) as of the row's (season, week): the latest roster row at or before it, any earlier season included."""
    left = df.with_columns(_key=pl.col("season") * 100 + pl.col("week")).sort("_key")
    right = positions.select(pl.col("gsis_id").alias(id_col), "key", "position", *extra).sort("key")
    out = left.join_asof(right, left_on="_key", right_on="key", by=id_col, strategy="backward").drop("_key", "key")
    return out.with_columns(group=pl.col("position").replace_strict(_POS_GROUP, default=None))


def load_snaps(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """Offense snap share per player-game (snap_counts joined to gsis ids through the master id table)."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        s = con.execute("SELECT game_id, season, week, team, pfr_player_id, position, offense_snaps, offense_pct FROM "
                        "(SELECT DISTINCT ON (game_id, pfr_player_id, team) * FROM snap_counts WHERE game_type = 'REG' "
                        f" AND {_window(max_season)} ORDER BY game_id, pfr_player_id, team, pulled_at DESC) WHERE offense_snaps > 0").pl()
    finally:
        con.close()
    return ids.add_canonical_gsis_id(s).drop_nulls("gsis_id").with_columns(team=canon("team")).select("game_id", "team", "gsis_id", "offense_snaps", "offense_pct")


def load_pfr(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """PFR advanced passing (per QB-game) and rushing (per player-game, gsis ids attached)."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        p = con.execute("SELECT game_id, team, opponent, times_pressured, times_blitzed FROM "
                        "(SELECT DISTINCT ON (game_id, team, pfr_player_id) * FROM pfr_advstats_pass WHERE game_type = 'REG' "
                        f" AND {_window(max_season)} ORDER BY game_id, team, pfr_player_id, pulled_at DESC)").pl()
        r = con.execute("SELECT game_id, team, pfr_player_id, carries, rushing_yards_after_contact FROM "
                        "(SELECT DISTINCT ON (game_id, team, pfr_player_id) * FROM pfr_advstats_rush WHERE game_type = 'REG' "
                        f" AND {_window(max_season)} ORDER BY game_id, team, pfr_player_id, pulled_at DESC)").pl()
    finally:
        con.close()
    return (p.with_columns(team=canon("team"), opponent=canon("opponent")),
            ids.add_canonical_gsis_id(r).drop_nulls("gsis_id").with_columns(team=canon("team")).select("game_id", "team", "gsis_id", "carries", "rushing_yards_after_contact"))


def load_slots(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """Depth-chart slot bucket (1, 2, 3 = third or lower) and position family per (gsis_id, season, week), offense only; the pregame chart of that week."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute("SELECT season, week, gsis_id, position, depth_team FROM "
                        "(SELECT DISTINCT ON (season, week, club_code, gsis_id, depth_team, position) * FROM depth_charts WHERE game_type = 'REG' AND formation = 'Offense' "
                        f" AND gsis_id IS NOT NULL AND {_window(max_season)} ORDER BY season, week, club_code, gsis_id, depth_team, position, pulled_at DESC)").pl()
    finally:
        con.close()
    d = d.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"])).with_columns(slot=pl.col("depth_team").cast(pl.Int32, strict=False).clip(1, 3))
    return d.group_by("gsis_id", "season", "week").agg(pl.col("slot").min().fill_null(3), pl.col("position").first().alias("family")).sort("gsis_id", "season", "week")


# ------------------------------------------------------------------------------------------------------------------------------ expressions
def _c(name):
    return pl.col(name)


def nd(flag: pl.Expr, value: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """(sum of value, count) over the rows where flag holds and value is not null."""
    v = value.cast(pl.Float64)
    ok = flag.fill_null(False) & v.is_not_null()
    return pl.when(ok).then(v).otherwise(0.0).sum(), ok.cast(pl.Float64).sum()


def _rush_book() -> dict:
    f = _c("f_rush")
    loc = lambda s: (f & _c("run_location").is_not_null(), (_c("run_location") == s))
    gap = lambda s: (f & _c("run_gap").is_not_null(), (_c("run_gap") == s))
    return {"rush_epa_per_play": (f, _c("epa")), "rush_success_rate": (f, _c("success")), "yards_per_carry": (f, _c("yards_gained")),
            "explosive_run_rate": (f, _c("yards_gained") >= cs.EXPLOSIVE_RUN_YARDS),
            "run_location_share_left": loc("left"), "run_location_share_middle": loc("middle"), "run_location_share_right": loc("right"),
            "run_gap_share_end": gap("end"), "run_gap_share_tackle": gap("tackle"), "run_gap_share_guard": gap("guard"),
            "shotgun_run_share": (f, _c("shotgun"))}


def _pass_book() -> dict:
    db, att = _c("f_db"), _c("f_att")
    return {"epa_per_dropback": (db, _c("epa")), "cpoe": (att & _c("cpoe").is_not_null(), _c("cpoe")), "yards_per_attempt": (att, _c("yards_gained")),
            "explosive_pass_rate": (att, _c("yards_gained") >= cs.EXPLOSIVE_PASS_YARDS),
            "average_depth_of_target": (att & _c("air_yards").is_not_null(), _c("air_yards")), "sack_rate": (db, _c("sack"))}


def _team_book() -> dict:
    """'unit.feature' -> (side, flag, value) for the features computed from plays by team (side 'off' = posteam, 'def' = defteam)."""
    rush, pas = _rush_book(), _pass_book()
    f_rush, db, att, play = _c("f_rush"), _c("f_db"), _c("f_att"), _c("f_play")
    ftn = _c("ftn")
    b = {}
    for k, (fl, v) in rush.items():
        b[f"run_offense.{k}"] = ("off", fl, v)
    b["run_offense.rush_rate_over_expected"] = ("off", play & _c("pass_oe").is_not_null(), -_c("pass_oe") / 100.0)
    for k, (fl, v) in pas.items():
        b[f"pass_offense.{k}"] = ("off", fl, v)
    b["pass_offense.play_action_rate"] = ("off", db & ftn, _c("is_play_action"))
    b["pass_offense.screen_rate"] = ("off", db & ftn, _c("is_screen_pass"))
    b["pass_offense.motion_rate"] = ("off", play & ftn, _c("is_motion"))
    b["pass_offense.no_huddle_rate"] = ("off", play, _c("no_huddle"))
    b["ol_protection.sack_rate_allowed"] = ("off", db, _c("sack"))
    for s in ("left", "middle", "right"):
        b[f"ol_protection.rush_epa_{s}"] = ("off", f_rush & (_c("run_location") == s), _c("epa"))
        b[f"run_defense.rush_epa_allowed_{s}"] = ("def", f_rush & (_c("run_location") == s), _c("epa"))
    b["run_defense.rush_epa_allowed"] = ("def", f_rush, _c("epa"))
    b["run_defense.rush_success_allowed"] = ("def", f_rush, _c("success"))
    b["run_defense.yards_per_carry_allowed"] = ("def", f_rush, _c("yards_gained"))
    b["run_defense.explosive_runs_allowed"] = ("def", f_rush, _c("yards_gained") >= cs.EXPLOSIVE_RUN_YARDS)
    b["run_defense.box_count_faced"] = ("def", f_rush & _c("n_defense_box").is_not_null(), _c("n_defense_box"))
    b["pass_rush.sack_rate"] = ("def", db, _c("sack"))
    b["pass_rush.blitz_rate_ftn"] = ("def", db & _c("n_blitzers").is_not_null(), _c("n_blitzers") > 0)
    b["pass_rush.qb_hit_rate"] = ("def", db, _c("qb_hit"))
    b["pass_coverage.epa_per_dropback_allowed"] = ("def", db, _c("epa"))
    b["pass_coverage.cpoe_allowed"] = ("def", att & _c("cpoe").is_not_null(), _c("cpoe"))
    b["pass_coverage.yards_per_attempt_allowed"] = ("def", att, _c("yards_gained"))
    b["pass_coverage.explosive_passes_allowed"] = ("def", att, _c("yards_gained") >= cs.EXPLOSIVE_PASS_YARDS)
    classified = _c("man_zone").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"])
    b["coverage_mix.man_rate"] = ("def", att & classified, _c("man_zone") == "MAN_COVERAGE")
    b["coverage_mix.zone_rate"] = ("def", att & classified, _c("man_zone") == "ZONE_COVERAGE")
    shell = _c("coverage_type").is_in(list(CLOSED_SHELLS + OPEN_SHELLS))
    b["coverage_mix.middle_closed_rate"] = ("def", att & shell, _c("coverage_type").is_in(list(CLOSED_SHELLS)))
    return b


def _player_book() -> dict:
    """'unit.feature' -> (entity, flag, value) for the features computed from plays by player (entity = rusher / passer / receiver)."""
    rush, pas = _rush_book(), _pass_book()
    f_rush, db, att, tgt = _c("f_rush"), _c("f_db"), _c("f_att"), _c("f_target")
    b = {}
    for k, (fl, v) in rush.items():
        b[f"run_offense.{k}"] = ("rusher", fl, v)
    for k in ("epa_per_dropback", "cpoe", "yards_per_attempt", "explosive_pass_rate", "average_depth_of_target", "sack_rate"):
        b[f"pass_offense.{k}"] = ("passer", *pas[k])
    # archetypes
    b["rb_archetype.yards_per_carry"] = ("rusher", f_rush, _c("yards_gained"))
    b["rb_archetype.explosive_rate"] = ("rusher", f_rush, _c("yards_gained") >= cs.EXPLOSIVE_RUN_YARDS)
    b["rb_archetype.rush_epa"] = ("rusher", f_rush, _c("epa"))
    b["receiver_archetype.average_depth_of_target"] = ("receiver", tgt & _c("air_yards").is_not_null(), _c("air_yards"))
    b["receiver_archetype.yards_after_catch"] = ("receiver", tgt & (_c("complete_pass") == 1), _c("yards_after_catch"))
    b["receiver_archetype.catch_rate_over_expected"] = ("receiver", tgt & _c("cp").is_not_null(), _c("complete_pass") - _c("cp"))
    b["qb_archetype.epa_per_dropback"] = ("passer", db, _c("epa"))
    b["qb_archetype.cpoe"] = ("passer", att & _c("cpoe").is_not_null(), _c("cpoe"))
    b["qb_archetype.average_depth_of_target"] = ("passer", att & _c("air_yards").is_not_null(), _c("air_yards"))
    b["qb_archetype.sack_rate"] = ("passer", db, _c("sack"))
    b["qb_archetype.scramble_rate"] = ("passer", db, _c("qb_scramble"))
    b["qb_archetype.play_action_rate"] = ("passer", db & _c("ftn"), _c("is_play_action"))
    return b


_ENTITY_COL = {"rusher": "rusher_player_id", "passer": "qb_id", "receiver": "receiver_player_id"}


def _agg(book: dict, pick) -> list:
    out = []
    for key, spec in book.items():
        if pick(spec):
            num, den = nd(spec[-2], spec[-1])
            out += [num.alias(f"{key}|n"), den.alias(f"{key}|d")]
    return out


# ------------------------------------------------------------------------------------------------------------------------------ team ledger
def team_ledger(plays: pl.DataFrame, games: pl.DataFrame, pfr_pass: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team). Offense features use the team's own plays; defense features the plays it faced; PFR pressure features come
    from the quarterback rows (offense: the team's own, defense: the opposing team's, since the PFR def_* columns are empty)."""
    book = _team_book()
    help_off = [pl.col("f_rush").sum().alias("x.carries"), pl.col("f_db").sum().alias("x.dropbacks"),
                (pl.col("f_target")).sum().alias("x.targets"),
                (pl.col("f_target") & pl.col("air_yards").is_not_null()).sum().alias("x.targets_air"),
                pl.when(pl.col("f_target")).then(pl.col("air_yards").fill_null(0.0)).otherwise(0.0).sum().alias("x.air_yards"),
                (pl.col("f_rush") & (pl.col("yardline_100") <= cs.GOAL_LINE_YARDLINE)).sum().alias("x.gl_carries")]
    off = plays.group_by("game_id", "posteam").agg(_agg(book, lambda s: s[0] == "off") + help_off).rename({"posteam": "team"})
    dff = plays.group_by("game_id", "defteam").agg(_agg(book, lambda s: s[0] == "def") + [pl.col("f_db").sum().alias("x.dropbacks_faced")]).rename({"defteam": "team"})
    t = games.select("game_id", "team").join(off, on=["game_id", "team"], how="left").join(dff, on=["game_id", "team"], how="left")
    # PFR pressure / blitz (offense: own QB rows; defense: the opposing QB rows). A game without PFR rows has denominator 0 (missing).
    own = pfr_pass.group_by("game_id", "team").agg(pl.col("times_pressured").sum().alias("pf_pr_own"), pl.col("times_blitzed").sum().alias("pf_bl_own"))
    opp = pfr_pass.group_by("game_id", "opponent").agg(pl.col("times_pressured").sum().alias("pf_pr_opp"), pl.col("times_blitzed").sum().alias("pf_bl_opp")).rename({"opponent": "team"})
    t = t.join(own, on=["game_id", "team"], how="left").join(opp, on=["game_id", "team"], how="left")
    has_own, has_opp = pl.col("pf_pr_own").is_not_null(), pl.col("pf_pr_opp").is_not_null()
    t = t.with_columns(
        pl.col("pf_pr_own").fill_null(0.0).alias("ol_protection.pressure_rate_allowed|n"),
        pl.when(has_own).then(pl.col("x.dropbacks")).otherwise(0.0).alias("ol_protection.pressure_rate_allowed|d"),
        pl.col("pf_pr_opp").fill_null(0.0).alias("pass_rush.pressure_rate|n"),
        pl.when(has_opp).then(pl.col("x.dropbacks_faced")).otherwise(0.0).alias("pass_rush.pressure_rate|d"),
        pl.col("pf_bl_opp").fill_null(0.0).alias("pass_rush.blitz_rate_pfr|n"),
        pl.when(has_opp).then(pl.col("x.dropbacks_faced")).otherwise(0.0).alias("pass_rush.blitz_rate_pfr|d")).drop("pf_pr_own", "pf_bl_own", "pf_pr_opp", "pf_bl_opp")
    return t.with_columns(pl.col(c).fill_null(0.0) for c in t.columns if c.endswith("|n") or c.endswith("|d") or c.startswith("x."))


# ---------------------------------------------------------------------------------------------------------------------------- player ledger
def player_ledger(plays: pl.DataFrame, team: pl.DataFrame, positions: pl.DataFrame, snaps: pl.DataFrame, pfr_rush: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team, player_id) for every player with a rush, target or dropback; carries the shares of the team totals and the
    archetype / run / pass / receiving pieces as '<unit>.<feature>|n' and '|d'."""
    book = _player_book()
    frames = []
    for entity, col in _ENTITY_COL.items():
        aggs = _agg(book, lambda s, e=entity: s[0] == e)
        base = [pl.col("f_rush").sum().alias("c.carries")] if entity == "rusher" else \
               [pl.col("f_db").sum().alias("c.dropbacks")] if entity == "passer" else \
               [pl.col("f_target").sum().alias("c.targets"), (pl.col("f_target") & pl.col("air_yards").is_not_null()).sum().alias("c.targets_air"),
                pl.when(pl.col("f_target")).then(pl.col("air_yards").fill_null(0.0)).otherwise(0.0).sum().alias("c.air_yards")]
        if entity == "rusher":
            base.append((pl.col("f_rush") & (pl.col("yardline_100") <= cs.GOAL_LINE_YARDLINE)).sum().alias("c.gl_carries"))
        sub = plays.filter(pl.col(col).is_not_null() & (pl.col("f_rush") if entity == "rusher" else pl.col("f_db") if entity == "passer" else pl.col("f_target")))
        frames.append(sub.group_by("game_id", "posteam", col).agg(aggs + base).rename({"posteam": "team", col: "player_id"}))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=["game_id", "team", "player_id"], how="full", coalesce=True)
    out = out.join(games.select("game_id", "team", "season", "week", "gameday", "team_game_num", "opponent"), on=["game_id", "team"], how="inner")
    out = out.with_columns(pl.col(c).fill_null(0.0) for c in out.columns if c.startswith("c.") or c.endswith("|n") or c.endswith("|d"))
    out = attach_position(out, "player_id", positions)
    # shares of the team's totals (denominators are the team's totals in that game)
    t = team.select("game_id", "team", tc="x.carries", tt="x.targets", ta="x.air_yards", tg="x.gl_carries", td="x.dropbacks")
    out = out.join(t, on=["game_id", "team"], how="left")
    def share(key, num, den):
        return [pl.col(num).alias(f"{key}|n"), pl.col(den).alias(f"{key}|d")]
    out = out.with_columns(
        *share("rb_archetype.carry_share", "c.carries", "tc"), *share("rb_archetype.target_share", "c.targets", "tt"),
        *share("rb_archetype.goal_line_share", "c.gl_carries", "tg"), *share("receiver_archetype.target_share", "c.targets", "tt"),
        *share("receiver_archetype.air_yards_share", "c.air_yards", "ta"),
        pl.col("c.carries").alias("qb_archetype.rush_attempts_per_game|n"),
        pl.lit(1.0).alias("qb_archetype.rush_attempts_per_game|d"),
        pl.col("c.dropbacks").alias("x.dropback_share|n"), pl.col("td").alias("x.dropback_share|d"))
    # roster body measures (the same value every game: a pooled mean returns it; missing height / weight stays missing)
    for unit in ("rb_archetype", "receiver_archetype"):
        for m in ("height", "weight"):
            out = out.with_columns(pl.col(m).fill_null(0.0).alias(f"{unit}.{m}|n"), pl.when(pl.col(m).is_not_null()).then(1.0).otherwise(0.0).alias(f"{unit}.{m}|d"))
    # PFR yards after contact (per carry); snap share (RB1 snap share is built from it)
    pr = pfr_rush.rename({"gsis_id": "player_id"}).select("game_id", "team", "player_id", pfr_n="rushing_yards_after_contact", pfr_d="carries")
    sn = snaps.rename({"gsis_id": "player_id"}).select("game_id", "team", "player_id", "offense_pct")
    out = out.join(pr, on=["game_id", "team", "player_id"], how="left").join(sn, on=["game_id", "team", "player_id"], how="left")
    out = out.with_columns(pl.col("offense_pct").fill_null(0.0).alias("x.snap|n"), pl.when(pl.col("offense_pct").is_not_null()).then(1.0).otherwise(0.0).alias("x.snap|d"))
    out = out.with_columns(pl.col("pfr_n").fill_null(0.0).alias("rb_archetype.yards_after_contact|n"), pl.col("pfr_d").fill_null(0.0).alias("rb_archetype.yards_after_contact|d")).drop("pfr_n", "pfr_d", "tc", "tt", "ta", "tg", "td")
    return out.sort("season", "week", "game_id", "team", "player_id")


# ---------------------------------------------------------------------------------------- position-split team features
def structure_ledger(player: pl.DataFrame, team: pl.DataFrame) -> pl.DataFrame:
    """Team-game n / d pairs for the features built from who got the ball: receiver_usage.* and rb_rotation.*. RB1 / RB2 are the two running backs
    (position RB) with the most designed carries in that game (ties go to the smaller player id)."""
    tt = team.select("game_id", "team", "x.targets", "x.gl_carries")
    out = tt.select("game_id", "team")
    tg = player.filter(pl.col("c.targets") > 0)
    for grp in GROUPS:
        a = tg.filter(pl.col("group") == grp).group_by("game_id", "team").agg(
            pl.col("c.targets").sum().alias(f"receiver_usage.target_split_{grp}|n"), pl.col("c.air_yards").sum().alias(f"receiver_usage.adot_{grp}|n"),
            pl.col("c.targets_air").sum().alias(f"receiver_usage.adot_{grp}|d"))
        out = out.join(a, on=["game_id", "team"], how="left")
    top3 = (tg.sort("game_id", "team", "c.targets", "player_id", descending=[False, False, True, False])
            .group_by("game_id", "team", maintain_order=True).agg(pl.col("c.targets").head(3).sum().alias("receiver_usage.top3_target_share|n")))
    out = out.join(top3, on=["game_id", "team"], how="left").join(tt.select("game_id", "team", "x.targets", "x.gl_carries"), on=["game_id", "team"], how="left")
    out = out.with_columns(*[pl.col("x.targets").alias(f"receiver_usage.target_split_{g}|d") for g in GROUPS],
                           pl.col("x.targets").alias("receiver_usage.top3_target_share|d"),
                           pl.col("receiver_usage.target_split_rb|n").alias("rb_rotation.rb_target_share|n"), pl.col("x.targets").alias("rb_rotation.rb_target_share|d"))
    rb = (player.filter((pl.col("group") == "rb") & (pl.col("c.carries") > 0))
          .sort("game_id", "team", "c.carries", "player_id", descending=[False, False, True, False])
          .with_columns(rank=pl.int_range(pl.len()).over("game_id", "team"), rb_total=pl.col("c.carries").sum().over("game_id", "team")))
    for r, name in ((0, "rb1"), (1, "rb2")):
        a = rb.filter(pl.col("rank") == r).select("game_id", "team", pl.col("c.carries").alias(f"rb_rotation.{name}_carry_share|n"), pl.col("rb_total").alias(f"rb_rotation.{name}_carry_share|d"),
                                                  *( [pl.col("c.gl_carries").alias("rb_rotation.goal_line_carry_share|n"), pl.col("offense_pct").fill_null(0.0).alias("rb_rotation.rb1_snap_share|n"),
                                                      pl.when(pl.col("offense_pct").is_not_null()).then(1.0).otherwise(0.0).alias("rb_rotation.rb1_snap_share|d")] if r == 0 else []))
        out = out.join(a, on=["game_id", "team"], how="left")
    out = out.with_columns(pl.col("x.gl_carries").alias("rb_rotation.goal_line_carry_share|d"))
    return out.with_columns(pl.col(c).fill_null(0.0) for c in out.columns if c.endswith("|n") or c.endswith("|d")).drop("x.targets", "x.gl_carries")


def coverage_position_ledger(plays: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """pass_coverage.completion_rate_allowed_<wr|te|rb>: completion rate on targets, split by the targeted receiver's position (defense side)."""
    t = plays.filter(pl.col("f_target")).select("game_id", "season", "week", "defteam", "complete_pass", gsis_id=pl.col("receiver_player_id"))
    t = attach_position(t, "gsis_id", positions, extra=()).filter(pl.col("group").is_not_null())
    out = None
    for grp in GROUPS:
        a = (t.filter(pl.col("group") == grp).group_by("game_id", "defteam")
             .agg(pl.col("complete_pass").sum().alias(f"pass_coverage.completion_rate_allowed_{grp}|n"), pl.len().cast(pl.Float64).alias(f"pass_coverage.completion_rate_allowed_{grp}|d"))
             .rename({"defteam": "team"}))
        out = a if out is None else out.join(a, on=["game_id", "team"], how="full", coalesce=True)
    return out.with_columns(pl.col(c).fill_null(0.0) for c in out.columns if c.endswith("|n") or c.endswith("|d"))


def completeness(plays: pl.DataFrame, games: pl.DataFrame, team: pl.DataFrame, pfr_pass: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """Share of the relevant rows each source populates, per season (the participation coverage columns are the estimated ones)."""
    att = plays.filter(pl.col("f_att"))
    rows = []
    def add(src, what, df, expr):
        for r in df.group_by("season").agg(expr.alias("pct")).sort("season").iter_rows(named=True):
            rows.append(dict(source=src, measure=what, season=r["season"], pct=None if r["pct"] is None else float(r["pct"]) * 100))
    add("participation", "pass attempts with man/zone classified", att, pl.col("man_zone").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"]).mean())
    add("participation", "pass attempts with a coverage shell classified (COVER_1/2/3/4/6, 2_MAN)", att, pl.col("coverage_type").is_in(list(CLOSED_SHELLS + OPEN_SHELLS)).mean())
    add("ftn_charting", "team plays with an FTN row", plays.filter(pl.col("f_play")), pl.col("ftn").mean())
    tg = games.select("game_id", "team", "season").join(pfr_pass.group_by("game_id", "team").agg(pl.len().alias("n")), on=["game_id", "team"], how="left")
    add("pfr_advstats_pass", "team-games with a PFR passing row", tg, pl.col("n").is_not_null().mean())
    tg = games.select("game_id", "team", "season").join(snaps.group_by("game_id", "team").agg(pl.len().alias("n")), on=["game_id", "team"], how="left")
    add("snap_counts", "team-games with snap counts", tg, pl.col("n").is_not_null().mean())
    return pl.DataFrame(rows, schema={"source": pl.String, "measure": pl.String, "season": pl.Int32, "pct": pl.Float64})


# ------------------------------------------------------------------------------------------------------------------------------ S5 results-side statistics
def interaction_ledger(plays: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """'x.<name>|n' / '|d' per (game_id, team) for models/comps_spec.INTERACTION_EXTRA (FTN, 2022+; 0 denominators before): the defense's epa allowed on motion /
    play-action / screen plays (grouped by defteam) and the offense's epa per dropback against a blitz (grouped by posteam)."""
    play, db, ftn = _c("f_play"), _c("f_db"), _c("ftn")
    d_specs = {"x.def_epa_vs_motion": (play & ftn & (_c("is_motion") == 1), _c("epa")),
               "x.def_epa_vs_play_action": (db & ftn & (_c("is_play_action") == 1), _c("epa")),
               "x.def_epa_vs_screen": (db & ftn & (_c("is_screen_pass") == 1), _c("epa"))}
    o_specs = {"x.off_epa_vs_blitz": (db & ftn & (_c("n_blitzers") > 0), _c("epa"))}
    def agg(specs, by, name):
        cols = []
        for k, (fl, v) in specs.items():
            n, d = nd(fl, v)
            cols += [n.alias(f"{k}|n"), d.alias(f"{k}|d")]
        return plays.group_by("game_id", by).agg(cols).rename({by: "team"})
    out = (games.select("game_id", "team").join(agg(d_specs, "defteam", "def"), on=["game_id", "team"], how="left")
           .join(agg(o_specs, "posteam", "off"), on=["game_id", "team"], how="left"))
    return out.with_columns(pl.col(c).fill_null(0.0) for c in out.columns if c.endswith("|n") or c.endswith("|d"))

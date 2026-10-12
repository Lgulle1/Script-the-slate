"""4d.1 coaching prior (build plan 4d.1; gate and design fixed in docs/overnight_plan.md "4d.1 gate" before anything was built).

For a team whose head coach is new, a prior from his PRIOR stops with other teams, three tendencies from play-by-play:
  pass rate over expected   mean(pass - xpass) over designed pass / run plays (nflfastR xpass; derived)
  pace                      median seconds between a team's consecutive scrimmage plays in neutral states
  fourth-down aggression    go rate minus the league's same-season go rate in the same situation, in neutral states
A stop is weighted by recency, 0.5 ** (seasons since it ended / 3), and by length, games / (games + 17); the prior is the weighted mean over the coach's
earlier stops. A coach with no earlier stop has no prior (null); a team without a change has none either. Walk-forward: a stop counts only when it
ended before the target game, and never a stop with the target's own team. A coach is "new" for his first NEW_COACH_GAMES games with a team.

Head coaches are the nflverse per-game column (ingest/build_coaches_history.py); under option 3 the offensive coordinator and play caller are
UNKNOWN, so only head-coach stops make a prior. Feature columns (float, null when unknown, never zero): coach_prior_proe, coach_prior_pace,
coach_prior_go4, coach_prior_weight (BLEND_K / (BLEND_K + games under him so far)), coach_games, coach_state (0 not new, 1 new with a prior, 2 new
without one). 2025 (the holdout) is never loaded.
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from ingest.build_coaches_history import FRANCHISE

RECENCY_HALF_LIFE_SEASONS = 3.0
LENGTH_K = 17.0
BLEND_K = 6.0
NEW_COACH_GAMES = 17
WINDOW_GAMES = 6
EARLY_WEEKS = 6
WP_LOW, WP_HIGH = 0.2, 0.8
END_OF_HALF_SECONDS = 120.0
MAX_GAP_SECONDS = 60.0
YDSTOGO_BUCKETS = ((1, "1"), (2, "2"), (4, "3-4"), (7, "5-7"), (99, "8+"))
YARDLINE_BUCKETS = ((20, "1-20"), (40, "21-40"), (60, "41-60"), (80, "61-80"), (99, "81-99"))
PBP_COLUMNS = ("game_id", "season", "week", "posteam", "drive", "play_id", "down", "ydstogo", "yardline_100", "play_type", "pass", "rush", "xpass", "wp",
               "game_seconds_remaining", "half_seconds_remaining", "timeout", "penalty", "qb_spike", "qb_kneel")
TENDENCIES = ("proe", "pace", "go4")


def load_pbp(raw_db=config.RAW_DUCKDB_PATH, first: int = 2016, last: int = max(config.BACKTEST_SEASONS)) -> pl.DataFrame:
    """Regular-season plays with a possession team, first..last; never the holdout."""
    if last >= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"plays through {last} would read the holdout season {config.HOLDOUT_SEASON}")
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute(f"SELECT {', '.join(chr(34) + c + chr(34) for c in PBP_COLUMNS)} FROM pbp WHERE season_type = 'REG' AND posteam IS NOT NULL "
                           "AND season BETWEEN ? AND ? ORDER BY game_id, play_id", [first, last]).pl()
    finally:
        con.close()


# ------------------------------------------------------------------------------------------------------------------------ stints
def coach_games(games: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: franchise, coach, key (season * 100 + week), stint (a run of games under one coach with one franchise), games_before (games
    under him in this stint before this one) and change_event (his first game of a stint that follows another coach's game)."""
    g = (games.with_columns(franchise=pl.col("team").replace(FRANCHISE), key=pl.col("season") * 100 + pl.col("week"), coach=pl.col("head_coach"))
         .sort("franchise", "season", "week", "game_id"))
    g = g.with_columns(coach=pl.col("coach").forward_fill().over("franchise"))
    prev = pl.col("coach").shift(1).over("franchise")
    g = g.with_columns(new_stint=prev.is_null() | (pl.col("coach") != prev), had_prev=prev.is_not_null())
    g = g.with_columns(stint=pl.col("new_stint").cum_sum().over("franchise"), change_event=pl.col("new_stint") & pl.col("had_prev"))
    g = g.with_columns(games_before=pl.int_range(pl.len()).over("franchise", "stint"))
    return g.select("season", "week", "game_id", "team", "franchise", "coach", "key", "stint", "games_before", "change_event")


# ------------------------------------------------------------------------------------------------------------------------ tendencies
def _flag(name: str) -> pl.Expr:
    return pl.col(name).fill_null(0.0) == 1.0


def _scrimmage() -> pl.Expr:
    """A designed pass or run: no penalty-only, timeout, spike or kneel row."""
    return (_flag("pass") | _flag("rush")) & ~_flag("penalty") & ~_flag("timeout") & ~_flag("qb_spike") & ~_flag("qb_kneel")


def _neutral() -> pl.Expr:
    return pl.col("wp").is_between(WP_LOW, WP_HIGH) & (pl.col("half_seconds_remaining") > END_OF_HALF_SECONDS)


def proe_rows(p: pl.DataFrame) -> pl.DataFrame:
    return p.filter(_scrimmage() & pl.col("xpass").is_not_null()).with_columns(proe=pl.col("pass").fill_null(0.0) - pl.col("xpass"))


def proe(p: pl.DataFrame) -> tuple:
    r = proe_rows(p)
    return (float(r["proe"].mean()), r.height) if r.height else (None, 0)


def pace_gaps(p: pl.DataFrame) -> pl.DataFrame:
    """Per neutral scrimmage play with a previous scrimmage play in the same drive and no timeout between them: the seconds since that play."""
    q = (p.filter(pl.col("game_seconds_remaining").is_not_null()).sort("game_id", "play_id")
         .with_columns(scr=_scrimmage(), tcount=_flag("timeout").cast(pl.Int64).cum_sum().over("game_id", "drive")))
    s = q.filter(pl.col("scr")).with_columns(prev_clock=pl.col("game_seconds_remaining").shift(1).over("game_id", "drive"),
                                              prev_t=pl.col("tcount").shift(1).over("game_id", "drive"))
    s = s.with_columns(dt=pl.col("prev_clock") - pl.col("game_seconds_remaining"))
    return s.filter(pl.col("prev_clock").is_not_null() & (pl.col("prev_t") == pl.col("tcount")) & (pl.col("dt") > 0) & (pl.col("dt") <= MAX_GAP_SECONDS)
                    & _neutral()).select("game_id", "posteam", "dt")


def pace(p: pl.DataFrame) -> tuple:
    d = pace_gaps(p)
    return (float(d["dt"].median()), d.height) if d.height else (None, 0)


def _bucket(col: str, edges) -> pl.Expr:
    e = pl.when(pl.col(col) <= edges[0][0]).then(pl.lit(edges[0][1]))
    for hi, name in edges[1:]:
        e = e.when(pl.col(col) <= hi).then(pl.lit(name))
    return e.otherwise(pl.lit(edges[-1][1]))


def fourth_rows(p: pl.DataFrame) -> pl.DataFrame:
    """Neutral fourth downs with a decision (pass / run = go; punt / field goal): go, and the league's same-season go rate in the same situation."""
    f = (p.filter((pl.col("down") == 4) & _neutral() & pl.col("play_type").is_in(["pass", "run", "punt", "field_goal"]))
         .with_columns(go=pl.col("play_type").is_in(["pass", "run"]).cast(pl.Float64), yb=_bucket("ydstogo", YDSTOGO_BUCKETS),
                       lb=_bucket("yardline_100", YARDLINE_BUCKETS)))
    return f.with_columns(exp_go=pl.col("go").mean().over("season", "yb", "lb"))


def fourth_down(p: pl.DataFrame, by: str = "posteam") -> pl.DataFrame:
    return fourth_rows(p).group_by(by).agg(go4=(pl.col("go") - pl.col("exp_go")).mean(), n=pl.len()).sort(by)


def tendencies(pbp: pl.DataFrame, cg: pl.DataFrame) -> pl.DataFrame:
    """One row per stop (coach, franchise, stint): games, first / last key and season, and the three tendencies with their sample sizes."""
    p = pbp.with_columns(franchise=pl.col("posteam").replace(FRANCHISE))
    who = cg.filter(pl.col("coach").is_not_null()).select("game_id", "franchise", "coach", "stint")
    stops = (cg.filter(pl.col("coach").is_not_null()).group_by("coach", "franchise", "stint")
             .agg(n_games=pl.len(), first_key=pl.col("key").min(), last_key=pl.col("key").max(), last_season=pl.col("season").max()))
    keys = ["coach", "franchise", "stint"]
    a = proe_rows(p).join(who, on=["game_id", "franchise"]).group_by(keys).agg(proe=pl.col("proe").mean(), n_proe=pl.len())
    b = (pace_gaps(p).with_columns(franchise=pl.col("posteam").replace(FRANCHISE)).join(who, on=["game_id", "franchise"])
         .group_by(keys).agg(pace=pl.col("dt").median(), n_pace=pl.len()))
    c = (fourth_rows(p).join(who, on=["game_id", "franchise"]).group_by(keys)
         .agg(go4=(pl.col("go") - pl.col("exp_go")).mean(), n_go4=pl.len()))
    out = stops.join(a, on=keys, how="left").join(b, on=keys, how="left").join(c, on=keys, how="left")
    return out.sort("coach", "franchise", "stint")


# ------------------------------------------------------------------------------------------------------------------------ the prior
def prior(stops: pl.DataFrame, coach: str, franchise: str, season: int, week: int):
    """The coach's prior at the start of (season, week) at `franchise`: the weighted mean of each tendency over his stops with OTHER franchises that
    ended strictly before that week, or None without such a stop. Also 'weight', the sum of the stop weights."""
    s = stops.filter((pl.col("coach") == coach) & (pl.col("franchise") != franchise) & (pl.col("last_key") < season * 100 + week))
    if s.height == 0:
        return None
    w = ((0.5 ** ((season - s["last_season"]) / RECENCY_HALF_LIFE_SEASONS)) * (s["n_games"] / (s["n_games"] + LENGTH_K))).to_numpy()
    out = {"weight": float(w.sum())}
    for name in TENDENCIES:
        v = s[name].cast(pl.Float64).to_numpy()                            # a missing tendency is NaN
        ok = ~np.isnan(v)
        out[name] = float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else None
    return out


def features(games: pl.DataFrame, stops: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: coach_state / coach_games / coach_prior_* (see the module doc), from the stints and the stop tendencies."""
    cg = coach_games(games)
    stint_new = cg.group_by("franchise", "stint").agg(new=pl.col("change_event").any(), season0=pl.col("season").min(), week0=pl.col("week").min(),
                                                     coach=pl.col("coach").first())
    pri = {}
    for r in stint_new.filter(pl.col("new")).iter_rows(named=True):
        pri[(r["franchise"], r["stint"])] = prior(stops, r["coach"], r["franchise"], r["season0"], r["week0"])
    new_by = {(r["franchise"], r["stint"]): r["new"] for r in stint_new.iter_rows(named=True)}
    out = []
    for r in cg.iter_rows(named=True):
        k = (r["franchise"], r["stint"])
        is_new = new_by[k] and r["games_before"] < NEW_COACH_GAMES
        pr = pri.get(k) if is_new else None
        state = 0.0 if not is_new else (1.0 if pr is not None else 2.0)
        has = state == 1.0
        out.append(dict(game_id=r["game_id"], team=r["team"], season=r["season"], week=r["week"], coach_state=state, coach_games=float(r["games_before"]),
                        coach_prior_proe=pr["proe"] if has else None, coach_prior_pace=pr["pace"] if has else None,
                        coach_prior_go4=pr["go4"] if has else None,
                        coach_prior_weight=BLEND_K / (BLEND_K + r["games_before"]) if has else None))
    schema = {"game_id": pl.String, "team": pl.String, "season": pl.Int64, "week": pl.Int64, "coach_state": pl.Float64, "coach_games": pl.Float64,
              "coach_prior_proe": pl.Float64, "coach_prior_pace": pl.Float64, "coach_prior_go4": pl.Float64, "coach_prior_weight": pl.Float64}
    return pl.DataFrame(out, schema=schema).sort("season", "week", "game_id", "team")


def windows(f: pl.DataFrame) -> pl.DataFrame:
    """The gate's row sets per team-game: early (weeks 1-6), post_change (new head coach, fewer than WINDOW_GAMES games under him), and window
    'W1+W2' (either) or 'W0' (neither)."""
    early = pl.col("week") <= EARLY_WEEKS
    post = (pl.col("coach_state") >= 1.0) & (pl.col("coach_games") < WINDOW_GAMES)
    return (f.with_columns(early=early.fill_null(False), post_change=post.fill_null(False))
            .with_columns(window=pl.when(pl.col("early") | pl.col("post_change")).then(pl.lit("W1+W2")).otherwise(pl.lit("W0")))
            .select("game_id", "team", "season", "week", "early", "post_change", "window"))


COACH_COLUMNS = ["coach_state", "coach_games", "coach_prior_proe", "coach_prior_pace", "coach_prior_go4", "coach_prior_weight"]


def attach(tables, feats: pl.DataFrame | None):
    """FeatureTables with the coaching columns joined on (game_id, team): players and the team table alike. None = the Phase 3 tables unchanged."""
    from features.volume_features import FeatureTables
    if feats is None:
        return tables
    missing = [c for c in COACH_COLUMNS if c not in feats.columns]
    if missing:
        raise KeyError(f"coaching features are missing columns {missing}")
    if feats.select("game_id", "team").is_duplicated().any():
        raise ValueError("coaching features have more than one row for a (game_id, team)")
    f = feats.select("game_id", "team", *[pl.col(c).cast(pl.Float64) for c in COACH_COLUMNS])
    players = {q: df.join(f, on=["game_id", "team"], how="left", maintain_order="left") for q, df in tables.players.items()}
    return FeatureTables(players, tables.teams.join(f, on=["game_id", "team"], how="left", maintain_order="left"))


def build(raw_db=config.RAW_DUCKDB_PATH) -> tuple:
    """(features for 2016-2024, stops): head coaches per game from nflverse schedules, plays from the raw database; the holdout is never read."""
    from ingest.build_coaches_history import load_head_coach_games
    games = load_head_coach_games(raw_db)
    cg = coach_games(games)
    stops = tendencies(load_pbp(raw_db), cg)
    return features(games, stops), stops

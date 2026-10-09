"""4c.1 -- multi-window fingerprint vectors for the comparables engine.

What a vector is. For an entity (a team, or a player archetype) and a target game, one standardized vector per UNIT and per WINDOW, built from
games played strictly before the target week's first kickoff. The units, their features, spaces, windows and data-quality tags are defined in
models/comps_spec.py; nothing here decides what a component is.

  windows     last_3, last_6, season_to_date   the entity's own earlier games, weight 1
              recency_weighted                 weight 0.5 ** (games ago / RECENCY_HALF_LIFE_GAMES); an offseason counts OFFSEASON_GAP_GAMES
                                               (features/weights.py: the same clock as games_elapsed)
              continuity_weighted              recency weight x features.weights.continuity_weight, per penalty row (one of the rows the
                                               markets use: pass_yds, rush_att, rush_yds, rec, rec_yds, game_total). It multiplies the recency
                                               weight, otherwise a game from years ago with an unchanged lineup would count as much as last week's.
  value       a pooled rate  sum(w n) / sum(w d)  over the window (n / d from features/comps_ledger.py); missing when the denominator is 0
              (no games, or the source does not exist for those seasons): a missing feature is NaN, never 0.
  z-score     against the league for the same season-week and the same window: the mean and SD of that window's values over every team
              (player archetypes: every player with a game that week) whose window denominator is at least LEAGUE_MIN_REL_DEN of that week's median
              positive denominator for the feature (so a per-game feature and a per-carry feature are each judged on their own scale), computed from
              games before that week only; clipped to +-Z_CLIP. Player rates are first shrunk toward the league mean by SHRINK_K opportunities,
              except the per-game and body-measure features (NO_SHRINK).
  spaces      BASE = the unit's BASE features. EXTENDED = BASE + the FTN / PFR / participation features. Units whose EXTENDED feature list equals the
              BASE one are stored once, as BASE. A search uses EXTENDED only when every EXTENDED feature of the unit is present on both sides.

Lineup versions (team units only; player archetypes describe the player himself and do not change with the lineup)
  healthy     the team's own pooled value: its normal lineup. Also the vector of a historical pool game, for units no player decomposition moves.
  adjusted    healthy + sum over players of (expected share - normal share) x (his window value - the team's), with the shares from the 4a injury
              layer for the game (4a.1 play probability and 4a.2 redistributed carry / target / dropback shares: detail.exp_* vs detail.b_*).
  actual      the same correction for who actually played (an offensive snap in snap_counts): each player's normal usage share (his trailing
              share of the team's plays over the team's games, absences counted as 0) restricted to the players who played and renormalised, against
              the full normal shares. The touches that fell in the game are NOT used (that is game-to-game noise, not the lineup). Used for HISTORICAL POOL
              games: a pool vector reflects who played, not the pre-game expectation.
Units with a lineup correction: run_offense, pass_offense (quarterback-attributed features), receiver_usage, rb_rotation. ol_protection and the
four defense units have no player-level data in the spec (the 4a layer models offensive skill players), so adjusted = healthy there, and so do the
scheme features (play action, screen, motion, no-huddle, rush rate over expected); LINEUP_NOT_ADJUSTED lists them and the build log prints it.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

import config
from features import weights as wts
from models import comps_spec as cs

Z_CLIP = 4.0
LEAGUE_MIN_REL_DEN = 0.5               # a window feeds the league mean / SD only if its denominator is at least this share of the week's median positive one
MIN_LEAGUE_ENTITIES = 8                # fewer entities than this in a week-window: no league stats, z is missing
LINEUP_COMPONENTS = ("QB", "RB", "WRTE", "OL")        # features/lineups.py component order
PENALTY_KEYS = tuple(sorted({s["penalty_key"] for s in config.MARKETS.values()}))
VARIANTS = tuple((w, None) for w in ("last_3", "last_6", "season_to_date", "recency_weighted")) + tuple(("continuity_weighted", k) for k in PENALTY_KEYS)
assert {w for w, _ in VARIANTS} == set(cs.WINDOWS), "comps_spec.WINDOWS and the windows built here must agree"

TEAM_UNITS = cs.OFFENSE_UNITS + cs.DEFENSE_UNITS
LINEUP_UNITS = ("run_offense", "pass_offense", "receiver_usage", "rb_rotation")


def vname(variant: tuple) -> str:
    return variant[0] if variant[1] is None else f"{variant[0]}:{variant[1]}"


def feature_keys(unit: str, space: str = "extended") -> list:
    """'unit.feature' keys of a unit in a space (EXTENDED = every feature, BASE = the base ones)."""
    return [f"{unit}.{f.name}" for f in cs.FEATURES[unit] if space == "extended" or f.space == "base"]


def stored_spaces(unit: str) -> tuple:
    """The spaces stored for a unit: BASE when it has base features, and EXTENDED too when that adds features (a unit with no BASE feature is
    EXTENDED only)."""
    spaces = {f.space for f in cs.FEATURES[unit]}
    return tuple(s for s in cs.SPACES if (s == "base" and "base" in spaces) or (s == "extended" and ("extended" in spaces)))


# ---------------------------------------------------------------------------------------------------------------------------------- the clock
def _games_in_season(season: int) -> int:
    return 17 if season >= 2021 else 16


_CLOCK_OFFSET = {}
_acc = 0.0
for _s in range(1999, 2041):
    _CLOCK_OFFSET[_s] = _acc
    _acc += _games_in_season(_s) + config.OFFSEASON_GAP_GAMES


def game_clock(season, game_num) -> np.ndarray:
    """Position of a team game on one line, in team games, offseason gaps included: clock(target) - clock(past) is wts.games_elapsed(...)."""
    season = np.asarray(season)
    return np.array([_CLOCK_OFFSET[int(s)] for s in season.ravel()], dtype=float).reshape(season.shape) + np.asarray(game_num, dtype=float)


@dataclass
class Timeline:
    """Parallel arrays describing games: for the history side `day` is the game date, for the query side the week's cutoff (first kickoff)."""
    day: np.ndarray
    clock: np.ndarray
    season: np.ndarray
    lin: np.ndarray            # (4, n) lineup-group ids, -1 = unknown
    slot: np.ndarray
    fam: np.ndarray
    team: np.ndarray

    def __len__(self):
        return self.day.size


def window_weights(hist: Timeline, query: Timeline, variant: tuple) -> np.ndarray:
    """(n_query, n_hist) weights of the history games in the variant's window as of each query. `hist` must be sorted by day. Games on or after
    the query's cutoff weigh 0: nothing from the target week or later enters."""
    window, key = variant
    prior = hist.day[None, :] < query.day[:, None]
    if window.startswith("last_"):
        k = int(window[5:])
        idx = np.arange(len(hist))[None, :]
        return (prior & (idx >= (prior.sum(axis=1) - k)[:, None])).astype(float)
    if window == "season_to_date":
        return (prior & (hist.season[None, :] == query.season[:, None])).astype(float)
    rec = np.where(prior, 0.5 ** (np.maximum(query.clock[:, None] - hist.clock[None, :], 0.0) / config.RECENCY_HALF_LIFE_GAMES), 0.0)
    if window == "recency_weighted":
        return rec
    lin_past = {c: hist.lin[i][None, :] for i, c in enumerate(LINEUP_COMPONENTS)}
    lin_target = {c: query.lin[i][:, None] for i, c in enumerate(LINEUP_COMPONENTS)}
    flags = wts.continuity_flags(lin_past, lin_target, hist.slot[None, :], query.slot[:, None], hist.fam[None, :], query.fam[:, None],
                                 hist.team[None, :] != query.team[:, None])
    return rec * wts.continuity_weight(flags, key)


def pooled(weights: np.ndarray, num: np.ndarray, den: np.ndarray) -> tuple:
    """(rate, weighted denominator) of the weighted pooled rate; the rate is NaN where the weighted denominator is 0."""
    n, d = weights @ num, weights @ den
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0), d


# ---------------------------------------------------------------------------------------------------------------------------------- league z
def week_stats(values: np.ndarray, den: np.ndarray, keys: np.ndarray, rel: float = LEAGUE_MIN_REL_DEN) -> dict:
    """{key: (mean (F,), sd (F,))} per season-week over the rows of that key whose denominator is at least `rel` x the key's median positive
    denominator for the feature. NaN where fewer than MIN_LEAGUE_ENTITIES rows qualify or there is no spread."""
    out = {}
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    cuts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        v, d = values[idx], den[idx]
        pos = np.where(d > 0, d, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # an all-missing column has no median: no league for it
            thr = rel * np.nanmedian(pos, axis=0)
        ok = (d > 0) & (d >= np.nan_to_num(thr, nan=np.inf)[None, :]) & ~np.isnan(v)
        cnt = ok.sum(axis=0)
        vv = np.where(ok, v, 0.0)
        m = np.divide(vv.sum(axis=0), cnt, out=np.full(v.shape[1], np.nan), where=cnt > 0)
        var = np.divide((np.where(ok, v - m, 0.0) ** 2).sum(axis=0), np.maximum(cnt - 1, 1), out=np.full(v.shape[1], np.nan), where=cnt > 1)
        sd = np.sqrt(var)
        sd = np.where((cnt >= MIN_LEAGUE_ENTITIES) & (sd > 0), sd, np.nan)
        out[int(ks[a])] = (np.where(np.isnan(sd), np.nan, m), sd)
    return out


def apply_week_stats(values: np.ndarray, keys: np.ndarray, stats: dict) -> tuple:
    """(z clipped to +-Z_CLIP, mean, sd) of `values` (n, F) against the stats of their own key (missing key -> NaN)."""
    nf = values.shape[1]
    nan = (np.full(nf, np.nan), np.full(nf, np.nan))
    mu = np.vstack([stats.get(int(k), nan)[0] for k in keys]) if len(keys) else np.zeros((0, nf))
    sd = np.vstack([stats.get(int(k), nan)[1] for k in keys]) if len(keys) else np.zeros((0, nf))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.clip((values - mu) / sd, -Z_CLIP, Z_CLIP)
    return z, mu, sd


def league_z(values: np.ndarray, den: np.ndarray, keys: np.ndarray, rel: float = LEAGUE_MIN_REL_DEN) -> tuple:
    """z-scores of `values` (n, F) against the mean / SD of the rows sharing the same key (season-week), among rows whose denominator is at least
    `rel` x the median positive denominator of that key for the feature. Returns (z clipped to +-Z_CLIP, mean, sd), the last two (n, F) broadcast
    per row. NaN where the league has fewer than MIN_LEAGUE_ENTITIES eligible rows or no spread."""
    return apply_week_stats(values, keys, week_stats(values, den, keys, rel))


# ---------------------------------------------------------------------------------------------------------------------------------- inputs
@dataclass
class Inputs:
    games: pl.DataFrame            # team-games (game_id, team, opponent, season, week, gameday, team_game_num), sorted season, week, game_id, team
    lineups: pl.DataFrame          # team, season, week, QB, RB, WRTE, OL
    team: pl.DataFrame             # wide '<unit>.<feature>|n' / '|d' per (game_id, team), team-level features only
    player: pl.DataFrame           # player ledger (features/comps_ledger.player_ledger)
    slots: pl.DataFrame            # gsis_id, season, week, slot, family
    positions: pl.DataFrame        # gsis_id, season, week, position, height, weight (rosters_weekly)
    snaps: pl.DataFrame | None = None        # game_id, team, gsis_id of every player with an offensive snap (snap_counts): who played
    depth_chart: pl.DataFrame | None = None  # team, season, week, gsis_id, family, slot: the PREGAME population the archetype z-scores use
    targets: pl.DataFrame | None = None      # game_id, team, player_id, family of every scored player-game: each gets a vector, played or not
    detail: pl.DataFrame | None = None       # 4a expected shares (sim.inputs._detail_and_exit): game_id, team, player_id, group, exp_*, b_*, p_out
    completeness: pl.DataFrame | None = None


def _epoch_days(col: pl.Series) -> np.ndarray:
    return col.dt.epoch("d").to_numpy().astype(np.int64)


def _lineup_array(games: pl.DataFrame, lineups: pl.DataFrame) -> np.ndarray:
    j = games.select("team", "season", "week").join(lineups, on=["team", "season", "week"], how="left", maintain_order="left")
    return np.vstack([j[c].fill_null(-1).to_numpy().astype(np.int64) for c in LINEUP_COMPONENTS])


def week_cutoffs(games: pl.DataFrame) -> pl.DataFrame:
    """The first kickoff of each (season, week): the harness cutoff. A window for a game in that week sees only games before it."""
    return games.group_by("season", "week").agg(cutoff=pl.col("gameday").min())


def team_windows(games: pl.DataFrame, lineups: pl.DataFrame, wide: pl.DataFrame, keys: list, variants=VARIANTS) -> dict:
    """{variant: (rate, den)} (n_games x n_keys) of the team's own pooled window values as of each of its games (rows follow `games`)."""
    tw = games.select("game_id", "team", "season", "week", "gameday", "team_game_num").join(wide, on=["game_id", "team"], how="left", maintain_order="left")
    assert tw.height == games.height
    cut = games.join(week_cutoffs(games), on=["season", "week"], how="left", maintain_order="left")["cutoff"]
    num = tw.select([f"{k}|n" for k in keys]).fill_null(0.0).to_numpy()
    den = tw.select([f"{k}|d" for k in keys]).fill_null(0.0).to_numpy()
    day, cut_day = _epoch_days(tw["gameday"]), _epoch_days(cut)
    clk = game_clock(tw["season"].to_numpy(), tw["team_game_num"].to_numpy())
    season = tw["season"].to_numpy()
    lin = _lineup_array(games, lineups)
    team_code = tw["team"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    zeros = np.zeros(len(tw), dtype=np.int64)
    out = {v: (np.full(num.shape, np.nan), np.zeros(num.shape)) for v in variants}
    for t in np.unique(team_code):
        idx = np.flatnonzero(team_code == t)
        idx = idx[np.argsort(day[idx], kind="stable")]
        hist = Timeline(day[idx], clk[idx], season[idx], lin[:, idx], zeros[idx], zeros[idx], team_code[idx])
        query = Timeline(cut_day[idx], clk[idx], season[idx], lin[:, idx], zeros[idx], zeros[idx], team_code[idx])
        for v in variants:
            rate, d = pooled(window_weights(hist, query, v), num[idx], den[idx])
            out[v][0][idx], out[v][1][idx] = rate, d
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- players
FAMILY_CODE = {"QB": 1, "RB": 2, "WR": 3, "TE": 4}
NO_SHRINK = ("height", "weight", "rush_attempts_per_game", "rb1_snap_share")      # body measures and per-game features: their denominator is games, not opportunities
# A player-game is an observation of an archetype when the player PLAYED at the position (an offensive snap in snap_counts, or a carry / target /
# dropback), not when he had a touch: a game with zero targets is an observation (of 0 targets) for the receiver archetype, and a target whose player got
# none must still have a vector.
ARCHETYPE_POSITIONS = {"rb_archetype": ("RB",), "receiver_archetype": ("WR", "TE", "RB"), "qb_archetype": ("QB",)}
SHARE_KEYS = {"carry": "u.carry_share", "target": "u.target_share", "dropback": "u.dropback_share"}      # usage shares over the TEAM's games (absences count as 0)
RUN_KEYS = [k for k in (f"run_offense.{f.name}" for f in cs.FEATURES["run_offense"]) if k != "run_offense.rush_rate_over_expected"]
PASS_KEYS = [f"pass_offense.{n}" for n in ("epa_per_dropback", "cpoe", "yards_per_attempt", "explosive_pass_rate", "average_depth_of_target", "sack_rate")]
EXTRA_KEYS = ["receiver_archetype.average_depth_of_target", "rb_archetype.goal_line_share", "x.snap"]


def player_keys(player: pl.DataFrame) -> list:
    """Every player-ledger feature that has both a numerator and a denominator column."""
    return sorted({c[:-2] for c in player.columns if c.endswith("|n")} & {c[:-2] for c in player.columns if c.endswith("|d")})


def _team_codes(*frames) -> dict:
    names = sorted({t for f in frames for t in f.unique().to_list()})
    return {t: i for i, t in enumerate(names)}


@dataclass
class PlayerWindows:
    queries: pl.DataFrame          # game_id, team, player_id, season, week, group, position (one row per query)
    keys: list
    rate: np.ndarray               # (n_variants, n_queries, n_keys) float32: pooled window value (NaN = no opportunities)
    den: np.ndarray                # same shape: weighted denominator
    variants: tuple


def population_table(inp: Inputs) -> pl.DataFrame:
    """(game_id, team, player_id, position) of every player who played: a ledger row (a touch) or an offensive snap, at QB / RB / WR / TE by his roster
    position or by his family on that week's depth chart."""
    from features import comps_ledger as L
    led = inp.player.select("game_id", "team", "player_id", "position")
    parts = [led]
    if inp.snaps is not None:
        sn = (inp.snaps.select("game_id", "team", player_id="gsis_id").join(inp.games.select("game_id", "team", "season", "week"), on=["game_id", "team"], how="inner"))
        parts.append(L.attach_position(sn, "player_id", inp.positions, extra=()).select("game_id", "team", "player_id", "position"))
    pop = pl.concat(parts).unique(["game_id", "team", "player_id"], keep="first", maintain_order=True)
    if inp.depth_chart is not None:
        dc = _depth_chart_games(inp).select("game_id", "team", "player_id", dc_family="family")
        pop = pop.join(dc, on=["game_id", "team", "player_id"], how="left")
    else:
        pop = pop.with_columns(dc_family=pl.lit(None, dtype=pl.String))
    return pop.filter(pl.col("position").is_in(list(FAMILY_CODE)) | pl.col("dc_family").is_in(list(FAMILY_CODE))).sort("game_id", "team", "player_id")


def _depth_chart_games(inp: Inputs) -> pl.DataFrame:
    """The pregame depth-chart players of every team-game: (game_id, team, player_id, family)."""
    return (inp.depth_chart.rename({"gsis_id": "player_id"})
            .join(inp.games.select("game_id", "team", "season", "week"), on=["team", "season", "week"], how="inner")
            .select("game_id", "team", "player_id", "family"))


TARGET_UNITS = {"QB": ("qb_archetype",), "RB": ("rb_archetype", "receiver_archetype"), "WR": ("receiver_archetype",), "TE": ("receiver_archetype",)}


def archetype_rows(inp: Inputs) -> pl.DataFrame:
    """(game_id, team, player_id, unit, in_pool, is_target, is_reference) for every player-game an archetype vector is needed for.

    in_pool: the player played (snap or touch) at one of the unit's positions -- a comparable observation.  is_target: a scored player-game (2020-2024),
    whose vector must exist whether or not he played, since whether he played is only known after the game.  is_reference: on that week's pregame depth
    chart at one of the unit's positions -- the population the week's league mean and SD come from, so no z-score depends on who played that week."""
    parts = []
    pop = population_table(inp)
    for unit, positions in ARCHETYPE_POSITIONS.items():
        member = pl.col("position").is_in(list(positions)) | pl.col("dc_family").is_in(list(positions))
        parts.append(pop.filter(member).select("game_id", "team", "player_id", unit=pl.lit(unit), role=pl.lit("pool")))
    if inp.targets is not None:
        for fam, units in TARGET_UNITS.items():
            for unit in units:
                parts.append(inp.targets.filter(pl.col("family") == fam).select("game_id", "team", "player_id", unit=pl.lit(unit), role=pl.lit("target")))
    if inp.depth_chart is not None:
        dc = _depth_chart_games(inp)
        for unit, positions in ARCHETYPE_POSITIONS.items():
            parts.append(dc.filter(pl.col("family").is_in(list(positions))).select("game_id", "team", "player_id", unit=pl.lit(unit), role=pl.lit("reference")))
    rows = pl.concat(parts)
    return (rows.group_by("game_id", "team", "player_id", "unit").agg(in_pool=(pl.col("role") == "pool").any(), is_target=(pl.col("role") == "target").any(),
                                                                       is_reference=(pl.col("role") == "reference").any())
            .sort("unit", "game_id", "team", "player_id"))


def lineup_queries(inp: Inputs) -> pl.DataFrame:
    """The (game, team, player) pairs of a game's lineup correction: every player-game in the ledger and every player who played (snap counts); the
    players of a team's previous six games who did not play (absent regulars: their absence is what the lineup correction measures); and the
    players the 4a layer lists for the game. The normal usage shares are normalised over exactly these players."""
    led = inp.player.select("game_id", "team", "player_id")
    seq = inp.games.sort("team", "gameday").with_columns(seq=pl.int_range(pl.len()).over("team")).select("game_id", "team", "seq")
    lseq = led.join(seq, on=["game_id", "team"], how="inner")
    regs = pl.concat([lseq.select("team", "player_id", seq=pl.col("seq") + o) for o in range(1, 7)]).unique()
    absent = regs.join(seq, on=["team", "seq"], how="inner").select("game_id", "team", "player_id")
    parts = [led, absent, population_table(inp).select("game_id", "team", "player_id")]
    if inp.detail is not None:
        parts.append(inp.detail.filter(pl.col("player_id") != "rest").select("game_id", "team", "player_id"))
    return pl.concat(parts).unique()


def build_queries(inp: Inputs) -> pl.DataFrame:
    """The (game, team, player) pairs whose window values are needed: the lineup-correction players (lineup_queries) and every row an archetype vector
    is stored or standardised for (archetype_rows: pool, scored targets, pregame depth chart)."""
    parts = [lineup_queries(inp), archetype_rows(inp).select("game_id", "team", "player_id")]
    q = pl.concat(parts).unique().join(inp.games.select("game_id", "team", "season", "week"), on=["game_id", "team"], how="inner")
    q = L_attach(q, inp)
    return q.sort("player_id", "season", "week", "game_id", "team")      # adjacent per player: build_player_windows walks them in runs


def subset_windows(pw: PlayerWindows, mask: np.ndarray) -> PlayerWindows:
    return PlayerWindows(pw.queries.filter(pl.Series(mask)), pw.keys, pw.rate[:, mask], pw.den[:, mask], pw.variants)


def L_attach(q: pl.DataFrame, inp: Inputs) -> pl.DataFrame:
    from features import comps_ledger as L
    return L.attach_position(q, "player_id", inp.positions, extra=())


def share_ledger(inp: Inputs) -> pl.DataFrame:
    """Per (player, team) one row for every game the team played from his first to his last game with it, with his carries / targets / dropbacks (0 when
    he did not play) over the team's totals: a pooled window value of it is his usage share of the team's plays, absences included, and the shares of a
    team's players add up to about 1. These are the 'normal' shares of the lineup correction."""
    span = inp.player.group_by("player_id", "team").agg(first=pl.col("gameday").min(), last=pl.col("gameday").max())
    g = inp.games.select("game_id", "team", "season", "week", "gameday", "team_game_num")
    tot = inp.team.select("game_id", "team", tc="x.carries", tt="x.targets", td="x.dropbacks")
    rows = span.join(g, on="team", how="inner").filter((pl.col("gameday") >= pl.col("first")) & (pl.col("gameday") <= pl.col("last"))).drop("first", "last")
    elsewhere = inp.player.select("player_id", "season", "week", other="team").unique()
    rows = (rows.join(elsewhere, on=["player_id", "season", "week"], how="left").filter(pl.col("other").is_null() | (pl.col("other") == pl.col("team"))).drop("other"))
    rows = (rows.join(inp.player.select("game_id", "team", "player_id", "c.carries", "c.targets", "c.dropbacks"), on=["game_id", "team", "player_id"], how="left")
            .join(tot, on=["game_id", "team"], how="left").with_columns(pl.col(c).fill_null(0.0) for c in ("c.carries", "c.targets", "c.dropbacks", "tc", "tt", "td")))
    return rows.select("player_id", "team", "game_id", "season", "week", "gameday", "team_game_num",
                       pl.col("c.carries").alias("u.carry_share|n"), pl.col("tc").alias("u.carry_share|d"),
                       pl.col("c.targets").alias("u.target_share|n"), pl.col("tt").alias("u.target_share|d"),
                       pl.col("c.dropbacks").alias("u.dropback_share|n"), pl.col("td").alias("u.dropback_share|d")).sort("player_id", "gameday", "team", "game_id")


def build_player_windows(inp: Inputs, queries: pl.DataFrame, keys: list, variants=VARIANTS, ledger: pl.DataFrame | None = None) -> PlayerWindows:
    """Window values of every player in `queries` as of that query's game, for every variant, from the player's own earlier games (rows of `ledger`,
    default the player ledger)."""
    led = (inp.player if ledger is None else ledger).sort("player_id", "gameday", "team", "game_id")      # a total order: ties would make "last k games" depend on row order
    cut = week_cutoffs(inp.games)
    teams = _team_codes(led["team"], queries["team"])
    lin_tbl = inp.lineups

    def context(df: pl.DataFrame, id_col: str, with_day: str):
        j = (df.join(lin_tbl, on=["team", "season", "week"], how="left", maintain_order="left")
             .join(inp.slots.rename({"gsis_id": id_col, "family": "fam_s"}), on=[id_col, "season", "week"], how="left", maintain_order="left"))
        lin = np.vstack([j[c].fill_null(-1).to_numpy().astype(np.int64) for c in LINEUP_COMPONENTS])
        slot = j["slot"].fill_null(0).to_numpy().astype(np.int64)
        fam = j["fam_s"].replace_strict(FAMILY_CODE, default=0).to_numpy().astype(np.int64) if j["fam_s"].null_count() < j.height else np.zeros(j.height, dtype=np.int64)
        return lin, slot, fam

    hl, hs, hf = context(led.select("player_id", "team", "season", "week"), "player_id", "gameday")
    h_day = _epoch_days(led["gameday"])
    h_clk = game_clock(led["season"].to_numpy(), led["team_game_num"].to_numpy())
    h_team = np.array([teams[t] for t in led["team"].to_list()], dtype=np.int64)
    h_season = led["season"].to_numpy()
    num = led.select([f"{k}|n" for k in keys]).to_numpy().astype(np.float64)
    den = led.select([f"{k}|d" for k in keys]).to_numpy().astype(np.float64)

    qdf = queries.join(cut, on=["season", "week"], how="left", maintain_order="left").join(
        inp.games.select("game_id", "team", "team_game_num"), on=["game_id", "team"], how="left", maintain_order="left")       # rows stay aligned with `queries`
    ql, qs, qf = context(qdf.select("player_id", "team", "season", "week"), "player_id", "cutoff")
    q_day = _epoch_days(qdf["cutoff"])
    q_clk = game_clock(qdf["season"].to_numpy(), qdf["team_game_num"].to_numpy())
    q_team = np.array([teams[t] for t in qdf["team"].to_list()], dtype=np.int64)
    q_season = qdf["season"].to_numpy()

    hp = led["player_id"].to_numpy()
    qp = qdf["player_id"].to_numpy()
    rate = np.full((len(variants), len(qdf), len(keys)), np.nan, dtype=np.float32)
    dsum = np.zeros((len(variants), len(qdf), len(keys)), dtype=np.float32)
    h_bounds = {p: (a, b) for p, a, b in _runs(hp)}
    for p, qa, qb in _runs_sorted(qp):
        if p not in h_bounds:
            continue
        a, b = h_bounds[p]
        hist = Timeline(h_day[a:b], h_clk[a:b], h_season[a:b], hl[:, a:b], hs[a:b], hf[a:b], h_team[a:b])
        query = Timeline(q_day[qa:qb], q_clk[qa:qb], q_season[qa:qb], ql[:, qa:qb], qs[qa:qb], qf[qa:qb], q_team[qa:qb])
        for vi, v in enumerate(variants):
            r, d = pooled(window_weights(hist, query, v), num[a:b], den[a:b])
            rate[vi, qa:qb], dsum[vi, qa:qb] = r, d
    return PlayerWindows(qdf.drop("cutoff", "team_game_num"), list(keys), rate, dsum, tuple(variants))


def _runs(arr: np.ndarray):
    """(value, start, stop) runs of a sorted-by-value array (assumes equal values are adjacent)."""
    if len(arr) == 0:
        return
    cuts = np.flatnonzero(np.r_[True, arr[1:] != arr[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        yield arr[a], a, b


def _runs_sorted(arr: np.ndarray):
    yield from _runs(arr)


# ---------------------------------------------------------------------------------------------------------------------------------- lineup corrections
STRUCT_KEYS = (["receiver_usage.top3_target_share"] + [f"receiver_usage.target_split_{g}" for g in ("wr", "te", "rb")]
               + [f"receiver_usage.adot_{g}" for g in ("wr", "te", "rb")]
               + ["rb_rotation.rb1_carry_share", "rb_rotation.rb2_carry_share", "rb_rotation.rb_target_share", "rb_rotation.rb1_snap_share",
                  "rb_rotation.goal_line_carry_share"])
DELTA_KEYS = RUN_KEYS + PASS_KEYS + STRUCT_KEYS
SCHEME_KEYS = ["run_offense.rush_rate_over_expected", "pass_offense.play_action_rate", "pass_offense.screen_rate", "pass_offense.motion_rate",
               "pass_offense.no_huddle_rate"]
LINEUP_NOT_ADJUSTED = (SCHEME_KEYS + [k for u in ("ol_protection",) + cs.DEFENSE_UNITS for k in feature_keys(u)])
BOUNDED_01 = {k for u in LINEUP_UNITS for k in feature_keys(u)
              if (k.endswith("_rate") or k.endswith("_share")) and not k.endswith("rush_rate_over_expected")}


def struct_values(group: np.ndarray, pid: np.ndarray, s_carry: np.ndarray, s_target: np.ndarray, v_adot: np.ndarray, v_gl: np.ndarray, v_snap: np.ndarray) -> dict:
    """The share-structure features of ONE team-game from a lineup's carry / target shares: top-3 target share, target split by position group,
    target-weighted depth of target by group, and the running-back rotation (RB1 / RB2 share of the RBs' carries, RBs' target share, RB1's snap share
    and goal-line share). NaN when the lineup has nothing to measure it from."""
    out = {}
    st = np.where(np.isfinite(s_target), s_target, 0.0)
    sc = np.where(np.isfinite(s_carry), s_carry, 0.0)
    out["receiver_usage.top3_target_share"] = float(np.sort(st)[::-1][:3].sum()) if st.sum() > 0 else np.nan
    for g in ("wr", "te", "rb"):
        m = group == g
        out[f"receiver_usage.target_split_{g}"] = float(st[m].sum()) if st.sum() > 0 else np.nan
        w = st[m]
        ok = np.isfinite(v_adot[m]) & (w > 0)
        out[f"receiver_usage.adot_{g}"] = float((w[ok] * v_adot[m][ok]).sum() / w[ok].sum()) if ok.any() else np.nan
    rb = np.flatnonzero((group == "rb") & (sc > 0))
    out["rb_rotation.rb_target_share"] = out["receiver_usage.target_split_rb"]
    nan = {"rb_rotation.rb1_carry_share": np.nan, "rb_rotation.rb2_carry_share": np.nan, "rb_rotation.rb1_snap_share": np.nan, "rb_rotation.goal_line_carry_share": np.nan}
    if len(rb):
        order = rb[np.lexsort((pid[rb], -sc[rb]))]          # most carries first, ties by player id
        tot = sc[rb].sum()
        nan["rb_rotation.rb1_carry_share"] = float(sc[order[0]] / tot)
        nan["rb_rotation.rb2_carry_share"] = float(sc[order[1]] / tot) if len(order) > 1 else 0.0
        nan["rb_rotation.rb1_snap_share"] = float(v_snap[order[0]])
        nan["rb_rotation.goal_line_carry_share"] = float(v_gl[order[0]])
    out.update(nan)
    return out


def _gid(df: pl.DataFrame) -> np.ndarray:
    return df.select(pl.struct("game_id", "team").rank("dense").alias("g"))["g"].to_numpy().astype(np.int64) - 1


def share_tables(inp: Inputs, queries: pl.DataFrame) -> pl.DataFrame:
    """The query rows with `active` (the player played: an offensive snap in snap_counts, or a carry / target / dropback in the game) and, where the
    4a layer has the game, the expected (exp_*) and baseline (b_*) shares; 'rest' is the 4a bucket for unlisted players, valued like the team (no
    correction)."""
    played = inp.player.select("game_id", "team", "player_id").with_columns(active=pl.lit(True))
    if inp.snaps is not None:
        played = pl.concat([played, inp.snaps.select("game_id", "team", pl.col("gsis_id").alias("player_id")).with_columns(active=pl.lit(True))]).unique()
    q = queries.join(played, on=["game_id", "team", "player_id"], how="left", maintain_order="left").with_columns(pl.col("active").fill_null(False))
    zero = ("exp_carry", "exp_target", "exp_dropback", "b_carry", "b_target", "b_dropback")
    if inp.detail is not None:
        d = inp.detail.filter(pl.col("player_id") != "rest").select("game_id", "team", "player_id", *zero)
        has = inp.detail.select("game_id", "team").unique().with_columns(has_detail=pl.lit(True))
        q = (q.join(d, on=["game_id", "team", "player_id"], how="left", maintain_order="left").join(has, on=["game_id", "team"], how="left", maintain_order="left")
             .with_columns(pl.col("has_detail").fill_null(False)))
        return q.with_columns(*[pl.col(c).fill_null(0.0) for c in zero])
    return q.with_columns(has_detail=pl.lit(False), **{c: pl.lit(0.0) for c in zero})


def lineup_deltas(pw: PlayerWindows, pw_share: PlayerWindows, shares: pl.DataFrame, tw: dict, team_keys: list, games_index: dict, n_games: int) -> dict:
    """{variant: {'adjusted': (n_games, len(DELTA_KEYS)), 'actual': (...)}} additive corrections to the team's own window values (rows of the games
    table; NaN = no correction available, which the caller treats as none).

    Correction for a feature = sum over players of (share in this lineup - normal share) x (his shrunk window value - the team's value), with
    his window value shrunk toward the team's by SHRINK_K opportunities; for the share-structure features the structure is evaluated on the lineup's
    shares and on the normal shares and the difference taken. 'adjusted' uses the 4a expected vs baseline shares (games the 4a layer covers);
    'actual' each player's normal usage share restricted to the players who played (snap counts) and renormalised, vs the full normal shares, for every game."""
    K = cs.SHRINK_K
    kidx = {k: i for i, k in enumerate(pw.keys)}
    tidx = {k: i for i, k in enumerate(team_keys)}
    gi = np.array([games_index[(g, t)] for g, t in zip(shares["game_id"].to_list(), shares["team"].to_list())], dtype=np.int64)
    gid = _gid(shares)
    n_groups = int(gid.max()) + 1 if len(gid) else 0
    order = np.argsort(gid, kind="stable")
    bounds = np.flatnonzero(np.r_[True, gid[order][1:] != gid[order][:-1], True])
    groups_idx = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    group_games = np.array([gi[r[0]] for r in groups_idx], dtype=np.int64)
    group_has_detail = np.array([bool(shares["has_detail"][int(r[0])]) for r in groups_idx])
    grp = shares["group"].fill_null("").to_numpy()
    pid = shares["player_id"].to_numpy()
    active = shares["active"].to_numpy().astype(float)
    s_exp = {k: shares[f"exp_{k}"].to_numpy() for k in ("carry", "target", "dropback")}
    s_base = {k: shares[f"b_{k}"].to_numpy() for k in ("carry", "target", "dropback")}
    off = len(RUN_KEYS) + len(PASS_KEYS)
    out = {}
    for vi, v in enumerate(pw.variants):
        rate, den = pw.rate[vi].astype(np.float64), pw.den[vi].astype(np.float64)
        rr = tw[v][0]
        mu_team = lambda key: rr[:, tidx[key]][gi]
        norm = {}
        sidx = {k: i for i, k in enumerate(pw_share.keys)}
        for k, key in SHARE_KEYS.items():                              # normal usage shares, rescaled to sum to 1 over the players queried for the game
            x = np.nan_to_num(pw_share.rate[vi][:, sidx[key]].astype(np.float64), nan=0.0)
            tot = np.bincount(gid, weights=x, minlength=n_groups)
            norm[k] = np.divide(x, tot[gid], out=np.zeros_like(x), where=tot[gid] > 0)

        def shrunk(key, mu_key, k=K):
            j = kidx[key]
            mu = mu_team(mu_key)
            val = (np.nan_to_num(rate[:, j], nan=0.0) * den[:, j] + k * mu) / np.maximum(den[:, j] + k, 1e-9)
            return np.where(np.isfinite(val) & (den[:, j] > 0), val, mu)

        s_actual = {}                                                  # who played: the normal usage shares of the players who played, renormalised
        for k in norm:
            x = norm[k] * active
            tot = np.bincount(gid, weights=x, minlength=n_groups)
            s_actual[k] = np.where(tot[gid] > 0, np.divide(x, tot[gid], out=np.zeros_like(x), where=tot[gid] > 0), norm[k])      # nobody with usage played: no correction
        res = {ver: np.full((n_games, len(DELTA_KEYS)), np.nan) for ver in ("adjusted", "actual")}
        for j, key in enumerate(RUN_KEYS + PASS_KEYS):
            share = "carry" if key in RUN_KEYS else "dropback"
            vp, mu = shrunk(key, key), mu_team(key)
            dev = np.where(np.isfinite(vp - mu), vp - mu, 0.0)
            for ver, (s, n) in (("adjusted", (s_exp[share], s_base[share])), ("actual", (s_actual[share], norm[share]))):
                contrib = np.bincount(gid, weights=(s - n) * dev, minlength=n_groups)
                ok = group_has_detail if ver == "adjusted" else np.ones(n_groups, dtype=bool)
                res[ver][group_games[ok], j] = contrib[ok]
        v_adot = {g: shrunk("receiver_archetype.average_depth_of_target", f"receiver_usage.adot_{g}") for g in ("wr", "te", "rb")}
        v_gl = shrunk("rb_archetype.goal_line_share", "rb_rotation.goal_line_carry_share")
        v_snap = shrunk("x.snap", "rb_rotation.rb1_snap_share", k=0.0)       # a per-game mean: no shrinkage
        for g, rows in enumerate(groups_idx):
            adot = np.full(len(rows), np.nan)
            for gname in ("wr", "te", "rb"):
                m = grp[rows] == gname
                adot[m] = v_adot[gname][rows][m]
            for ver, (s, n) in (("adjusted", (s_exp, s_base)), ("actual", (s_actual, norm))):
                if ver == "adjusted" and not group_has_detail[g]:
                    continue
                a = struct_values(grp[rows], pid[rows], s["carry"][rows], s["target"][rows], adot, v_gl[rows], v_snap[rows])
                b = struct_values(grp[rows], pid[rows], n["carry"][rows], n["target"][rows], adot, v_gl[rows], v_snap[rows])
                for j, key in enumerate(STRUCT_KEYS):
                    d = a[key] - b[key]
                    res[ver][group_games[g], off + j] = d if np.isfinite(d) else 0.0
        out[v] = res
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- assembling vectors
def _pack(meta: pl.DataFrame, z: np.ndarray, raw: np.ndarray, n: np.ndarray, **const) -> pl.DataFrame:
    """Rows of vectors: z / raw / n as lists of Float32 (a missing feature is null, never 0)."""
    def lst(a):
        s = pl.Series(a.astype(np.float32)).cast(pl.List(pl.Float32)) if a.shape[1] else pl.Series([[] for _ in range(a.shape[0])], dtype=pl.List(pl.Float32))
        return s.list.eval(pl.element().fill_nan(None))
    present = (~np.isnan(z)).sum(axis=1)
    return meta.with_columns(**{k: pl.lit(v) for k, v in const.items()}, z=lst(z), raw=lst(raw), n=lst(n),
                             n_present=pl.Series(present.astype(np.int16)), complete=pl.Series(present == z.shape[1]))


def feature_meta() -> pl.DataFrame:
    """One row per (unit, space, position in the vector): the feature, its source table(s), quality tag and first season. The vector rows are
    keyed to this table by (unit, space) and the position of the value in their lists."""
    rows = []
    for unit in cs.UNITS:
        for space in stored_spaces(unit):
            feats = [f for f in cs.FEATURES[unit] if space == "extended" or f.space == "base"]
            for i, f in enumerate(feats):
                rows.append(dict(unit=unit, space=space, idx=i, feature=f.name, source=f.source, columns=", ".join(f.columns), quality=f.quality,
                                 feature_space=f.space, first_season=f.first_season, definition=f.note))
    return pl.DataFrame(rows)


def team_vector_frames(inp: Inputs, tw: dict, deltas: dict | None, team_keys: list, variants=VARIANTS, units=TEAM_UNITS) -> list:
    """Vector rows for every team-game, unit, window, space and lineup version ('healthy' always; 'adjusted' / 'actual' for the LINEUP_UNITS)."""
    g = inp.games
    cut = g.join(week_cutoffs(g), on=["season", "week"], how="left", maintain_order="left")["cutoff"]
    meta = pl.DataFrame({"game_id": g["game_id"], "team": g["team"], "player_id": pl.Series([None] * g.height, dtype=pl.String), "season": g["season"],
                         "week": g["week"], "as_of": cut})
    wk = (g["season"].to_numpy() * 100 + g["week"].to_numpy()).astype(np.int64)
    kidx = {k: i for i, k in enumerate(team_keys)}
    dcol = {k: i for i, k in enumerate(DELTA_KEYS)}
    out = []
    for v in variants:
        rate, den = tw[v]
        z, mu, sd = league_z(rate, den, wk)
        versions = {"healthy": (rate, np.ones(g.height, dtype=bool))}
        if deltas is not None:
            for ver in ("adjusted", "actual"):
                d = deltas[v][ver]
                raw = rate.copy()
                have = ~np.isnan(d).all(axis=1)
                for k, j in dcol.items():
                    c = kidx[k]
                    adj = rate[:, c] + np.nan_to_num(d[:, j], nan=0.0)
                    raw[:, c] = np.clip(adj, 0.0, 1.0) if k in BOUNDED_01 else adj
                versions[ver] = (raw, have)
        for ver, (raw, rows_ok) in versions.items():
            zz = z if ver == "healthy" else np.clip((raw - mu) / sd, -Z_CLIP, Z_CLIP)
            for unit in (units if ver == "healthy" else [u for u in units if u in LINEUP_UNITS]):
                for space in stored_spaces(unit):
                    cols = [kidx[k] for k in feature_keys(unit, space)]
                    sel = np.flatnonzero(rows_ok)
                    out.append(_pack(meta[sel], zz[sel][:, cols], raw[sel][:, cols], den[sel][:, cols], version=ver, window=vname(v), unit=unit, space=space))
    return out


def player_vector_frames(inp: Inputs, pw: PlayerWindows, variants=VARIANTS) -> list:
    """Archetype vectors for every player-game in the pool (he played at one of the unit's positions) or scored as a target (he gets a vector whether or not
    he played), with in_pool / is_target flags. The shrinkage mean and the league mean / SD of a week come from that week's PREGAME depth-chart players
    (is_reference) only, so no vector depends on who turned out to play that week. Without a depth chart (tests) the stored rows stand in for it."""
    K = cs.SHRINK_K
    q = pw.queries.with_row_index("qi").select("qi", "game_id", "team", "player_id", "season", "week")
    cut = week_cutoffs(inp.games)
    rows = archetype_rows(inp).join(q, on=["game_id", "team", "player_id"], how="inner").join(cut, on=["season", "week"], how="left")
    kidx = {k: i for i, k in enumerate(pw.keys)}
    out = []
    for unit in ARCHETYPE_POSITIONS:
        u = rows.filter(pl.col("unit") == unit)
        sto = u.filter(pl.col("in_pool") | pl.col("is_target")).sort("season", "week", "game_id", "team", "player_id")
        ref = u.filter(pl.col("is_reference")).sort("season", "week", "game_id", "team", "player_id")      # a fixed summation order
        if ref.height == 0:
            ref = sto
        qs, qr = sto["qi"].to_numpy(), ref["qi"].to_numpy()
        meta = sto.select("game_id", "team", "player_id", "season", "week", "in_pool", "is_target", as_of="cutoff")
        ks = (sto["season"].to_numpy() * 100 + sto["week"].to_numpy()).astype(np.int64)
        kr = (ref["season"].to_numpy() * 100 + ref["week"].to_numpy()).astype(np.int64)
        for space in stored_spaces(unit):
            keys = feature_keys(unit, space)
            cols = [kidx[k] for k in keys]
            static = np.array([k.split(".")[1] in NO_SHRINK for k in keys])
            for vi, v in enumerate(variants):
                def take(qi):
                    r, d = pw.rate[vi][qi][:, cols].astype(np.float64), pw.den[vi][qi][:, cols].astype(np.float64)
                    return r, d, np.nan_to_num(r) * d
                rr, dr, nr = take(qr)
                rs, ds, ns = take(qs)
                mu_week = {}                                                  # the reference players' pooled mean of the week (earlier games only)
                for w in np.unique(kr):
                    m = kr == w
                    tot = dr[m].sum(axis=0)
                    mu_week[int(w)] = np.divide(nr[m].sum(axis=0), tot, out=np.full(tot.shape, np.nan), where=tot > 0)
                nan = np.full(len(cols), np.nan)
                def shrink(r, d, n, kk):
                    mu = np.vstack([mu_week.get(int(k), nan) for k in kk]) if len(kk) else np.zeros((0, len(cols)))
                    val = np.where(static[None, :], r, (n + K * mu) / (d + K))
                    return np.where(np.isfinite(val) & (d > 0), val, np.nan)
                vr, vs = shrink(rr, dr, nr, kr), shrink(rs, ds, ns, ks)
                z = apply_week_stats(vs, ks, week_stats(vr, dr, kr))[0]
                z = np.where(ds > 0, z, np.nan)
                out.append(_pack(meta, z, vs, ds, version="player", window=vname(v), unit=unit, space=space))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- which version a search reads
def target_version(unit: str) -> str:
    """Every comparable search runs on the LINEUP-ADJUSTED target vector; a unit with no lineup correction has one version, 'healthy'."""
    return "adjusted" if unit in LINEUP_UNITS else "healthy"


def pool_version(unit: str) -> str:
    """Historical pool vectors reflect who actually played that game ('actual'); units without a lineup correction use 'healthy'."""
    return "actual" if unit in LINEUP_UNITS else "healthy"


def retrieval_change(healthy_hits: list, adjusted_hits: list, k: int | None = None) -> dict:
    """How the retrieved comparables change when the search runs on the lineup-adjusted target vector instead of the healthy one.

    Each hit is (comparable_id, weight, outcome). Returns the overlap of the top-k ids (share of the k ids in both; k defaults to the smaller list),
    the shared weight mass (sum over ids of min(normalized weight)) and the shift in the weighted mean outcome (adjusted - healthy). 4c.2-4c.5
    call this for every search and store the result next to the retrieval."""
    def norm(hits):
        w = np.array([h[1] for h in hits], dtype=float)
        return (w / w.sum()) if len(hits) and w.sum() > 0 else w
    k = k or min(len(healthy_hits), len(adjusted_hits))
    top = lambda hits: [h[0] for h in sorted(hits, key=lambda h: (-h[1], str(h[0])))[:k]]
    a, b = set(top(healthy_hits)), set(top(adjusted_hits))
    wa, wb = dict(zip([h[0] for h in healthy_hits], norm(healthy_hits))), dict(zip([h[0] for h in adjusted_hits], norm(adjusted_hits)))
    mean = lambda hits, w: float(sum(w[h[0]] * h[2] for h in hits)) if hits else float("nan")
    shared = sum(min(wa.get(i, 0.0), wb.get(i, 0.0)) for i in sorted(set(wa) | set(wb)))      # sorted: a set's order changes with the hash seed
    return dict(k=k, overlap_top_k=len(a & b) / k if k else float("nan"), weight_mass_shared=float(shared),
                outcome_shift=mean(adjusted_hits, wb) - mean(healthy_hits, wa))


# ---------------------------------------------------------------------------------------------------------------------------------- driver
def load_lineup_expectations(raw_db=config.RAW_DUCKDB_PATH, max_season=None, cache_dir=None) -> pl.DataFrame:
    """The 4a injury layer's expected game-day shares for every team-game of the backtest seasons (sim.inputs._detail_and_exit: 4a.1 play
    probability, 4a.2 redistributed carry / target / dropback shares, fit week by week on earlier games only)."""
    from eval import backtest as bt
    from sim import inputs as si
    cap = config.cap_season(max_season)
    data = bt.load_backtest_data(raw_db)
    detail, _, _ = si._detail_and_exit(data, raw_db, cap, cache_dir)
    return detail


def load_inputs(raw_db=config.RAW_DUCKDB_PATH, max_season=None, lineup_expectations: bool = True, cache_dir=None) -> Inputs:
    from features import comps_ledger as L
    from features import lineups as LU
    games = L.load_games(raw_db, max_season)
    plays = L.load_plays(raw_db, max_season)
    pos = L.load_positions(raw_db, max_season)
    snaps = L.load_snaps(raw_db, max_season)
    pfr_pass, pfr_rush = L.load_pfr(raw_db, max_season)
    team = L.team_ledger(plays, games, pfr_pass)
    player = L.player_ledger(plays, team, pos, snaps, pfr_rush, games)
    wide = (team.join(L.structure_ledger(player, team), on=["game_id", "team"], how="left")
            .join(L.coverage_position_ledger(plays, pos), on=["game_id", "team"], how="left"))
    wide = wide.with_columns(pl.col(c).fill_null(0.0) for c in wide.columns if c.endswith("|n") or c.endswith("|d"))
    lineups = LU.build_lineups(raw_db, max_season, start=cs.BASE_START).with_columns(team=L.canon("team"))
    detail = load_lineup_expectations(raw_db, max_season, cache_dir) if lineup_expectations else None
    return Inputs(games=games, lineups=lineups, team=wide, player=player, slots=L.load_slots(raw_db, max_season), positions=pos, detail=detail,
                  snaps=snaps, depth_chart=L.load_depth_chart_players(raw_db, max_season), targets=scored_targets(raw_db),
                  completeness=L.completeness(plays, games, team, pfr_pass, snaps))


def scored_targets(raw_db=config.RAW_DUCKDB_PATH) -> pl.DataFrame:
    """(game_id, team, player_id, family) of every scored player-game of the 2020-2024 backtest (the eligibility rule decides, from pregame information)."""
    from eval import backtest as bt
    data = bt.load_backtest_data(raw_db)
    return data.player_log.filter(pl.col("elig") != "").select("game_id", "team", "player_id", "family").unique().sort("game_id", "team", "player_id")


def lineup_change_log(deltas: dict, tw: dict, team_keys: list, games: pl.DataFrame, variants=VARIANTS) -> pl.DataFrame:
    """How far the lineup versions move the vectors: per version, window and lineup-corrected feature, the share of team-games moved by more than
    0.1 league SD and the mean / 95th percentile absolute move in league SDs."""
    kidx = {k: i for i, k in enumerate(team_keys)}
    wk = (games["season"].to_numpy() * 100 + games["week"].to_numpy()).astype(np.int64)
    rows = []
    for v in variants:
        rate, den = tw[v]
        _, _, sd = league_z(rate, den, wk)
        for ver in ("adjusted", "actual"):
            d = deltas[v][ver]
            for j, key in enumerate(DELTA_KEYS):
                x = np.abs(d[:, j] / sd[:, kidx[key]])
                x = x[np.isfinite(x)]
                if x.size:
                    rows.append(dict(version=ver, window=vname(v), feature=key, n_games=int(x.size), share_moved_gt_0p1sd=float((x > 0.1).mean()),
                                     mean_abs_move_sd=float(x.mean()), p95_abs_move_sd=float(np.quantile(x, 0.95))))
    return pl.DataFrame(rows)


@dataclass
class Vectors:
    team: pl.DataFrame
    player: pl.DataFrame
    features: pl.DataFrame
    completeness: pl.DataFrame
    lineup_change: pl.DataFrame
    notes: dict = field(default_factory=dict)


def build_vectors(inp: Inputs, variants=VARIANTS) -> Vectors:
    """Everything 4c.1 produces from the loaded inputs (pure function of `inp`)."""
    team_keys = [k for u in TEAM_UNITS for k in feature_keys(u)]
    missing = [k for k in team_keys if f"{k}|n" not in inp.team.columns or f"{k}|d" not in inp.team.columns]
    if missing:
        raise KeyError(f"team ledger lacks spec features: {missing}")
    games_index = {(g, t): i for i, (g, t) in enumerate(zip(inp.games["game_id"].to_list(), inp.games["team"].to_list()))}
    tw = team_windows(inp.games, inp.lineups, inp.team, team_keys, variants)
    queries = build_queries(inp)
    pkeys = player_keys(inp.player)
    pw = build_player_windows(inp, queries, pkeys, variants)
    pw_share = build_player_windows(inp, queries, list(SHARE_KEYS.values()), variants, ledger=share_ledger(inp))
    # the lineup correction normalises the normal shares over its own players only (the depth-chart / target-only rows exist for the archetype vectors)
    lm = pw.queries.select("game_id", "team", "player_id").join(lineup_queries(inp).with_columns(_l=pl.lit(True)), on=["game_id", "team", "player_id"],
                                                                 how="left", maintain_order="left")["_l"].fill_null(False).to_numpy()
    assert pw_share.queries.select("game_id", "team", "player_id").equals(pw.queries.select("game_id", "team", "player_id"))
    pl_, ps_ = subset_windows(pw, lm), subset_windows(pw_share, lm)
    deltas = lineup_deltas(pl_, ps_, share_tables(inp, pl_.queries), tw, team_keys, games_index, inp.games.height)
    sort_keys = ["version", "window", "unit", "space", "season", "week", "game_id", "team", "player_id"]
    team = pl.concat(team_vector_frames(inp, tw, deltas, team_keys, variants)).sort(sort_keys, nulls_last=True)
    player = pl.concat(player_vector_frames(inp, pw, variants)).sort(sort_keys, nulls_last=True)
    return Vectors(team, player, feature_meta(), inp.completeness if inp.completeness is not None else pl.DataFrame(),
                   lineup_change_log(deltas, tw, team_keys, inp.games, variants),
                   notes=dict(lineup_not_adjusted=LINEUP_NOT_ADJUSTED, variants=[vname(v) for v in variants],
                              adjusted_games=int(inp.detail.select("game_id", "team").unique().height) if inp.detail is not None else 0))


def write_vectors(vec: Vectors, out_dir=config.PROCESSED_DIR, log_dir=config.ROOT) -> dict:
    """The vectors go to data/processed (large, regenerable, not committed); the feature table, completeness log and lineup-change log (small)
    go to the repo root next to the other result files."""
    paths = {"team": out_dir / "comp_vectors_team.parquet", "player": out_dir / "comp_vectors_player.parquet"}
    vec.team.write_parquet(paths["team"])
    vec.player.write_parquet(paths["player"])
    for name, df in (("features", vec.features), ("completeness", vec.completeness), ("lineup_change", vec.lineup_change)):
        paths[name] = log_dir / f"comp_{name}.parquet"
        df.write_parquet(paths[name])
    return paths


# ---------------------------------------------------------------------------------------------------------------------------------- 4c.2 distance
@dataclass
class Distance:
    d2: np.ndarray               # (na, nb) unit distance d2: NaN where the two sides share no feature group
    completeness: np.ndarray     # (na, nb) share of the unit's weight mass (w x q) that is missing on either side: the completeness penalty (4c.5)
    groups: np.ndarray           # (na, nb) number of feature groups that could be compared
    quality: np.ndarray = None   # (na, nb) mean q_f of the features compared (weighted by w_f): the data-quality of the match
    shares: np.ndarray = None    # (3, na, nb) share of the distance's effective feature weight that is observed / derived / estimated (4c.5)


TAGS = ("observed", "derived", "estimated")
TAG_CODE = {t: i for i, t in enumerate(TAGS)}


def unit_features(unit: str, space: str) -> list:
    return [f for f in cs.FEATURES[unit] if space == "extended" or f.space == "base"]


def feature_quality(unit: str, space: str, quality: dict | None = None) -> np.ndarray:
    """q_f per feature in the vector order: observed 1.0, derived 0.9, estimated 0.65 (cs.QUALITY). `quality` maps a feature name to an override
    (a test sets one to 0 to show a quality-0 feature changes nothing)."""
    return np.array([(quality or {}).get(f.name, cs.QUALITY[f.quality]) for f in unit_features(unit, space)], dtype=float)


def group_columns(unit: str, space: str) -> list:
    """[(group name, column indices in the vector)] for the groups that have at least one feature in this space."""
    pos = {f.name: i for i, f in enumerate(unit_features(unit, space))}
    return [(name, [pos[f] for f in feats if f in pos]) for name, feats in cs.GROUPS[unit] if any(f in pos for f in feats)]


def group_distance(A: np.ndarray, B: np.ndarray, groups: list, w: np.ndarray, q: np.ndarray, group_w: np.ndarray | None = None,
                   tags: np.ndarray | None = None) -> Distance:
    """The distance of unit_distance for any feature layout: `groups` is a list of column-index lists, w and q per column (NaN = missing).

    With `tags` (TAG_CODE per column) it also returns `shares`: each feature's effective weight in this d2 -- its group's share of the group
    weights times its w x q over the group's features present on both sides -- summed by tag, so the three shares add up to 1 wherever d2 exists."""
    wq = w * q
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    ma, mb = ~np.isnan(A), ~np.isnan(B)
    a0, b0 = np.where(ma, A, 0.0), np.where(mb, B, 0.0)
    shape = (A.shape[0], B.shape[0])
    num_u, den_u, got, mass, mass_w = (np.zeros(shape) for _ in range(5))
    tags = None if tags is None else np.asarray(tags)
    tag_num = np.zeros((len(TAGS),) + shape) if tags is not None else None
    for gi, cols in enumerate(groups):
        c = np.array(cols)
        f = wq[c]
        den = (ma[:, c] * f) @ mb[:, c].T
        num = (a0[:, c] ** 2 * f) @ mb[:, c].T + ma[:, c].astype(float) @ (b0[:, c] ** 2 * f).T - 2.0 * (a0[:, c] * f) @ b0[:, c].T
        with np.errstate(invalid="ignore", divide="ignore"):
            d2g = np.where(den > 1e-12, np.maximum(num, 0.0) / den, np.nan)
        d2g = np.where(d2g < 1e-10, 0.0, d2g)                   # the matrix-product form leaves ~1e-16 for identical vectors: identical means exactly 0
        gw = 1.0 if group_w is None else float(group_w[gi])
        ok = ~np.isnan(d2g)
        num_u += np.where(ok, gw * d2g, 0.0)
        den_u += np.where(ok, gw, 0.0)
        got += ok
        mass += den
        mass_w += (ma[:, c] * w[c]) @ mb[:, c].T
        if tags is not None:
            safe = np.where(den > 1e-12, den, 1.0)
            for t in range(len(TAGS)):
                ct = c[tags[c] == t]
                if len(ct):
                    tag_num[t] += np.where(ok, gw * ((ma[:, ct] * wq[ct]) @ mb[:, ct].T) / safe, 0.0)
    total = float(wq.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        d2 = np.where(den_u > 0, num_u / den_u, np.nan)
        comp = (1.0 - mass / total) if total > 0 else np.full(shape, np.nan)
        qual = np.where(mass_w > 0, mass / np.maximum(mass_w, 1e-12), np.nan)
        shares = np.where(den_u > 0, tag_num / np.where(den_u > 0, den_u, 1.0), np.nan) if tags is not None else None
    return Distance(d2, comp, got.astype(int), qual, shares)


def unit_distance(A: np.ndarray, B: np.ndarray, unit: str, space: str, quality: dict | None = None, weights: dict | None = None,
                  group_weights: dict | None = None, tags: bool = False) -> Distance:
    """d2 between every row of A and every row of B (z-vectors of the same unit, window and space; NaN = missing).

    d2_g = sum_f(w_f q_f (a_f - b_f)^2) / sum_f(w_f q_f) over the features of group g present on BOTH sides (a feature missing on either side has
    weight 0). The unit d2 is the weighted mean of the d2_g over the groups that could be compared (start: equal weights). The completeness penalty is
    the share of the unit's total w x q mass that was missing on either side. Computed with matrix products, so it scales to a large pool."""
    feats = unit_features(unit, space)
    w = np.array([(weights or {}).get(f.name, cs.FEATURE_WEIGHT_DEFAULT) for f in feats], dtype=float)
    groups = group_columns(unit, space)
    gw = np.array([(group_weights or {}).get(name, cs.GROUP_WEIGHT_DEFAULT) for name, _ in groups], dtype=float)
    return group_distance(A, B, [cols for _, cols in groups], w, feature_quality(unit, space, quality), gw,
                          np.array([TAG_CODE[f.quality] for f in feats]) if tags else None)


def similarity(d2, sigma):
    """exp(-d2 / (2 sigma^2)) with d2 the unit distance above (already a squared distance: it is NOT squared again). NaN where d2 or sigma is missing."""
    return np.exp(-np.asarray(d2, dtype=float) / (2.0 * np.asarray(sigma, dtype=float) ** 2))


def choose_space(unit: str, ext_complete_a: bool, ext_complete_b: bool) -> str | None:
    """EXTENDED only when every EXTENDED feature of the unit exists on both sides, else BASE; None for a unit with no BASE features (coverage_mix)."""
    spaces = stored_spaces(unit)
    if "extended" in spaces and ext_complete_a and ext_complete_b:
        return "extended"
    return "base" if "base" in spaces else None


def vector_matrix(df: pl.DataFrame, n_features: int) -> np.ndarray:
    """(n, F) float matrix of the z lists of a vector frame (null -> NaN)."""
    if df.height == 0:
        return np.zeros((0, n_features))
    return df["z"].explode().cast(pl.Float64).fill_null(np.nan).to_numpy().reshape(df.height, n_features)


# ---------------------------------------------------------------------------------------------------------------------------------- sigma
SIGMA_MIN_TARGETS = 30            # a sigma needs at least this many earlier targets, else it is missing (and so is every similarity that needs it)
SIGMA_PLAYER_TARGETS_PER_WEEK = 20


def _week_key(df: pl.DataFrame) -> np.ndarray:
    return (df["season"].to_numpy() * 100 + df["week"].to_numpy()).astype(np.int64)


def sigma_targets(Z: np.ndarray, keys: np.ndarray, ent: np.ndarray, gid: np.ndarray, dist_fn, per_week: int | None = None) -> tuple:
    """(week key of each pseudo-target, 20th-neighbour distance excluding its own entity, the same including it). Rows of Z are sorted by key; the
    neighbours of a target in week k are the rows of earlier weeks (key < k). dist_fn(A, B) -> d2 matrix. A fixed pseudo-random subset of `per_week`
    targets per week when given."""
    import zlib
    ks, ke, ki = [], [], []
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    for a, b in zip(starts, ends):
        if a <= cs.SIGMA_NEIGHBOR_RANK:
            continue
        idx = np.arange(a, b)
        if per_week is not None and len(idx) > per_week:
            order = sorted(idx, key=lambda i: zlib.crc32(f"{ent[i]}|{gid[i]}".encode()))
            idx = np.array(sorted(order[:per_week]))
        D = np.sqrt(dist_fn(Z[idx], Z[:a]))
        D = np.where(np.isnan(D), np.inf, D)
        same = ent[idx][:, None] == ent[:a][None, :]
        k = cs.SIGMA_NEIGHBOR_RANK - 1
        kth_ex = np.partition(np.where(same, np.inf, D), k, axis=1)[:, k]
        kth_in = np.partition(D, k, axis=1)[:, k]
        ks.append(np.full(len(idx), keys[a])); ke.append(kth_ex); ki.append(kth_in)
    if not ks:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    return np.concatenate(ks), np.concatenate(ke), np.concatenate(ki)


def _sigma_block(df: pl.DataFrame, unit: str, space: str, entity: str, per_week: int | None) -> tuple:
    """sigma_targets for one (unit, window, space) vector frame."""
    nf = len(unit_features(unit, space))
    df = df.filter(pl.col("complete") if space == "extended" and any(f.space == "base" for f in cs.FEATURES[unit]) else pl.col("n_present") > 0)
    df = df.with_columns(_k=pl.Series(_week_key(df))).sort("_k", entity, "game_id")
    return sigma_targets(vector_matrix(df, nf), df["_k"].to_numpy(), df[entity].cast(pl.String).to_numpy(), df["game_id"].to_numpy(),
                         lambda A, B: unit_distance(A, B, unit, space).d2, per_week)


def build_sigma(team: pl.DataFrame, player: pl.DataFrame, per_week_players: int | None = SIGMA_PLAYER_TARGETS_PER_WEEK) -> pl.DataFrame:
    """sigma per (unit, window, space) and cutoff week: the median over EARLIER targets of the distance to the target's 20th nearest neighbour.

    Every vector of the unit (a team-game, or a player-game; players: a fixed pseudo-random subset of `per_week_players` per week) is a pseudo-target; its
    neighbours are the unit's vectors of earlier weeks, same window and space, from the pool versions (`pool_version`; players: the in_pool rows), excluding
    the target's own team / player (a team's consecutive games share window data, so they are not independent matches). Distance D = sqrt(d2), so sigma is on the scale that
    similarity = exp(-d2 / (2 sigma^2)) needs. sigma at week K uses targets of weeks before K only. `sigma_incl_own` keeps the own entity, for reference."""
    rows = []
    for unit in cs.UNITS:
        is_team = unit in TEAM_UNITS
        src = team if is_team else player
        for space in stored_spaces(unit):
            for window in sorted(src["window"].unique().to_list()):
                df = src.filter((pl.col("unit") == unit) & (pl.col("space") == space) & (pl.col("window") == window))
                if is_team:
                    df = df.filter(pl.col("version") == pool_version(unit))
                elif "in_pool" in df.columns:
                    df = df.filter(pl.col("in_pool"))           # a scored player-game in which he did not play is a target, never a pool row
                if df.height == 0:
                    continue
                keys, d_ex, d_in = _sigma_block(df, unit, space, "team" if is_team else "player_id", None if is_team else per_week_players)
                cut = np.unique(_week_key(df))
                for k in cut:
                    m = keys < k
                    n = int(m.sum())
                    ok = n >= SIGMA_MIN_TARGETS
                    rows.append(dict(unit=unit, window=window, space=space, season=int(k // 100), week=int(k % 100), n_targets=n,
                                     sigma=float(np.nanmedian(d_ex[m])) if ok and np.isfinite(d_ex[m]).any() else None,
                                     sigma_incl_own=float(np.nanmedian(d_in[m])) if ok and np.isfinite(d_in[m]).any() else None))
    return pl.DataFrame(rows, schema={"unit": pl.String, "window": pl.String, "space": pl.String, "season": pl.Int32, "week": pl.Int32, "n_targets": pl.Int64,
                                      "sigma": pl.Float64, "sigma_incl_own": pl.Float64})


def sigma_at(table: pl.DataFrame, unit: str, window: str, space: str, season: int, week: int) -> float:
    """sigma for a target in (season, week): from targets of earlier weeks only. NaN when missing."""
    r = table.filter((pl.col("unit") == unit) & (pl.col("window") == window) & (pl.col("space") == space) & (pl.col("season") == season) & (pl.col("week") == week))
    return float(r["sigma"][0]) if r.height and r["sigma"][0] is not None else float("nan")


# ================================================================================================================================ 4c.3 searches
def _team_meta(games: pl.DataFrame) -> pl.DataFrame:
    cut = games.join(week_cutoffs(games), on=["season", "week"], how="left", maintain_order="left")["cutoff"]
    return pl.DataFrame({"game_id": games["game_id"], "team": games["team"], "player_id": pl.Series([None] * games.height, dtype=pl.String),
                         "season": games["season"], "week": games["week"], "as_of": cut})


def matchup_vector_frames(games: pl.DataFrame, lineups: pl.DataFrame, extra: pl.DataFrame, variants=VARIANTS) -> pl.DataFrame:
    """Vectors of the S5 results-side statistics (cs.INTERACTION_EXTRA) for every team-game and window: the same windows and league z-scores as the
    units, unit name 'matchup_extra', space 'extended', version 'healthy'."""
    keys = [f"x.{f.name}" for f in cs.INTERACTION_EXTRA]
    tw = team_windows(games, lineups, extra, keys, variants)
    wk = (games["season"].to_numpy() * 100 + games["week"].to_numpy()).astype(np.int64)
    meta = _team_meta(games)
    out = []
    for v in variants:
        rate, den = tw[v]
        z = league_z(rate, den, wk)[0]
        out.append(_pack(meta, z, rate, den, version="healthy", window=vname(v), unit="matchup_extra", space="extended"))
    return pl.concat(out).sort(["window", "season", "week", "game_id", "team"])


@dataclass(frozen=True)
class Target:
    """A thing to predict: a team-side of a game (player_id None) or a player in a game, for a market."""
    market: str
    game_id: str
    team: str
    opponent: str
    season: int
    week: int
    player_id: str | None = None


class _Block:
    """Aligned base / extended matrices of one unit's vectors."""
    def __init__(self, base, ext, ext_complete):
        self.base, self.ext, self.ext_complete = base, ext, ext_complete


@dataclass
class SearchResult:
    search: str
    summary: dict
    matches: pl.DataFrame | None = None


def _list_matrix(df: pl.DataFrame, col: str, n_features: int) -> np.ndarray:
    if df.height == 0:
        return np.zeros((0, n_features))
    return df[col].explode().cast(pl.Float64).fill_null(np.nan).to_numpy().reshape(df.height, n_features)


def cluster_neff(w: np.ndarray, clusters) -> float:
    """n_eff = (sum w)^2 / sum(w^2) over the total weight of each cluster (the historical team-game). Five receivers of one past game are one
    matchup's evidence, not five; with one observation per team-game this is the plain n_eff."""
    w = np.asarray(w, dtype=float)
    if not len(w):
        return 0.0
    inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)[1]
    c = np.bincount(inv, weights=w)
    s2 = float((c ** 2).sum())
    return float(c.sum() ** 2 / s2) if s2 > 0 else 0.0


# Decisions of 2026-10-09 (docs/overnight_plan.md): the searches read the healthy target vectors -- the lineup-adjusted ones double-count a player
# already missing from the window; injuries reach the models through the 4a injury features -- and the adjusted run is the stored comparison.
SEARCH_VERSION = "healthy"
COMPARISON_VERSION = "adjusted"
# A search whose similarity multiplies two factors (S2, S4, S5: offense x defense; S3: archetype x defense faced) is checked against SIM_THRESHOLD on
# their geometric mean, the per-factor scale of the one-factor S1, so one threshold means the same in every search; the weaker factor must also reach
# SIDE_FLOOR (a 0.95 / 0.30 pair does not pass on the average). The match weight stays the product (plan 4c.4.2).
SIDE_FLOOR = 0.40
TWO_FACTOR_SEARCHES = ("S2", "S3", "S4", "S5")


def check_similarity(sname: str, combined: np.ndarray, sim_a: np.ndarray, sim_b: np.ndarray) -> np.ndarray:
    """The similarity the no-match rule reads: one-factor S1 its similarity; a two-factor search sqrt(a x b) where min(a, b) >= SIDE_FLOOR, else NaN
    (never a match). `combined` is a x b for a two-factor search."""
    if sname not in TWO_FACTOR_SEARCHES:
        return np.asarray(combined, dtype=float)
    with np.errstate(invalid="ignore"):
        ok = np.minimum(sim_a, sim_b) >= SIDE_FLOOR
        return np.where(ok, np.sqrt(np.where(ok, combined, 0.0)), np.nan)


class Pool:
    """The comparable pool for one window: every vector, aligned to the games table, with the sigma of each unit and space at each week.

    Walk-forward by construction: a search at week K reads only rows with key < K, and sigma at K is built from targets before K."""

    def __init__(self, team: pl.DataFrame, player: pl.DataFrame, matchup: pl.DataFrame | None, games: pl.DataFrame, lineups: pl.DataFrame,
                 slots: pl.DataFrame, sigma: pl.DataFrame, window: str):
        self.window = window
        self.games = games
        n = self.n = games.height
        self.idx = {(g, t): i for i, (g, t) in enumerate(zip(games["game_id"].to_list(), games["team"].to_list()))}
        self.key = (games["season"].to_numpy() * 100 + games["week"].to_numpy()).astype(np.int64)
        assert (np.diff(self.key) >= 0).all(), "games must be sorted by week"
        self.team = games["team"].to_numpy()
        self._gids = games["game_id"].to_list()
        self.clock = game_clock(games["season"].to_numpy(), games["team_game_num"].to_numpy())
        self.opp_row = np.array([self.idx[(g, o)] for g, o in zip(games["game_id"].to_list(), games["opponent"].to_list())], dtype=np.int64)
        self.lin = _lineup_array(games, lineups)
        self.slots = slots
        self.sig = {(r["unit"], r["space"], r["season"] * 100 + r["week"]): r["sigma"] for r in sigma.filter(pl.col("window") == window).iter_rows(named=True)}
        self.T, self.P, self.N = {}, {}, {}
        gi = games.select("game_id", "team").with_row_index("_r")
        self.target_fallback = {}
        self.TH = {}
        for unit in TEAM_UNITS:
            # a target-version row that does not exist (a game the 4a layer has no expectations for) falls back to the healthy vector
            versions = [target_version(unit)] + (["healthy"] if target_version(unit) != "healthy" else [])
            for versions_, store in ((versions, self.T), ([pool_version(unit)], self.P), (["healthy"], self.TH)):
                blocks = {}
                for space in stored_spaces(unit):
                    nf = len(unit_features(unit, space))
                    M, Nn = np.full((n, nf), np.nan), np.zeros((n, nf))
                    comp = np.zeros(n, dtype=bool)
                    have = np.zeros(n, dtype=bool)
                    for version in versions_:
                        df = team.filter((pl.col("unit") == unit) & (pl.col("space") == space) & (pl.col("window") == window) & (pl.col("version") == version))
                        j = gi.join(df, on=["game_id", "team"], how="inner")
                        if not j.height:
                            continue
                        r = j["_r"].to_numpy()
                        fill = ~have[r]
                        M[r[fill]] = _list_matrix(j, "z", nf)[fill]
                        Nn[r[fill]] = np.nan_to_num(_list_matrix(j, "n", nf))[fill]
                        comp[r[fill]] = j["complete"].to_numpy()[fill]
                        if store is self.T and version == "healthy" and len(versions_) > 1:
                            self.target_fallback[(unit, space)] = int(fill.sum())        # rows served by the fallback (logged, not silent)
                        have[r] = True
                    blocks[space] = (M, comp, Nn)
                store[unit] = blocks
        self.pl = {}
        for unit in cs.ARCHETYPE_UNITS:
            blocks = {}
            order = None
            for space in stored_spaces(unit):
                nf = len(unit_features(unit, space))
                df = player.filter((pl.col("unit") == unit) & (pl.col("space") == space) & (pl.col("window") == window)).sort(
                    "season", "week", "game_id", "team", "player_id")
                if order is None:
                    order = df.select("season", "week", "game_id", "team", "player_id")
                    meta = df.select("season", "week", "game_id", "team", "player_id",
                                     *(["in_pool"] if "in_pool" in df.columns else []))
                else:
                    assert df.select("game_id", "team", "player_id").equals(order.select("game_id", "team", "player_id")), "spaces must hold the same rows"
                blocks[space] = (_list_matrix(df, "z", nf), df["complete"].to_numpy())
            pkey = (meta["season"].to_numpy() * 100 + meta["week"].to_numpy()).astype(np.int64)
            prow = np.array([self.idx[(g, t)] for g, t in zip(meta["game_id"].to_list(), meta["team"].to_list())], dtype=np.int64)
            sl = meta.join(slots.rename({"gsis_id": "player_id"}), on=["player_id", "season", "week"], how="left", maintain_order="left")
            in_pool = meta["in_pool"].to_numpy() if "in_pool" in meta.columns else np.ones(meta.height, dtype=bool)
            self.pl[unit] = dict(blocks=blocks, key=pkey, row=prow, pid=meta["player_id"].to_numpy(), team=meta["team"].to_numpy(), in_pool=in_pool,
                                 slot=sl["slot"].fill_null(0).to_numpy().astype(np.int64),
                                 fam=sl["family"].replace_strict(FAMILY_CODE, default=0).to_numpy().astype(np.int64) if sl["family"].null_count() < sl.height else np.zeros(meta.height, dtype=np.int64))
        self.extra = np.full((n, len(cs.INTERACTION_EXTRA)), np.nan)
        if matchup is not None and matchup.height:
            m = gi.join(matchup.filter(pl.col("window") == window), on=["game_id", "team"], how="inner")
            self.extra[m["_r"].to_numpy()] = _list_matrix(m, "z", len(cs.INTERACTION_EXTRA))
        self._build_profile()
        self._cache = {}

    # ---- S5 profile
    def _feature_col(self, store: dict, key: str) -> tuple:
        if key.startswith("x."):
            return self.extra[:, [f.name for f in cs.INTERACTION_EXTRA].index(key[2:])], None
        unit, feat = key.split(".", 1)
        blocks = store[unit]
        space = "extended" if "extended" in blocks else "base"
        names = [f.name for f in unit_features(unit, space)]
        j = names.index(feat)
        return blocks[space][0][:, j], blocks[space][2][:, j]

    def _build_profile(self):
        """S5 is a two-sided search (build plan 4c.4): the offense side of the interaction profile (the target team's features of every interaction) and
        the defense side (the opponent's) are compared separately, each with its own feature groups (one per interaction), quality weights and sigma."""
        off_keys, def_keys = [], []
        for _, o, d in cs.S5_PROFILE:
            off_keys += [k for k in o if k not in off_keys]
            def_keys += [k for k in d if k not in def_keys]
        self.off_keys, self.def_keys = off_keys, def_keys
        self.prof_off_t = np.column_stack([self._feature_col(self.T, k)[0] for k in off_keys])
        self.prof_off_th = np.column_stack([self._feature_col(self.TH, k)[0] for k in off_keys])      # healthy target version (the comparison run)
        self.prof_off_p = np.column_stack([self._feature_col(self.P, k)[0] for k in off_keys])
        dcols = []
        for k in def_keys:
            v, nn = self._feature_col(self.P, k)
            if nn is not None and k.startswith("run_defense.rush_epa_allowed_"):
                v = np.where(nn >= cs.S5_MIN_PLAYS, v, np.nan)                 # a run-direction result needs 30 plays
            dcols.append(v)
        self.prof_def = np.column_stack(dcols)
        self.groups5 = {"off": [[off_keys.index(k) for k in o] for _, o, _ in cs.S5_PROFILE],
                        "def": [[def_keys.index(k) for k in d] for _, _, d in cs.S5_PROFILE]}
        qmap = {f"{u}.{f.name}": f.quality for u in cs.UNITS for f in cs.FEATURES[u]} | {f"x.{f.name}": f.quality for f in cs.INTERACTION_EXTRA}
        ftn = {f"{u}.{f.name}" for u in cs.UNITS for f in cs.FEATURES[u] if cs.S5_REQUIRED_SOURCE in f.source} | {f"x.{f.name}" for f in cs.INTERACTION_EXTRA}
        self.q5 = {"off": np.array([cs.QUALITY[qmap[k]] for k in off_keys]), "def": np.array([cs.QUALITY[qmap[k]] for k in def_keys])}
        self.tag5 = {"off": np.array([TAG_CODE[qmap[k]] for k in off_keys]), "def": np.array([TAG_CODE[qmap[k]] for k in def_keys])}
        self.req5 = {"off": np.array([k in ftn for k in off_keys]), "def": np.array([k in ftn for k in def_keys])}
        # the matchups the S5 pool admits: every FTN-sourced profile feature present on both sides
        self.admit5 = ~np.isnan(self.prof_off_p[:, self.req5["off"]]).any(axis=1) & ~np.isnan(self.prof_def[self.opp_row][:, self.req5["def"]]).any(axis=1)
        self.sig5 = {}
        for side, Z in (("off", self.prof_off_p), ("def", self.prof_def[self.opp_row])):
            ok = self.admit5
            dist = lambda A, B, side=side: group_distance(A, B, self.groups5[side], np.ones(A.shape[1]), self.q5[side]).d2
            keys_t, d_ex, _ = sigma_targets(Z[ok], self.key[ok], self.team[ok] if side == "off" else self.team[self.opp_row][ok],
                                            np.zeros(int(ok.sum()), dtype=int), dist)
            for kk in np.unique(self.key):
                m = keys_t < kk
                self.sig5[(side, int(kk))] = float(np.nanmedian(d_ex[m])) if m.sum() >= SIGMA_MIN_TARGETS and np.isfinite(d_ex[m]).any() else float("nan")

    # ---- unit similarities over the team-game prefix
    def _sigma(self, unit, space, key):
        v = self.sig.get((unit, space, int(key)))
        return float("nan") if v is None else v

    def unit_sims(self, unit: str, kind: str, row: int, k: int, key: int, version: str = SEARCH_VERSION) -> tuple:
        """(sim, completeness, quality, shares (3, k)) arrays over the first k team-game rows for `unit` against the vector of team-game `row` (kind 'off': the target
        version -- healthy (the searches), or 'adjusted' for the comparison run -- vs the pool version; 'def': defense units, one version). A pair is compared
        in EXTENDED only when both sides are EXTENDED-complete and the EXTENDED sigma exists; otherwise in BASE."""
        version = version if unit in LINEUP_UNITS else "adjusted"         # a unit without a lineup correction has one target version
        ck = ("u", unit, kind, row, version)
        if ck in self._cache:
            sim, comp, qual, shr = self._cache[ck]
            return sim[:k], comp[:k], qual[:k], shr[:, :k]
        spaces = stored_spaces(unit)
        T = (self.TH if version == "healthy" else self.T)[unit]
        P = self.P[unit]
        kk = self._kmax(key)
        sim = np.full(self.n, np.nan); comp = np.full(self.n, np.nan); qual = np.full(self.n, np.nan); shr = np.full((len(TAGS), self.n), np.nan)
        res = {}
        for space in spaces:
            tv = T[space][0][row][None, :]
            sig = self._sigma(unit, space, key)
            if np.isnan(tv).all() or not np.isfinite(sig):
                continue
            d = unit_distance(tv, P[space][0][:kk], unit, space, tags=True)
            res[space] = (similarity(d.d2[0], sig), d.completeness[0], d.quality[0], d.shares[:, 0, :])
        if res:
            use_ext = (T["extended"][1][row] & P["extended"][1][:kk]) if "extended" in res else np.zeros(kk, dtype=bool)
            for j, arr in enumerate((sim, comp, qual)):
                base = res["base"][j] if "base" in res else np.full(kk, np.nan)
                ext = res["extended"][j] if "extended" in res else np.full(kk, np.nan)
                arr[:kk] = np.where(use_ext, ext, base)
            nan3 = np.full((len(TAGS), kk), np.nan)
            shr[:, :kk] = np.where(use_ext[None, :], res["extended"][3] if "extended" in res else nan3, res["base"][3] if "base" in res else nan3)
        self._cache[ck] = (sim, comp, qual, shr)
        return sim[:k], comp[:k], qual[:k], shr[:, :k]

    def _kmax(self, key: int) -> int:
        return int(np.searchsorted(self.key, key, side="left"))

    def adjusted_differs(self, g: int, units) -> bool:
        """Whether the lineup-adjusted target vector of team-game row g differs from the healthy one in any of `units` or in the S5 offense profile
        (when it does not, the healthy run retrieves exactly what the adjusted run does)."""
        same = lambda a, b: bool(np.array_equal(a, b, equal_nan=True))
        for u in units:
            if u in LINEUP_UNITS:
                for space in self.T[u]:
                    if not (same(self.T[u][space][0][g], self.TH[u][space][0][g]) and self.T[u][space][1][g] == self.TH[u][space][1][g]):
                        return True
        return not same(self.prof_off_t[g], self.prof_off_th[g])

    def clear_cache(self):
        self._cache = {}

    def profile_sims(self, g: int, o: int, k: int, key: int, version: str = SEARCH_VERSION) -> dict:
        """S5 over the first k team-game matchups: {'off': (sim, completeness, quality), 'def': (...)} and whether the target misses a required FTN feature."""
        out, missing = {}, False
        off_t = self.prof_off_th if version == "healthy" else self.prof_off_t
        for side, tv, Pm in (("off", off_t[g], self.prof_off_p[:k]), ("def", self.prof_def[o], self.prof_def[self.opp_row[:k]])):
            missing |= bool(np.isnan(tv[self.req5[side]]).any())
            if np.isnan(tv).all():
                out[side] = (np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan), np.full((len(TAGS), k), np.nan))
                continue
            d = group_distance(tv[None, :], Pm, self.groups5[side], np.ones(len(tv)), self.q5[side], tags=self.tag5[side])
            sim = np.where(self.admit5[:k], similarity(d.d2[0], self.sig5.get((side, int(key)), float("nan"))), np.nan)   # EXTENDED-only pool
            out[side] = (sim, d.completeness[0], d.quality[0], d.shares[:, 0, :])
        return out, missing

    # ---- the five searches
    def search(self, tg: Target, which=("S1", "S2", "S3", "S4", "S5"), keep: bool = True, sim_threshold: float = cs.SIM_THRESHOLD,
               min_neff: float = cs.MIN_NEFF, version: str = SEARCH_VERSION, sensitivity: tuple = (), top_any: int = 0, unit_pools: dict | None = None,
               obs_positions: bool = False) -> dict:
        """Run the searches for one target on the healthy target vectors (SEARCH_VERSION, decision of 2026-10-09; version='adjusted': on the
        lineup-adjusted ones, the stored comparison of build plan 4c.1.4); returns {search: SearchResult}.

        Observations are past (key < the target week) team-games, or, for a player market, past player-games in which the player PLAYED at one of the
        archetype's positions (in_pool). S1: the target's own past games (same team / same player); S2: past games against tonight's defense; S3, S4, S5:
        any team, any defense (build plan 4c.0.6). The similarity of an observation is one-sided for S1 (the defenses faced) and two-sided for S2, S4, S5
        (offense similarity x defense similarity; build plan 4c.4); S3 multiplies the player's archetype similarity by the defense faced. A side's
        similarity is the mean of its units' similarities. A match is an observation whose checked similarity (`check_similarity`: a two-factor
        search's geometric mean, with the weaker factor at least SIDE_FLOOR) is at least `sim_threshold`; its weight is similarity (the product) x recency x
        continuity x data-quality  (separate columns in `matches`; recency and continuity from features/weights.py for the market).
        n_eff counts historical team-games (`cluster_neff`): player-games of one team-game are one cluster. The search is no_match when its best
        checked similarity is below the threshold or its n_eff below `min_neff`: shift = 0 and the widened-uncertainty flag is set.
        `unit_pools` (unit -> Pool of another window over the same games) lets each unit read its own window (4c.6.1); everything else -- the
        observation sets, recency, continuity and the S5 profile -- comes from this pool. obs_positions: each summary also carries 'obs_pos', the
        search's observation set (pool rows for a team market, positions in the archetype population for a player market; the threshold sweep's
        random pairing draws from it)."""
        up = unit_pools or {}
        units = cs.MARKET_UNITS[tg.market]
        off_units = [u for u in units if u in cs.OFFENSE_UNITS]
        def_units = [u for u in units if u in cs.DEFENSE_UNITS]
        arch = [u for u in units if u in cs.ARCHETYPE_UNITS]
        is_player = tg.player_id is not None
        g, o = self.idx[(tg.game_id, tg.team)], self.idx[(tg.game_id, tg.opponent)]
        key = int(self.key[g])
        k = self._kmax(key)
        out = {}
        # ---- the observation set (past team-games, or past player-games of the market's population)
        if is_player:
            pa = up.get(arch[0], self)                             # the pool of the archetype's window (the same player rows in every window)
            pop = pa.pl[arch[0]]
            kp = int(np.searchsorted(pop["key"], key, side="left"))
            oi = np.flatnonzero(pop["in_pool"][:kp])               # a scored player-game in which he did not play is no observation
            obs_row, obs_pid, obs_team = pop["row"][oi], pop["pid"][oi], pop["team"][oi]
            obs_slot, obs_fam = pop["slot"][oi], pop["fam"][oi]
            tpos = np.flatnonzero((pop["pid"] == tg.player_id) & (pop["row"] == g))       # the target's own row: stored whether or not he played
            tvec = {s: pop["blocks"][s][0][tpos[0]][None, :] for s in pop["blocks"]} if len(tpos) else None
            t_ext = bool(pop["blocks"]["extended"][1][tpos[0]]) if len(tpos) and "extended" in pop["blocks"] else False
            t_slot, t_fam = (int(pop["slot"][tpos[0]]), int(pop["fam"][tpos[0]])) if len(tpos) else (0, 0)
        else:
            obs_row = np.arange(k)
            obs_pid = np.full(k, None, dtype=object)
            obs_team = self.team[:k]
            obs_slot = obs_fam = np.zeros(k, dtype=np.int64)
            tvec, t_ext, t_slot, t_fam = None, False, 0, 0
        obs_opp = self.opp_row[obs_row] if len(obs_row) else obs_row
        obs_def_team = self.team[obs_opp] if len(obs_row) else obs_team
        # ---- component similarities (each: sim, completeness, quality arrays over the observations)
        def team_comp(unit, kind, row, idx):
            s, c, q, sh = up.get(unit, self).unit_sims(unit, kind, row, k, key, version)
            return s[idx], c[idx], q[idx], sh[:, idx]
        def arch_comp():
            """The player's archetype similarity to each observation (one version: a player's own history has no lineup correction). EXTENDED only when
            both sides are EXTENDED-complete and the EXTENDED sigma exists; otherwise BASE."""
            if not is_player or tvec is None:
                return None
            unit = arch[0]
            res = {}
            for space in stored_spaces(unit):
                sig = pa._sigma(unit, space, key)
                if not np.isfinite(sig):
                    continue
                d = unit_distance(tvec[space], pop["blocks"][space][0][oi], unit, space, tags=True)
                res[space] = (similarity(d.d2[0], sig), d.completeness[0], d.quality[0], d.shares[:, 0, :])
            nan, nan3 = np.full(len(oi), np.nan), np.full((len(TAGS), len(oi)), np.nan)
            use_ext = (pop["blocks"]["extended"][1][oi] & t_ext) if "extended" in res else np.zeros(len(oi), dtype=bool)
            pick = lambda j: np.where(use_ext, res["extended"][j] if "extended" in res else nan, res["base"][j] if "base" in res else nan)
            shr = np.where(use_ext[None, :], res["extended"][3] if "extended" in res else nan3, res["base"][3] if "base" in res else nan3)
            return pick(0), pick(1), pick(2), shr
        ac = None
        if arch and is_player and tvec is not None:
            ck = ("a", arch[0], g, tg.player_id, pa.window)         # one player's archetype similarities serve every search, market and version
            if ck not in self._cache:
                self._cache[ck] = arch_comp()
            ac = self._cache[ck]

        def components(search):
            """The unit similarities a search uses (build plan 4c.4): S1 one-sided on the defenses faced (the offense is the target itself); S3 the player's
            archetype and the defense faced; S2 and S4 two-sided, offense (team units + the player's archetype) and defense. In S2 the defense is
            tonight's opponent itself, compared with what it was in that past game."""
            comps = {}
            for u in def_units:
                comps[u] = ("def", team_comp(u, "def", o, obs_opp))
            if search in ("S2", "S4"):
                for u in off_units:
                    comps[u] = ("off", team_comp(u, "off", g, obs_row))
            if search in ("S2", "S3", "S4") and ac is not None:
                comps[arch[0]] = ("arch", ac)
            return comps

        def side_mean(comps, kinds):
            sims = [c[0] for kd, c in comps.values() if kd in kinds]
            if not sims:
                n_ = len(next(iter(comps.values()))[1][0]) if comps else len(obs_row)
                return np.full(n_, np.nan)
            arr = np.vstack(sims)
            cnt = (~np.isnan(arr)).sum(axis=0)
            return np.where(cnt > 0, np.nansum(arr, axis=0) / np.maximum(cnt, 1), np.nan)

        for sname in which:
            if sname == "S3" and not is_player:
                out[sname] = SearchResult(sname, self._summary(tg, sname, None, None, None, False, "not_applicable", sim_threshold, min_neff, applicable=False))
                continue
            if is_player and tvec is None:
                out[sname] = SearchResult(sname, self._summary(tg, sname, None, None, None, False, "no_target_vector", sim_threshold, min_neff))
                continue
            forced = None
            # which observations the search looks at
            if sname == "S1":
                sel = (obs_pid == tg.player_id) if is_player else (obs_team == tg.team)
            elif sname == "S2":
                sel = obs_def_team == tg.opponent
            else:
                sel = np.ones(len(obs_row), dtype=bool)            # S3 "on any team", S4 and S5: every past observation
            idx = np.flatnonzero(sel)
            if sname == "S5":
                prof, miss = self.profile_sims(g, o, k, key, version)
                comps_sel = {f"matchup_{'offense' if side == 'off' else 'defense'}": (side, tuple(a[..., obs_row[idx]] for a in prof[side])) for side in ("off", "def")}
                sim_off, sim_def = side_mean(comps_sel, ("off",)), side_mean(comps_sel, ("def",))
                combined = sim_off * sim_def
                forced = "missing_required_ftn_features" if miss else None
            else:
                allc = components(sname)
                comps_sel = {u: (kd, tuple(a[..., idx] for a in c)) for u, (kd, c) in allc.items()}
                sim_def = side_mean(comps_sel, ("def",))
                if sname == "S1":
                    sim_off = np.full(len(idx), np.nan)
                    combined = sim_def
                elif sname == "S3":
                    sim_off = side_mean(comps_sel, ("arch",))
                    combined = sim_off * sim_def
                else:
                    sim_off = side_mean(comps_sel, ("off", "arch"))
                    combined = sim_off * sim_def
            check = check_similarity(sname, combined, sim_off, sim_def)
            # weights
            rec = 0.5 ** (np.maximum(self.clock[g] - self.clock[obs_row[idx]], 0.0) / config.RECENCY_HALF_LIFE_GAMES)
            # continuity from features/weights.py for the market, in every search (build plan 4c.4): a past game of another team flags every factor
            lin_past = {c: self.lin[i][obs_row[idx]] for i, c in enumerate(LINEUP_COMPONENTS)}
            lin_t = {c: int(self.lin[i][g]) for i, c in enumerate(LINEUP_COMPONENTS)}
            flags = wts.continuity_flags(lin_past, lin_t, obs_slot[idx], t_slot, obs_fam[idx], t_fam, obs_team[idx] != tg.team)
            cont = np.asarray(wts.continuity_weight(flags, config.BASELINE_TO_PENALTY_MARKET[tg.market]), dtype=float) * np.ones(len(idx))
            comp_arr = np.vstack([c[1][1] for c in comps_sel.values()]) if comps_sel else np.zeros((0, len(idx)))
            qual_arr = np.vstack([c[1][2] for c in comps_sel.values()]) if comps_sel else np.zeros((0, len(idx)))
            n_exp = max(len(comps_sel), 1)
            completeness = np.where(np.isnan(comp_arr), 1.0, comp_arr).sum(axis=0) / n_exp if comps_sel else np.full(len(idx), np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                quality = np.nanmean(qual_arr, axis=0) if comps_sel else np.full(len(idx), np.nan)
            final = combined * rec * cont * quality
            share = _match_shares(comps_sel, sname, len(idx))        # 4c.5: the observed / derived / estimated share behind each similarity
            summ, matches = self._finish(tg, sname, idx, obs_row, obs_pid, obs_team, combined, comps_sel, rec, cont, quality, completeness, final,
                                         sim_threshold, min_neff, keep, sim_off, sim_def, sensitivity, top_any, share, check)
            if obs_positions:
                summ["obs_pos"] = oi[idx] if is_player else obs_row[idx]
            if forced is not None:                   # S5 without its required FTN features: no_match whatever its similarities (plan 4c.3)
                summ.update(no_match=True, widened_uncertainty=True, shift=0.0, reason=forced, n_matches=0, n_eff=0.0)
                if "sensitivity" in summ:
                    summ["sensitivity"] = {}
                matches = None
            out[sname] = SearchResult(sname, summ, matches)
        return out

    def _summary(self, tg, sname, combined, comps, final, matched, reason, sim_threshold, min_neff, applicable=True, best_override=None, n_matches=0, n_eff=0.0):
        best = best_override if best_override is not None else (float(np.nanmax(combined)) if combined is not None and np.isfinite(combined).any() else float("nan"))
        no_match = applicable and (matched is False or reason is not None)
        unit_best = {}
        if comps:
            for u, (kd, c) in comps.items():
                s = c[0]
                unit_best[u] = float(np.nanmax(s)) if np.isfinite(s).any() else float("nan")
        return dict(search=sname, market=tg.market, game_id=tg.game_id, team=tg.team, player_id=tg.player_id, season=tg.season, week=tg.week,
                    applicable=applicable, n_matches=n_matches, n_eff=n_eff, best_similarity=best, no_match=bool(no_match) if applicable else False,
                    widened_uncertainty=bool(no_match) if applicable else False, shift=0.0 if no_match else float("nan"), reason=reason, unit_best=unit_best)

    def _finish(self, tg, sname, idx, obs_row, obs_pid, obs_team, combined, comps, rec, cont, quality, completeness, final, sim_threshold, min_neff, keep,
                sim_off=None, sim_def=None, sensitivity=(), top_any=0, share=None, check=None):
        chk = combined if check is None else check                     # the similarity the no-match rule reads (check_similarity)
        ok = np.isfinite(combined) & np.isfinite(final)
        best = float(np.nanmax(chk)) if np.isfinite(chk).any() else float("nan")
        with np.errstate(invalid="ignore"):
            match = ok & (chk >= sim_threshold)
        n_eff = cluster_neff(final[match], obs_row[idx][match])      # player-games of one historical team-game count as one cluster
        reason = None
        if not np.isfinite(best):
            reason = "weaker_factor_below_floor" if np.isfinite(combined).any() else "no_similarity"
        elif best < sim_threshold:
            reason = "best_similarity_below_threshold"
        elif not meets_min_neff(n_eff, min_neff):
            reason = "n_eff_below_minimum"
        summary = self._summary(tg, sname, combined, comps, final, reason is None, reason, sim_threshold, min_neff, best_override=best,
                                n_matches=int(match.sum()), n_eff=n_eff)
        if top_any:                         # the closest past games whatever the threshold (the healthy-vs-adjusted comparison): (id, final weight)
            fin = np.flatnonzero(ok)
            if len(fin) > top_any:                    # keep every observation tied with the k-th largest, then order them exactly: ties by position
                kth = np.partition(final[fin], len(fin) - top_any)[len(fin) - top_any]
                fin = fin[final[fin] >= kth]
            top = fin[np.lexsort((fin, -final[fin]))][:top_any]
            ti = idx[top]                             # positions in the observation arrays (index them once: they can hold 70,000 objects)
            sh_ = share[:, top] if share is not None else np.full((len(TAGS), len(top)), np.nan)
            summary["top_any"] = [(f"{self._gids[r]}|{t_}|{'' if p_ is None else p_}", float(w_), *map(float, x_), float(c_))
                                  for r, t_, p_, w_, x_, c_ in zip(obs_row[ti].tolist(), obs_team[ti].tolist(), obs_pid[ti].tolist(), final[top].tolist(),
                                                                   sh_.T.tolist(), completeness[top].tolist())]
        if sensitivity:                     # analysis only (the no-match memo): matches and n_eff at other thresholds, from the same similarities
            with np.errstate(invalid="ignore"):
                summary["sensitivity"] = {f"{t:g}": (int((ok & (chk >= t)).sum()), cluster_neff(final[ok & (chk >= t)], obs_row[idx][ok & (chk >= t)]))
                                          for t in sensitivity}
        matches = None
        if keep:
            sel = np.flatnonzero(match)
            cols = {"search": [sname] * len(sel), "market": [tg.market] * len(sel), "target_game_id": [tg.game_id] * len(sel), "target_team": [tg.team] * len(sel),
                    "target_player_id": [tg.player_id] * len(sel),
                    "obs_game_id": [self._gids[i] for i in obs_row[idx][sel]], "obs_team": [str(t) for t in obs_team[idx][sel]],
                    "obs_player_id": [None if p is None else str(p) for p in obs_pid[idx][sel]],
                    "obs_season": self.games["season"].to_numpy()[obs_row[idx][sel]], "obs_week": self.games["week"].to_numpy()[obs_row[idx][sel]]}
            for u, (kd, c) in comps.items():
                cols[f"sim_{u}"] = c[0][sel]
            nan = np.full(len(combined), np.nan)
            cols.update(sim_offense=(nan if sim_off is None else sim_off)[sel], sim_defense=(nan if sim_def is None else sim_def)[sel])
            cols.update(sim_combined=combined[sel], sim_check=chk[sel], completeness_penalty=completeness[sel], recency_weight=rec[sel], continuity_weight=cont[sel],
                        quality_weight=quality[sel], final_weight=final[sel])
            if share is not None:
                cols.update({f"share_{t[:3]}": share[i][sel] for i, t in enumerate(TAGS)})        # share_obs / share_der / share_est
            matches = pl.DataFrame(cols, schema_overrides={"target_player_id": pl.String, "obs_player_id": pl.String}, strict=False).sort(
                ["sim_combined", "obs_game_id", "obs_team", "obs_player_id"], descending=[True, False, False, False], nulls_last=True)
        return summary, matches


class WindowedPool:
    """Comparable searches whose units each read the window chosen for them (4c.6.1; `choice`: market -> unit -> window, comp_windows.json). Every
    pool holds the same games, so the observation sets, recency and continuity are the base pool's; so is the S5 matchup profile (not one of the
    plan's units; it stays on the base window, LOG_WINDOW)."""

    def __init__(self, pools: dict, choice: dict, base: str = None):
        missing = [(m, u) for m, units in cs.MARKET_UNITS.items() for u in units if u not in choice.get(m, {})]
        unknown = sorted({w for c in choice.values() for w in c.values()} - set(pools))
        if missing or unknown:                         # never fall back to the base window silently
            raise ValueError(f"window choice incomplete: {len(missing)} (market, unit) without a window {missing[:3]}; windows without a pool {unknown}")
        self.pools, self.choice = pools, choice
        self.base = pools[base or LOG_WINDOW]
        self.window = "selected"

    def __getattr__(self, name):                       # idx, key, games ... are the base pool's
        if name.startswith("__") or name in ("base", "pools", "choice"):
            raise AttributeError(name)                 # copy / pickle probe the instance before __init__ ran: no recursion through self.base
        return getattr(self.base, name)

    def clear_cache(self):
        for p in self.pools.values():
            p.clear_cache()

    def search(self, tg: Target, *args, **kwargs) -> dict:
        return self.base.search(tg, *args, unit_pools={u: self.pools[w] for u, w in self.choice[tg.market].items()}, **kwargs)

    def adjusted_differs(self, g: int, units) -> bool:
        return any(p.adjusted_differs(g, units) for p in self.pools.values())


# ---------------------------------------------------------------------------------------------------------------------------------- targets and the no-match log
MARKET_UNITS_TEAM = set(cs.MARKET_UNITS["spread"])
MARKET_FAMILIES = {"pass": ("pass_att", "pass_cmp", "pass_yds"), "qb_rush": ("qb_rush_att", "qb_rush_yds"), "rush": ("rush_att", "rush_yds"),
                   "receiving": ("targets", "rec", "rec_yds"), "game": ("spread", "moneyline", "total")}
FAMILY_OF_MARKET = {m: f for f, ms in MARKET_FAMILIES.items() for m in ms}
assert all(MARKET_FAMILIES[f] and len({cs.MARKET_UNITS[m] for m in ms}) == 1 for f, ms in MARKET_FAMILIES.items()), "markets in a family must share their units"
LOG_WINDOW = "recency_weighted"        # the window the no-match log is run on (4c.6 chooses a window per unit and market)


def backtest_targets(raw_db=config.RAW_DUCKDB_PATH) -> list:
    """Every scored target of the 2020-2024 backtest, once per market family (markets of a family search identically): each eligible player-game for
    the families it is eligible for, and both team sides of every game for the game markets."""
    from eval import backtest as bt
    data = bt.load_backtest_data(raw_db)
    out = []
    for r in data.player_log.filter(pl.col("elig") != "").iter_rows(named=True):
        fams = []
        for m in bt.eligible_markets(r):
            if FAMILY_OF_MARKET[m] not in fams:
                fams.append(FAMILY_OF_MARKET[m])
        for f in fams:
            out.append(Target(MARKET_FAMILIES[f][0], r["game_id"], r["team"], r["opponent"], r["season"], r["week"], r["player_id"]))
    for r in data.team_log.iter_rows(named=True):
        out.append(Target(MARKET_FAMILIES["game"][0], r["game_id"], r["team"], r["opponent"], r["season"], r["week"], None))
    return sorted(out, key=lambda t: (t.season, t.week, t.game_id, t.team, t.player_id or "", t.market))


def run_search_log(pool: Pool, targets: list, which=("S1", "S2", "S3", "S4", "S5"), sim_threshold: float = cs.SIM_THRESHOLD, min_neff: float = cs.MIN_NEFF,
                   sensitivity: tuple = ()) -> pl.DataFrame:
    """One row per (target, search): n_matches, n_eff, best similarity, no_match, reason and the best similarity of every unit. No match rows are not
    stored here; call Pool.search(..., keep=True) for them."""
    rows = []
    last = None
    for tg in targets:
        if (tg.game_id, tg.team) != last:
            pool.clear_cache()
            last = (tg.game_id, tg.team)
        for sname, res in pool.search(tg, which, keep=False, sim_threshold=sim_threshold, min_neff=min_neff, sensitivity=sensitivity).items():
            r = dict(res.summary)
            ub = r.pop("unit_best")
            r["unit_best"] = json.dumps({k: (None if v != v else round(v, 6)) for k, v in sorted(ub.items())})
            if sensitivity:                  # a search that never reached the similarity step (not applicable, no vector, S5 without FTN) matches at no threshold
                r["sensitivity"] = json.dumps({k: [n, round(e, 9)] for k, (n, e) in r.get("sensitivity", {}).items()})
            rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def aggregate_search_log(summary: pl.DataFrame, sim_threshold: float = cs.SIM_THRESHOLD) -> pl.DataFrame:
    """Per season, market, search: how often the search returns no_match (and why), and per unit how often the unit alone has no similarity at all or
    nothing above the threshold. Markets of a family are searched once and reported under each market."""
    s = summary.filter(pl.col("applicable"))
    rows = []
    for (season, market, search), g in s.group_by("season", "market", "search", maintain_order=True):
        fam = FAMILY_OF_MARKET[market]
        n = g.height
        med = g["best_similarity"].fill_nan(None).median()               # targets without any similarity are left out, not sorted above every value
        base = dict(season=season, search=search, n_targets=n, no_match_rate=float(g["no_match"].mean()), median_best_similarity=float("nan") if med is None else float(med),
                    median_n_eff=float(g["n_eff"].median()), reasons=json.dumps(dict(sorted(g["reason"].drop_nulls().value_counts().iter_rows()))))
        for m in MARKET_FAMILIES[fam]:
            rows.append(dict(base, market=m, unit="combined", unit_missing_rate=None, unit_below_threshold_rate=None))
        ub = [json.loads(x) for x in g["unit_best"]]
        for unit in sorted({u for d in ub for u in d}):
            vals = [d.get(unit) for d in ub]
            miss = np.mean([v is None for v in vals])
            below = np.mean([(v is None) or (v < sim_threshold) for v in vals])
            for m in MARKET_FAMILIES[fam]:
                rows.append(dict(base, market=m, unit=unit, unit_missing_rate=float(miss), unit_below_threshold_rate=float(below)))
    return pl.DataFrame(rows).sort("season", "market", "search", "unit")


def load_windowed_pool(choice_path=None, raw_db=config.RAW_DUCKDB_PATH, vec_dir=config.PROCESSED_DIR, log_dir=config.ROOT) -> WindowedPool:
    """The pools of every window comp_windows.json chooses (plus the base window), wrapped so each unit reads its own."""
    choice = json.loads(open(choice_path or log_dir / "comp_windows.json").read())
    windows = sorted({w for m in choice.values() for w in m.values()} | {LOG_WINDOW})
    bad = sorted(set(windows) - {vname(v) for v in VARIANTS})
    if bad:
        raise ValueError(f"comp_windows.json names windows that are not built: {bad}")
    return WindowedPool({w: load_pool(w, raw_db, vec_dir, log_dir) for w in windows}, choice)


def load_pool(window: str = LOG_WINDOW, raw_db=config.RAW_DUCKDB_PATH, vec_dir=config.PROCESSED_DIR, log_dir=config.ROOT) -> Pool:
    """The comparable pool for a window from the stored vectors, sigma table and the S5 results-side statistics (rebuilt from the plays in seconds)."""
    from features import comps_ledger as L
    from features import lineups as LU
    games = L.load_games(raw_db)
    plays = L.load_plays(raw_db)
    lineups = LU.build_lineups(raw_db, None, start=cs.BASE_START).with_columns(team=L.canon("team"))
    matchup = matchup_vector_frames(games, lineups, L.interaction_ledger(plays, games), [v for v in VARIANTS if vname(v) == window])
    return Pool(pl.read_parquet(vec_dir / "comp_vectors_team.parquet"), pl.read_parquet(vec_dir / "comp_vectors_player.parquet"), matchup, games, lineups,
                L.load_slots(raw_db), pl.read_parquet(log_dir / "comp_sigma.parquet"), window)


# ================================================================================================================================ 4c.4 residuals and shifts
# market -> (volume quantity, efficiency quantity) in walkforward_predictions (build plan 3.5.3): the comp-free expectations 4c.4 standardises against
MARKET_QUANTITIES = {"pass_att": ("pass_att", None), "pass_cmp": ("pass_att", "comp_rate"), "pass_yds": ("pass_att", "yds_per_cmp"),
                     "rush_att": ("rush_att", None), "rush_yds": ("rush_att", "ypc"),
                     "targets": ("targets", None), "rec": ("targets", "catch_rate"), "rec_yds": ("targets", "yds_per_rec"),
                     "qb_rush_att": ("qb_rush_att", None), "qb_rush_yds": ("qb_rush_att", "qb_ypc"),
                     "total": ("team_plays", "pts_per_play"), "spread": ("team_plays", "pts_per_play"), "moneyline": ("team_plays", "pts_per_play")}
Z_OWN_K = 10.0              # plan 4c.4: sigma blends toward the entity's own residual SD with n / (n + 10)
Z_ROLE_MIN = 30             # a role's error SD needs this many earlier residuals; fewer -> the quantity's all-role SD
SEARCHES = ("S1", "S2", "S3", "S4", "S5")


def standardized_residuals(wf: pl.DataFrame) -> pl.DataFrame:
    """z = (actual - expected) / sigma for every walk-forward row that has both (plan 4c.4.1). The expected value is the comp-free Phase 3 prediction.
    sigma is as of the row's OWN week: the RMS of the residuals of earlier weeks for that quantity and role (the quantity's all-role RMS while the role has
    fewer than Z_ROLE_MIN), blended toward the entity's own earlier RMS with n / (n + Z_OWN_K), n = his earlier residuals."""
    w = (wf.filter(pl.col("residual").is_not_null() & pl.col("expected").is_not_null() & pl.col("actual").is_not_null())
         .with_columns(key=pl.col("season") * 100 + pl.col("week"), entity=pl.coalesce("player_id", "team"), sq=pl.col("residual") ** 2)
         .sort("quantity", "key", "game_id", "entity"))
    def earlier(by: list) -> pl.DataFrame:
        """per (by..., key): sum of squares and count over strictly earlier weeks."""
        g = w.group_by(*by, "key").agg(ss=pl.col("sq").sum(), n=pl.len()).sort(*by, "key")
        return g.with_columns(ss_prev=(pl.col("ss").cum_sum() - pl.col("ss")).over(by), n_prev=(pl.col("n").cum_sum() - pl.col("n")).over(by)).drop("ss", "n")
    role = earlier(["quantity", "role"]).rename({"ss_prev": "ss_role", "n_prev": "n_role"})
    allr = earlier(["quantity"]).rename({"ss_prev": "ss_all", "n_prev": "n_all"})
    own = earlier(["quantity", "entity"]).rename({"ss_prev": "ss_own", "n_prev": "n_own"})
    z = (w.join(role, on=["quantity", "role", "key"], how="left").join(allr, on=["quantity", "key"], how="left")
         .join(own, on=["quantity", "entity", "key"], how="left"))
    sig_role = pl.when(pl.col("n_role") >= Z_ROLE_MIN).then((pl.col("ss_role") / pl.col("n_role")).sqrt()).otherwise(
        pl.when(pl.col("n_all") > 0).then((pl.col("ss_all") / pl.col("n_all")).sqrt()))
    sig_own = pl.when(pl.col("n_own") > 0).then((pl.col("ss_own") / pl.col("n_own")).sqrt())
    lam = pl.col("n_own") / (pl.col("n_own") + Z_OWN_K)
    z = z.with_columns(sigma_role=sig_role, sigma_own=sig_own).with_columns(
        sigma=pl.when(pl.col("sigma_own").is_not_null()).then(lam * pl.col("sigma_own") + (1 - lam) * pl.col("sigma_role")).otherwise(pl.col("sigma_role")))
    z = z.with_columns(z=pl.when(pl.col("sigma") > 0).then(pl.col("residual") / pl.col("sigma")))
    return z.select("season", "week", "game_id", "player_id", "team", "role", "quantity", "expected", "actual", "residual", "sigma", "n_own", "z").sort(
        "quantity", "season", "week", "game_id", "team", "player_id", nulls_last=True)


def cap_shares(weights: np.ndarray, groups: np.ndarray, cap: float) -> np.ndarray:
    """Weights rescaled so no group holds more than `cap` of their total; the excess goes to the other groups pro rata (plan 4c.4.3). The total is kept.
    Only groups with weight count (a past game without an expectation carries none and cannot absorb any). With fewer than 1 / cap such groups no split
    can respect the cap, and the weights are left as they are: n_eff, computed after the caps, then decides whether the search matches. (An equal split
    would claim an n_eff of 3 for weights 0.98 / 0.01 / 0.01.)"""
    w = np.asarray(weights, dtype=float)
    tot = w.sum()
    if len(w) == 0 or tot <= 0:
        return w.copy()
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    share = np.bincount(inv, weights=w) / tot
    pos = share > 0
    if pos.sum() * cap < 1.0 - 1e-12:
        return w.copy()
    target, capped = share.copy(), np.zeros(len(share), dtype=bool)
    for _ in range(len(share)):
        over = (target > cap + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        free = pos & ~capped
        target[capped] = cap
        rest = 1.0 - cap * capped.sum()
        if free.any() and share[free].sum() > 0:
            target[free] = share[free] / share[free].sum() * rest
    factor = np.divide(target, share, out=np.zeros(len(share)), where=pos)
    return w * factor[inv]


CAP_AVG_EXEMPT = ("S1",)    # S1 is the target's own history: one team by construction (plan 4c.0.6), so it cannot enter the across-search average


def cap_across_searches(per_search: dict, cap_one: float = cs.TEAM_CAP_PER_SEARCH, cap_avg: float = cs.TEAM_CAP_AVG_ACROSS_SEARCHES,
                        n_searches: int = len(SEARCHES), max_iter: int = 50, exempt=CAP_AVG_EXEMPT, info: dict | None = None) -> dict:
    """{search: (weights, team_game, team)} -> {search: capped weights}. No historical team-game above `cap_one` of a search's weight (cap_shares), and
    no historical team (the observation's offense, obs_team) averaging more than `cap_avg` of the weight across the five searches (a search without
    weight counts as 0). Pass the searches that match only: a search that ends no_match carries no weight.

    An over-cap team is scaled down by one common factor in the searches where other teams can take its weight (pro rata, totals kept), just enough to
    bring its average to the cap. Its share in a search where nobody else can take it is irreducible; when the irreducible part alone breaks the cap,
    the cap cannot hold, and the team is left as it is (listed in info['frozen']) instead of being pushed out of every other search. The searches in
    `exempt` (S1, the target's own past games: one team by definition) keep the per-team-game cap but neither count toward the average nor move."""
    w = {s: cap_shares(v[0], v[1], cap_one) for s, v in per_search.items()}
    tms = {s: np.asarray(v[2]).astype(str) for s, v in per_search.items()}
    frozen = set()

    def shares():
        live = [s for s in per_search if s not in exempt and w[s].sum() > 0]
        return live, {s: {t: float(w[s][tms[s] == t].sum() / w[s].sum()) for t in np.unique(tms[s]).tolist()} for s in live}

    for _ in range(max_iter):
        live, sh = shares()
        teams = sorted({t for s in live for t in sh[s]})
        avg = {t: sum(sh[s].get(t, 0.0) for s in live) / n_searches for t in teams}
        over = [t for t in teams if avg[t] > cap_avg + 1e-12 and t not in frozen]
        if not over:
            break
        moved = False
        for t in over:
            live, sh = shares()
            blocked = set(over) | frozen
            red = [s for s in live if sh[s].get(t, 0.0) > 0 and any(u not in blocked and v > 0 for u, v in sh[s].items())]
            irreducible = sum(sh[s].get(t, 0.0) for s in live if s not in red)
            reducible = sum(sh[s][t] for s in red)
            alpha = (n_searches * cap_avg - irreducible) / reducible if reducible > 0 else -1.0
            if alpha <= 0:
                frozen.add(t)
                continue
            alpha = min(alpha, 1.0)
            for s in red:
                m = tms[s] == t
                recv = ~np.isin(tms[s], list(blocked))
                freed = w[s][m].sum() * (1.0 - alpha)
                if freed <= 0 or w[s][recv].sum() <= 0:
                    continue
                w[s][recv] *= (w[s][recv].sum() + freed) / w[s][recv].sum()
                w[s][m] *= alpha
                moved = True
        for s in per_search:
            w[s] = cap_shares(w[s], per_search[s][1], cap_one)
        if not moved:
            break
    if info is not None:
        info["frozen"] = sorted(frozen)
    return w


NEFF_TOL = 1e-9          # n_eff of two equal past games is 2 up to rounding: compare with a tolerance, never by the last bit


def meets_min_neff(n: float, min_neff: float) -> bool:
    return bool(n >= min_neff - NEFF_TOL)


def neff(w: np.ndarray) -> float:
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def shift_value(w: np.ndarray, z: np.ndarray, clusters=None) -> tuple:
    """(shift, n_eff): the weighted mean z shrunk by n_eff / (n_eff + SHRINK_K) (plan 4c.4.4); (0, 0) without weight. With `clusters` (the historical
    team-game of each match) n_eff counts team-games (cluster_neff)."""
    ok = np.isfinite(z) & (w > 0)
    w, z = w[ok], z[ok]
    if not len(w):
        return 0.0, 0.0
    n = neff(w) if clusters is None else cluster_neff(w, np.asarray(clusters)[ok])
    return float((w * z).sum() / w.sum() * n / (n + cs.SHRINK_K)), n


# the denominator of each efficiency ratio (models/efficiency.py EFFICIENCY_SPECS, by its walkforward_predictions name): an efficiency match weighs
# its final weight x this count (decision of 2026-10-09), as the Phase 3 efficiency model weights its rows
EFF_DENOMINATORS = {"comp_rate": "attempts", "yds_per_cmp": "completions", "ypc": "carries", "qb_ypc": "rush_att_ex_kneel", "catch_rate": "targets",
                    "yds_per_rec": "receptions"}
TEAM_EFF_COUNT = ("pts_per_play", "team_plays")      # points per play: the team's plays that game (the team_plays actual)


def efficiency_counts(player_log: pl.DataFrame, wf: pl.DataFrame) -> dict:
    """quantity -> {(game_id, player_id or team): count}: the denominator behind each efficiency z, from the backtest player log and the team_plays
    actuals of walkforward_predictions (2020-2024 scored rows, where the z exist)."""
    out = {}
    for q, col in EFF_DENOMINATORS.items():
        out[q] = dict(zip(zip(player_log["game_id"].to_list(), player_log["player_id"].to_list()), player_log[col].cast(pl.Float64).to_list()))
    tp = wf.filter((pl.col("quantity") == TEAM_EFF_COUNT[1]) & pl.col("actual").is_not_null())
    out[TEAM_EFF_COUNT[0]] = dict(zip(zip(tp["game_id"].to_list(), tp["team"].to_list()), tp["actual"].cast(pl.Float64).to_list()))
    return out


def load_efficiency_counts(wf: pl.DataFrame) -> dict:
    """efficiency_counts from the backtest player log (eval.backtest, 2020-2024; 2025 never loaded) and walkforward_predictions `wf`."""
    from eval import backtest as bt
    return efficiency_counts(bt.load_backtest_data().player_log, wf)


def z_lookup(zt: pl.DataFrame) -> dict:
    """quantity -> {(game_id, player_id or team): z} from standardized_residuals (rows without z are left out)."""
    out = {}
    for (q,), g in zt.filter(pl.col("z").is_not_null()).group_by("quantity", maintain_order=True):
        ent = g.select(pl.coalesce("player_id", "team")).to_series().to_list()
        out[q] = dict(zip(zip(g["game_id"].to_list(), ent), g["z"].to_list()))
    return out


def _target_rows(targets: list, choice: dict | None = None) -> list:
    """Every backtest target expanded from its market family to its markets, grouped by the VALUES of their continuity penalty row (markets whose rows
    are equal search identically, whatever the row's name): [(target with a representative market, [markets])]."""
    out = []
    for tg in targets:
        fam = FAMILY_OF_MARKET[tg.market]
        by_pk = {}
        for m in MARKET_FAMILIES[fam]:
            wkey = tuple(sorted(choice[m].items())) if choice else ()         # markets whose units read different windows search separately
            by_pk.setdefault((tuple(sorted(config.CONTINUITY_PENALTIES[config.BASELINE_TO_PENALTY_MARKET[m]].items())), wkey), []).append(m)
        for pk, ms in by_pk.items():
            out.append((Target(ms[0], tg.game_id, tg.team, tg.opponent, tg.season, tg.week, tg.player_id), ms))
    return out


RETRIEVAL_TOP_K = 10        # healthy vs lineup-adjusted run: the overlap of the top 10 matches by final weight (build plan 4c.1.4)


_SHARE_FACTORS = {"S1": (("def",),), "S2": (("off", "arch"), ("def",)), "S3": (("arch",), ("def",)), "S4": (("off", "arch"), ("def",)),
                  "S5": (("off",), ("def",))}


def _match_shares(comps_sel: dict, sname: str, n: int) -> np.ndarray:
    """(3, n): the observed / derived / estimated share behind each observation's similarity, mirroring how the similarity is formed: the mean over
    its factors (S1 the defense side; S2 / S4 the offense side -- team units and the archetype -- and the defense side; S3 the archetype and the
    defense side; S5 the two profile sides), each factor the mean of its units' shares."""
    facs = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for kinds in _SHARE_FACTORS[sname]:
            arr = [c[3] for kd, c in comps_sel.values() if kd in kinds]
            if arr:
                facs.append(np.nanmean(np.stack(arr), axis=0))
        return np.nanmean(np.stack(facs), axis=0) if facs else np.full((len(TAGS), n), np.nan)


def _wmean(w: np.ndarray, x: np.ndarray):
    ok = np.isfinite(x) & (w > 0)
    return float((w[ok] * x[ok]).sum() / w[ok].sum()) if ok.any() else None


def _side_weights(sets: dict, j: int, min_neff: float) -> tuple:
    """The capped weights of one side (j = 1 volume z, 2 efficiency z) over the searches in `sets`: (weights per search, n_eff per search, the searches
    that match, the teams the across-search cap left frozen). A match counts only with its own expectation. On the efficiency side a match's weight
    is its final weight x its count (sets[s][4]; decision of 2026-10-09), before the caps, so the caps and n_eff see the count-weighted weights. A
    search matches when its n_eff after the per-team-game cap reaches MIN_NEFF; only matching searches enter the across-search cap (plan 4c.4.3), and
    the check is repeated after it until the set of matching searches is stable."""
    cand = {s: v for s, v in sets.items() if np.isfinite(v[j]).any()}
    base = {s: np.where(np.isfinite(v[j]), v[0]["final_weight"].to_numpy() * (v[4] if j == 2 else 1.0), 0.0) for s, v in cand.items()}
    weights = {s: cap_shares(base[s], cand[s][3], cs.TEAM_CAP_PER_SEARCH) for s in cand}
    nef = {s: cluster_neff(weights[s], cand[s][3]) for s in cand}
    alive = {s for s in cand if meets_min_neff(nef[s], min_neff)}
    frozen = []
    for _ in range(len(SEARCHES) + 1):
        info = {}
        capped = cap_across_searches({s: (base[s], cand[s][3], cand[s][0]["obs_team"].to_numpy()) for s in sorted(alive)}, info=info) if alive else {}
        frozen = info.get("frozen", [])
        for s_, wt in capped.items():
            weights[s_], nef[s_] = wt, cluster_neff(wt, cand[s_][3])
        still = {s_ for s_ in alive if meets_min_neff(nef[s_], min_neff)}
        if still == alive:
            break
        alive = still
    for s_ in cand:
        if s_ not in alive:
            weights[s_] = np.zeros(len(base[s_]))                 # a search that does not match carries no weight into any shift
    return weights, nef, alive, frozen


def _market_shifts(res: dict, tg: Target, market: str, zl: dict, min_neff: float, counts: dict | None = None) -> tuple:
    """The 4c.4 shifts of one target and market from its search results: (feature values, detail rows, {search: (matches, z_vol, z_eff, team_game,
    efficiency count)}, {'vol' / 'eff': {search: capped weights}}). `counts` (efficiency_counts): quantity -> {(game_id, player_id or team): the
    ratio's denominator}; a match with an efficiency expectation and no count is an error, never a silent full weight.

    A search is no_match when its similarity search was (4c.3), when none of its matches has a comp-free expectation, or when its n_eff over the
    matches with one stays below MIN_NEFF. The efficiency side of a matching search can be too thin on its own: shift_eff = 0 and nomatch_eff = True."""
    qv, qe = MARKET_QUANTITIES[market]
    is_player = tg.player_id is not None
    vals, details, sets = {}, [], {}
    for s in SEARCHES:
        summ = res[s].summary
        vals[f"best_sim_{s}"] = summ["best_similarity"]
        mt = res[s].matches
        if summ["applicable"] and not summ["no_match"] and mt is not None and mt.height > 0:
            ent = mt["obs_player_id"].to_list() if is_player else mt["obs_team"].to_list()
            keys = list(zip(mt["obs_game_id"].to_list(), ent))
            zv = np.array([zl.get(qv, {}).get(k, np.nan) for k in keys], dtype=float)
            ze = np.array([zl.get(qe, {}).get(k, np.nan) for k in keys], dtype=float) if qe else np.full(len(keys), np.nan)
            tgid = np.array([f"{g}|{t}" for g, t in zip(mt["obs_game_id"].to_list(), mt["obs_team"].to_list())])
            ce = np.full(len(keys), np.nan)
            if np.isfinite(ze).any():
                if counts is None or qe not in counts:
                    raise ValueError(f"efficiency matches of {market} need their count (efficiency_counts), none given for {qe}")
                ce = np.array([counts[qe].get(k, np.nan) for k in keys], dtype=float)
                bad = np.isfinite(ze) & ~(ce > 0)
                if bad.any():
                    raise ValueError(f"{int(bad.sum())} {qe} matches have an efficiency expectation but no positive count, e.g. {keys[int(np.flatnonzero(bad)[0])]}")
            sets[s] = (mt, zv, ze, tgid, ce)
    wv, nv, alive_v, frozen = _side_weights(sets, 1, min_neff)
    we, ne, alive_e, _ = _side_weights({s: sets[s] for s in sorted(alive_v)}, 2, min_neff) if qe else ({}, {}, set(), [])
    caps = {"vol": {s: wv.get(s, np.zeros(sets[s][0].height)) for s in sets},
            "eff": {s: we.get(s, np.zeros(sets[s][0].height)) for s in sets}}
    for s in SEARCHES:
        summ = res[s].summary
        reason = summ["reason"] if summ["no_match"] else None
        if s in sets and not np.isfinite(sets[s][1]).any():
            reason = "no_expectations"                     # matches exist, but none has a comp-free expectation to standardise against
        elif s in sets and s not in alive_v:
            reason = "n_eff_with_expectations_below_minimum"
        nomatch = (not summ["applicable"]) or summ["no_match"] or reason is not None
        sv = shift_value(wv[s], sets[s][1], sets[s][3])[0] if not nomatch else 0.0
        se = shift_value(we[s], sets[s][2], sets[s][3])[0] if (not nomatch and qe and s in alive_e) else 0.0
        nomatch_eff = (nomatch or s not in alive_e) if qe else None
        vals.update({f"shift_vol_{s}": sv, f"shift_eff_{s}": se if qe else None, f"n_eff_{s}": float(nv.get(s, 0.0)), f"nomatch_{s}": bool(nomatch),
                     f"n_eff_eff_{s}": float(ne.get(s, 0.0)) if qe else None, f"nomatch_eff_{s}": nomatch_eff})
        # 4c.5: the observed / derived / estimated share of the feature weight behind the search's matches, and their completeness penalty, averaged with
        # the weights the volume shift uses (null when the search does not match)
        mt = sets[s][0] if (s in sets and not nomatch) else None
        top = res[s].summary.get("top_any") or []
        for j, (col, name) in enumerate([(f"share_{t[:3]}", f"share_{t[:3]}_{s}") for t in TAGS] + [("completeness_penalty", f"completeness_{s}")]):
            if mt is not None and col in mt.columns:
                vals[name] = _wmean(wv[s], mt[col].to_numpy())
            else:                                     # no match: the search's closest past games, by their final weight (plan: for every search and target)
                vals[name] = _wmean(np.array([t[1] for t in top]), np.array([t[2 + j] for t in top])) if top else None
        details.append(dict(search=s, applicable=summ["applicable"], n_matches=summ["n_matches"], n_eff_similarity=summ["n_eff"],
                            n_with_expectation_vol=int(np.isfinite(sets[s][1]).sum()) if s in sets else 0,
                            n_with_expectation_eff=int(np.isfinite(sets[s][2]).sum()) if s in sets else 0,
                            n_eff_vol=float(nv.get(s, 0.0)), n_eff_eff=float(ne.get(s, 0.0)), nomatch=bool(nomatch), nomatch_eff=nomatch_eff, reason=reason,
                            team_cap_frozen=",".join(frozen)))
    return vals, details, sets, caps


def _hits(res: SearchResult) -> list:
    """(comparable id, final weight, 0) of every match of a search, for retrieval_change."""
    mt = res.matches
    if mt is None or mt.height == 0:
        return []
    ids = [f"{g}|{t}|{p or ''}" for g, t, p in zip(mt["obs_game_id"].to_list(), mt["obs_team"].to_list(), mt["obs_player_id"].to_list())]
    return list(zip(ids, mt["final_weight"].to_list(), [0.0] * len(ids)))


def comp_shifts(pool: Pool, targets: list, zl: dict, keep_matches: bool = True, sim_threshold: float = cs.SIM_THRESHOLD,
                min_neff: float = cs.MIN_NEFF, compare: bool = True, counts: dict | None = None) -> tuple:
    """4c.4 for every (target, market): (features, matches, detail, retrieval).

    features: one row per target and market with shift_vol_S1..S5, shift_eff_S1..S5, n_eff_S1..S5, nomatch_S1..S5, best_sim_S1..S5 (plan 4c.4.5),
    from the searches on the healthy target vectors (SEARCH_VERSION), plus n_eff_eff / nomatch_eff (the efficiency side) and, from 4c.5, share_obs / share_der /
    share_est and completeness per search: the observed / derived / estimated share of the feature weight behind the matches and their completeness
    penalty, averaged with the weights of the volume shift (null without a match).
    matches: the per-match table (similarity, recency, continuity, quality, final weight, the capped weights and z for volume and efficiency).
    detail: per target, market and search: how many matches had an expectation, n_eff before and after the caps, and why a search was no_match.
    retrieval (compare=True): per target, market and search, how the comparables change when the same search runs on the LINEUP-ADJUSTED target
    vector (plan 4c.1.4; the comparison since the decision of 2026-10-09): the overlap of the top RETRIEVAL_TOP_K matches, the shared weight mass and the
    change in the volume / efficiency shift (adjusted - healthy),
    and overlap_closest_10: the overlap of the 10 closest past games by final weight whatever the threshold (informative when a search has no match).
    Where the two target vectors are identical the adjusted run is not repeated (overlap 1, change 0).
    A match counts toward a shift only when its own comp-free expectation exists (walkforward_predictions); a search whose matches have none is no_match.
    n_eff counts historical team-games (cluster_neff). `counts` (efficiency_counts): an efficiency match's weight is multiplied by its count (decision of
    2026-10-09); required as soon as a match has an efficiency expectation."""
    feats, mrows, drows, crows = [], [], [], []
    last = None
    for tg, markets in _target_rows(targets, getattr(pool, "choice", None)):
        if (tg.game_id, tg.team) != last:
            pool.clear_cache()
            last = (tg.game_id, tg.team)
        top = RETRIEVAL_TOP_K                         # the closest games feed the healthy-vs-adjusted comparison and the no-match input shares
        res = pool.search(tg, SEARCHES, keep=True, sim_threshold=sim_threshold, min_neff=min_neff, version=SEARCH_VERSION, top_any=top)
        differs = compare and pool.adjusted_differs(pool.idx[(tg.game_id, tg.team)], cs.MARKET_UNITS[tg.market])
        res_a = (pool.search(tg, SEARCHES, keep=True, sim_threshold=sim_threshold, min_neff=min_neff, version=COMPARISON_VERSION, top_any=top)
                 if differs else res)
        for m in markets:
            ident = dict(season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team, player_id=tg.player_id, market=m)
            vals, det, sets, caps = _market_shifts(res, tg, m, zl, min_neff, counts)
            feats.append(dict(season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team, opponent=tg.opponent, player_id=tg.player_id, market=m) | vals)
            drows += [ident | d for d in det]
            if keep_matches:
                for s, (mt, zv, ze, _, ce) in sets.items():
                    mrows.append(mt.with_columns(market=pl.lit(m), z_vol=pl.Series(zv), z_eff=pl.Series(ze), eff_count=pl.Series(ce, dtype=pl.Float64),
                                                 weight_capped_vol=pl.Series(caps["vol"][s]),
                                                 weight_capped_eff=pl.Series(caps["eff"][s]) if MARKET_QUANTITIES[m][1] else pl.lit(None, dtype=pl.Float64)))
            if compare:
                hvals = vals                                  # the search itself runs on the healthy vectors
                avals = _market_shifts(res_a, tg, m, zl, min_neff, counts)[0] if differs else vals
                for s in SEARCHES:
                    hh, ha = _hits(res[s]), _hits(res_a[s])
                    rc = retrieval_change(hh, ha, min(RETRIEVAL_TOP_K, max(len(ha), len(hh))))      # one side empty: overlap 0; both: undefined
                    th = [(t[0], t[1], 0.0) for t in res[s].summary.get("top_any", [])]
                    ta = [(t[0], t[1], 0.0) for t in res_a[s].summary.get("top_any", [])]
                    rca = retrieval_change(th, ta, min(RETRIEVAL_TOP_K, max(len(ta), len(th))))
                    crows.append(dict(ident, search=s, adjusted_differs=bool(differs), n_matches_adjusted=len(ha), n_matches_healthy=len(hh),
                                      top_k=rc["k"], overlap_top_k=rc["overlap_top_k"], weight_mass_shared=rc["weight_mass_shared"] if ha or hh else float("nan"),
                                      overlap_closest_10=rca["overlap_top_k"],
                                      nomatch_adjusted=avals[f"nomatch_{s}"], nomatch_healthy=hvals[f"nomatch_{s}"],
                                      shift_vol_adjusted=avals[f"shift_vol_{s}"], shift_vol_healthy=hvals[f"shift_vol_{s}"],
                                      shift_vol_change=avals[f"shift_vol_{s}"] - hvals[f"shift_vol_{s}"],
                                      shift_eff_adjusted=avals[f"shift_eff_{s}"], shift_eff_healthy=hvals[f"shift_eff_{s}"],
                                      shift_eff_change=(avals[f"shift_eff_{s}"] - hvals[f"shift_eff_{s}"]) if avals[f"shift_eff_{s}"] is not None else None))
    features = pl.DataFrame(feats, infer_schema_length=None)
    if features.height:                               # explicit types: a column can be null in every row (S2 / S4 never match at 0.70)
        features = features.with_columns(
            *[pl.col(f"{c}_{s}").cast(pl.Float64) for s in SEARCHES for c in ("shift_vol", "shift_eff", "n_eff", "best_sim", "n_eff_eff", "share_obs", "share_der",
                                                                               "share_est", "completeness")],
            *[pl.col(f"{c}_{s}").cast(pl.Boolean) for s in SEARCHES for c in ("nomatch", "nomatch_eff")])
    matches = pl.concat(mrows, how="diagonal_relaxed") if mrows else pl.DataFrame()
    retrieval = pl.DataFrame(crows, infer_schema_length=None) if crows else pl.DataFrame()
    return features, matches, pl.DataFrame(drows, infer_schema_length=None), retrieval


def shift_summary(feats: pl.DataFrame, detail: pl.DataFrame) -> pl.DataFrame:
    """Per season, market and search: targets, the no-match rate (and its reasons), the mean absolute volume / efficiency shift over matched targets and
    the mean n_eff after the caps."""
    rows = []
    for s in SEARCHES:
        g = feats.group_by("season", "market").agg(
            n_targets=pl.len(), nomatch_rate=pl.col(f"nomatch_{s}").mean(),
            mean_abs_shift_vol=pl.col(f"shift_vol_{s}").filter(~pl.col(f"nomatch_{s}")).abs().mean(),
            mean_abs_shift_eff=pl.col(f"shift_eff_{s}").filter(~pl.col(f"nomatch_{s}")).abs().mean(),
            mean_n_eff=pl.col(f"n_eff_{s}").filter(~pl.col(f"nomatch_{s}")).mean()).with_columns(search=pl.lit(s))
        rows.append(g)
    out = pl.concat(rows, how="diagonal_relaxed")
    reasons = (detail.filter(pl.col("nomatch")).group_by("season", "market", "search", "reason").agg(n=pl.len())
               .group_by("season", "market", "search").agg(reasons=pl.struct("reason", "n").sort_by("reason")))
    out = out.join(reasons.with_columns(pl.col("reasons").map_elements(lambda x: json.dumps({r["reason"] or "": r["n"] for r in x}), return_dtype=pl.String)),
                   on=["season", "market", "search"], how="left")
    return out.sort("season", "market", "search")


def retrieval_summary(retrieval: pl.DataFrame) -> pl.DataFrame:
    """Per season, market and search: the share of targets whose lineup-adjusted vector differs from the healthy one, and over those, the mean overlap of
    the top matches, the mean shared weight, how often the no-match verdict flips, and the mean absolute change in the volume / efficiency shift."""
    if retrieval.height == 0:
        return pl.DataFrame()
    d = pl.col("adjusted_differs")
    return (retrieval.group_by("season", "market", "search")
            .agg(n_targets=pl.len(), share_adjusted_differs=d.mean(),
                 mean_overlap_top_k=pl.col("overlap_top_k").filter(d).fill_nan(None).mean(),
                 mean_overlap_closest_10=pl.col("overlap_closest_10").filter(d).fill_nan(None).mean(),
                 mean_weight_mass_shared=pl.col("weight_mass_shared").filter(d).fill_nan(None).mean(),
                 share_nomatch_flips=(pl.col("nomatch_adjusted") != pl.col("nomatch_healthy")).filter(d).mean(),
                 mean_abs_shift_vol_change=pl.col("shift_vol_change").filter(d).abs().mean(),
                 mean_abs_shift_eff_change=pl.col("shift_eff_change").filter(d).abs().mean())
            .sort("season", "market", "search"))


SENSITIVITY_THRESHOLDS = (0.4, 0.5, 0.6, 0.7)     # analysis only: the no-match memo's table (the threshold in use stays cs.SIM_THRESHOLD)


def nomatch_sensitivity(summary: pl.DataFrame, thresholds=SENSITIVITY_THRESHOLDS, min_neffs=(1.0, 2.0, 3.0)) -> pl.DataFrame:
    """Per season, market family (its first market), search, threshold and MIN_NEFF: the share of targets that would be no_match, from the
    `sensitivity` column of run_search_log (matches and n_eff recounted at each threshold over the same similarities). Nothing here changes a constant."""
    s = summary.filter(pl.col("applicable"))
    rows = []
    for (season, market, search), g in s.group_by("season", "market", "search", maintain_order=True):
        sens = [json.loads(x) for x in g["sensitivity"]]
        best = g["best_similarity"].to_numpy()
        for th in thresholds:
            t = f"{th:g}"                                # the key run_search_log wrote; a search without similarities has none: no match at any threshold
            for mn in min_neffs:
                nm = [not (b == b and b >= th and t in d and meets_min_neff(d[t][1], mn)) for b, d in zip(best, sens)]
                rows.append(dict(season=season, market=market, search=search, threshold=float(th), min_neff=float(mn), n_targets=g.height,
                                 no_match_rate=float(np.mean(nm)), median_n_eff=float(np.median([d.get(t, [0, 0.0])[1] for d in sens]))))
    return pl.DataFrame(rows).sort("season", "market", "search", "threshold", "min_neff")

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
def league_z(values: np.ndarray, den: np.ndarray, keys: np.ndarray, rel: float = LEAGUE_MIN_REL_DEN) -> tuple:
    """z-scores of `values` (n, F) against the mean / SD of the rows sharing the same key (season-week), among rows whose denominator is at least
    `rel` x the median positive denominator of that key for the feature. Returns (z clipped to +-Z_CLIP, mean, sd), the last two (n, F) broadcast
    per row. NaN where the league has fewer than MIN_LEAGUE_ENTITIES eligible rows or no spread."""
    z = np.full(values.shape, np.nan)
    mu = np.full(values.shape, np.nan)
    sd = np.full(values.shape, np.nan)
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
        s = np.sqrt(var)
        s = np.where((cnt >= MIN_LEAGUE_ENTITIES) & (s > 0), s, np.nan)
        m = np.where(np.isnan(s), np.nan, m)
        mu[idx], sd[idx] = m, s
        z[idx] = np.clip((v - m) / s, -Z_CLIP, Z_CLIP)
    return z, mu, sd


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
    detail: pl.DataFrame | None = None       # 4a expected shares (sim.inputs._detail_and_exit): game_id, team, player_id, group, exp_*, b_*, p_out
    completeness: pl.DataFrame | None = None


def _epoch_days(col: pl.Series) -> np.ndarray:
    return col.dt.epoch("d").to_numpy().astype(np.int64)


def _lineup_array(games: pl.DataFrame, lineups: pl.DataFrame) -> np.ndarray:
    j = games.select("team", "season", "week").join(lineups, on=["team", "season", "week"], how="left")
    return np.vstack([j[c].fill_null(-1).to_numpy().astype(np.int64) for c in LINEUP_COMPONENTS])


def week_cutoffs(games: pl.DataFrame) -> pl.DataFrame:
    """The first kickoff of each (season, week): the harness cutoff. A window for a game in that week sees only games before it."""
    return games.group_by("season", "week").agg(cutoff=pl.col("gameday").min())


def team_windows(games: pl.DataFrame, lineups: pl.DataFrame, wide: pl.DataFrame, keys: list, variants=VARIANTS) -> dict:
    """{variant: (rate, den)} (n_games x n_keys) of the team's own pooled window values as of each of its games (rows follow `games`)."""
    tw = games.select("game_id", "team", "season", "week", "gameday", "team_game_num").join(wide, on=["game_id", "team"], how="left")
    assert tw.height == games.height
    cut = games.join(week_cutoffs(games), on=["season", "week"], how="left")["cutoff"]
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
    """(game_id, team, player_id, position) of every player who played: a ledger row (a touch) or an offensive snap, at QB / RB / WR / TE."""
    from features import comps_ledger as L
    led = inp.player.select("game_id", "team", "player_id", "position")
    parts = [led]
    if inp.snaps is not None:
        sn = (inp.snaps.select("game_id", "team", player_id="gsis_id").join(inp.games.select("game_id", "team", "season", "week"), on=["game_id", "team"], how="inner"))
        parts.append(L.attach_position(sn, "player_id", inp.positions, extra=()).select("game_id", "team", "player_id", "position"))
    pop = pl.concat(parts).unique(["game_id", "team", "player_id"], keep="first", maintain_order=True)
    return pop.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"])).sort("game_id", "team", "player_id")


def build_queries(inp: Inputs) -> pl.DataFrame:
    """The (game, team, player) pairs whose window values are needed: every player-game in the ledger; the players of a team's previous six games
    who did not play (absent regulars: their absence is what the lineup correction measures); and the players the 4a layer lists for the game."""
    led = inp.player.select("game_id", "team", "player_id")
    g = inp.games.select("game_id", "team", "season", "week").with_row_index("_r")
    seq = inp.games.sort("team", "gameday").with_columns(seq=pl.int_range(pl.len()).over("team")).select("game_id", "team", "seq")
    lseq = led.join(seq, on=["game_id", "team"], how="inner")
    regs = pl.concat([lseq.select("team", "player_id", seq=pl.col("seq") + o) for o in range(1, 7)]).unique()
    absent = regs.join(seq, on=["team", "seq"], how="inner").select("game_id", "team", "player_id")
    parts = [led, absent, population_table(inp).select("game_id", "team", "player_id")]
    if inp.detail is not None:
        parts.append(inp.detail.filter(pl.col("player_id") != "rest").select("game_id", "team", "player_id"))
    q = pl.concat(parts).unique().join(inp.games.select("game_id", "team", "season", "week"), on=["game_id", "team"], how="inner")
    q = L_attach(q, inp)
    return q.sort("player_id", "season", "week", "game_id", "team")      # adjacent per player: build_player_windows walks them in runs


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
        j = (df.join(lin_tbl, on=["team", "season", "week"], how="left")
             .join(inp.slots.rename({"gsis_id": id_col, "family": "fam_s"}), on=[id_col, "season", "week"], how="left"))
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

    qdf = queries.join(cut, on=["season", "week"], how="left").join(
        inp.games.select("game_id", "team", "team_game_num"), on=["game_id", "team"], how="left")
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
    q = queries.join(played, on=["game_id", "team", "player_id"], how="left").with_columns(pl.col("active").fill_null(False))
    zero = ("exp_carry", "exp_target", "exp_dropback", "b_carry", "b_target", "b_dropback")
    if inp.detail is not None:
        d = inp.detail.filter(pl.col("player_id") != "rest").select("game_id", "team", "player_id", *zero)
        has = inp.detail.select("game_id", "team").unique().with_columns(has_detail=pl.lit(True))
        q = (q.join(d, on=["game_id", "team", "player_id"], how="left").join(has, on=["game_id", "team"], how="left")
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
    cut = g.join(week_cutoffs(g), on=["season", "week"], how="left")["cutoff"]
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
    """Archetype vectors for every player-game in the archetype's population: the player played at one of its positions (ARCHETYPE_POSITIONS)."""
    K = cs.SHRINK_K
    q = pw.queries.with_row_index("qi").select("qi", "game_id", "team", "player_id", "season", "week")
    cut = week_cutoffs(inp.games)
    m = population_table(inp).join(q, on=["game_id", "team", "player_id"], how="inner").join(cut, on=["season", "week"], how="left")
    kidx = {k: i for i, k in enumerate(pw.keys)}
    out = []
    for unit, positions in ARCHETYPE_POSITIONS.items():
        rows = m.filter(pl.col("position").is_in(list(positions)))
        qi = rows["qi"].to_numpy()
        meta = rows.select("game_id", "team", "player_id", "season", "week", as_of="cutoff")
        wk = (rows["season"].to_numpy() * 100 + rows["week"].to_numpy()).astype(np.int64)
        for space in stored_spaces(unit):
            keys = feature_keys(unit, space)
            cols = [kidx[k] for k in keys]
            static = np.array([k.split(".")[1] in NO_SHRINK for k in keys])
            for vi, v in enumerate(variants):
                rate, den = pw.rate[vi][qi][:, cols].astype(np.float64), pw.den[vi][qi][:, cols].astype(np.float64)
                num = np.nan_to_num(rate) * den
                mu = np.full(rate.shape, np.nan)
                for w in np.unique(wk):                                        # league pooled mean of the week, from earlier games only
                    r = wk == w
                    tot = den[r].sum(axis=0)
                    mu[r] = np.divide(num[r].sum(axis=0), tot, out=np.full(tot.shape, np.nan), where=tot > 0)
                val = np.where(static[None, :], rate, (num + K * mu) / (den + K))
                val = np.where(np.isfinite(val) & (den > 0), val, np.nan)
                z = league_z(val, den, wk)[0]
                z = np.where(den > 0, z, np.nan)
                out.append(_pack(meta, z, val, den, version="player", window=vname(v), unit=unit, space=space))
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
    return dict(k=k, overlap_top_k=len(a & b) / k if k else float("nan"), weight_mass_shared=float(sum(min(wa.get(i, 0.0), wb.get(i, 0.0)) for i in set(wa) | set(wb))),
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
                  snaps=snaps, completeness=L.completeness(plays, games, team, pfr_pass, snaps))


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
    deltas = lineup_deltas(pw, pw_share, share_tables(inp, pw.queries), tw, team_keys, games_index, inp.games.height)
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


def group_distance(A: np.ndarray, B: np.ndarray, groups: list, w: np.ndarray, q: np.ndarray, group_w: np.ndarray | None = None) -> Distance:
    """The distance of unit_distance for any feature layout: `groups` is a list of column-index lists, w and q per column (NaN = missing)."""
    wq = w * q
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    ma, mb = ~np.isnan(A), ~np.isnan(B)
    a0, b0 = np.where(ma, A, 0.0), np.where(mb, B, 0.0)
    shape = (A.shape[0], B.shape[0])
    num_u, den_u, got, mass, mass_w = (np.zeros(shape) for _ in range(5))
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
    total = float(wq.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        d2 = np.where(den_u > 0, num_u / den_u, np.nan)
        comp = (1.0 - mass / total) if total > 0 else np.full(shape, np.nan)
        qual = np.where(mass_w > 0, mass / np.maximum(mass_w, 1e-12), np.nan)
    return Distance(d2, comp, got.astype(int), qual)


def unit_distance(A: np.ndarray, B: np.ndarray, unit: str, space: str, quality: dict | None = None, weights: dict | None = None,
                  group_weights: dict | None = None) -> Distance:
    """d2 between every row of A and every row of B (z-vectors of the same unit, window and space; NaN = missing).

    d2_g = sum_f(w_f q_f (a_f - b_f)^2) / sum_f(w_f q_f) over the features of group g present on BOTH sides (a feature missing on either side has
    weight 0). The unit d2 is the weighted mean of the d2_g over the groups that could be compared (start: equal weights). The completeness penalty is
    the share of the unit's total w x q mass that was missing on either side. Computed with matrix products, so it scales to a large pool."""
    feats = unit_features(unit, space)
    w = np.array([(weights or {}).get(f.name, cs.FEATURE_WEIGHT_DEFAULT) for f in feats], dtype=float)
    groups = group_columns(unit, space)
    gw = np.array([(group_weights or {}).get(name, cs.GROUP_WEIGHT_DEFAULT) for name, _ in groups], dtype=float)
    return group_distance(A, B, [cols for _, cols in groups], w, feature_quality(unit, space, quality), gw)


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
    neighbours are the unit's vectors of earlier weeks, same window and space, from the pool versions (`pool_version`), excluding the target's own team /
    player (a team's consecutive games share window data, so they are not independent matches). Distance D = sqrt(d2), so sigma is on the scale that
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
    cut = games.join(week_cutoffs(games), on=["season", "week"], how="left")["cutoff"]
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
        for unit in TEAM_UNITS:
            # a target-version row that does not exist (a game the 4a layer has no expectations for) falls back to the healthy vector
            versions = [target_version(unit)] + (["healthy"] if target_version(unit) != "healthy" else [])
            for versions_, store in ((versions, self.T), ([pool_version(unit)], self.P)):
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
                    meta = order
                blocks[space] = (_list_matrix(df, "z", nf), df["complete"].to_numpy())
            pkey = (meta["season"].to_numpy() * 100 + meta["week"].to_numpy()).astype(np.int64)
            prow = np.array([self.idx[(g, t)] for g, t in zip(meta["game_id"].to_list(), meta["team"].to_list())], dtype=np.int64)
            sl = meta.join(slots.rename({"gsis_id": "player_id"}), on=["player_id", "season", "week"], how="left")
            self.pl[unit] = dict(blocks=blocks, key=pkey, row=prow, pid=meta["player_id"].to_numpy(), team=meta["team"].to_numpy(),
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

    def unit_sims(self, unit: str, kind: str, row: int, k: int, key: int) -> tuple:
        """(sim, completeness, quality) arrays over the first k team-game rows for `unit` against the vector of team-game `row` (kind 'off': target version
        vs pool version; 'def': defense units, one version)."""
        ck = ("u", unit, kind, row)
        if ck in self._cache:
            sim, comp, qual = self._cache[ck]
            return sim[:k], comp[:k], qual[:k]
        spaces = stored_spaces(unit)
        T, P = self.T[unit], self.P[unit]
        sim = np.full(self.n, np.nan); comp = np.full(self.n, np.nan); qual = np.full(self.n, np.nan)
        res = {}
        for space in spaces:
            tv = T[space][0][row][None, :]
            if np.isnan(tv).all():
                continue
            d = unit_distance(tv, P[space][0][:self._kmax(key)], unit, space)
            s = similarity(d.d2[0], self._sigma(unit, space, key))
            res[space] = (s, d.completeness[0], d.quality[0])
        kk = self._kmax(key)
        if not res:
            self._cache[ck] = (sim, comp, qual)
            return sim[:k], comp[:k], qual[:k]
        use_ext = np.zeros(kk, dtype=bool)
        if "extended" in res and "base" in spaces:
            use_ext = T["extended"][1][row] & P["extended"][1][:kk]
        elif "extended" in res:
            use_ext = T["extended"][1][row] & P["extended"][1][:kk]
        for j, name in enumerate(("sim", "comp", "qual")):
            arr = (sim, comp, qual)[j]
            base = res["base"][j] if "base" in res else np.full(kk, np.nan)
            ext = res["extended"][j] if "extended" in res else np.full(kk, np.nan)
            arr[:kk] = np.where(use_ext, ext, base)
        self._cache[ck] = (sim, comp, qual)
        return sim[:k], comp[:k], qual[:k]

    def _kmax(self, key: int) -> int:
        return int(np.searchsorted(self.key, key, side="left"))

    def clear_cache(self):
        self._cache = {}

    def profile_sims(self, g: int, o: int, k: int, key: int) -> dict:
        """S5 over the first k team-game matchups: {'off': (sim, completeness, quality), 'def': (...)} and whether the target misses a required FTN feature."""
        out, missing = {}, False
        for side, tv, Pm in (("off", self.prof_off_t[g], self.prof_off_p[:k]), ("def", self.prof_def[o], self.prof_def[self.opp_row[:k]])):
            missing |= bool(np.isnan(tv[self.req5[side]]).any())
            if np.isnan(tv).all():
                out[side] = (np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan))
                continue
            d = group_distance(tv[None, :], Pm, self.groups5[side], np.ones(len(tv)), self.q5[side])
            sim = np.where(self.admit5[:k], similarity(d.d2[0], self.sig5.get((side, int(key)), float("nan"))), np.nan)   # EXTENDED-only pool
            out[side] = (sim, d.completeness[0], d.quality[0])
        return out, missing

    # ---- the five searches
    def search(self, tg: Target, which=("S1", "S2", "S3", "S4", "S5"), keep: bool = True, sim_threshold: float = cs.SIM_THRESHOLD,
               min_neff: float = cs.MIN_NEFF) -> dict:
        """Run the searches for one target on the lineup-adjusted target vectors; returns {search: SearchResult}.

        Observations are past (key < the target week) team-games, or player-games for a player market. The similarity of an observation is one-sided for
        S1 (the defenses faced) and two-sided for S2, S4, S5 (offense similarity x defense similarity; build plan 4c.4); S3 multiplies the player's
        archetype similarity by the defense faced. A side's similarity is the mean of its units' similarities. A match is an observation whose similarity
        is at least `sim_threshold`; its weight is  similarity x recency x continuity x data-quality  (separate columns in `matches`; recency and continuity
        from features/weights.py for the market). The search is no_match when its best similarity is below the threshold or its n_eff below
        `min_neff`: shift = 0 and the widened-uncertainty flag is set."""
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
            pop = self.pl[arch[0]]
            kp = int(np.searchsorted(pop["key"], key, side="left"))
            obs_row, obs_pid, obs_team = pop["row"][:kp], pop["pid"][:kp], pop["team"][:kp]
            obs_slot, obs_fam = pop["slot"][:kp], pop["fam"][:kp]
            tpos = np.flatnonzero((pop["pid"] == tg.player_id) & (pop["row"] == g))
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
            s, c, q = self.unit_sims(unit, kind, row, k, key)
            return s[idx], c[idx], q[idx]
        def arch_comp():
            if not is_player or tvec is None:
                return None
            unit = arch[0]
            spaces = stored_spaces(unit)
            res = {}
            for space in spaces:
                d = unit_distance(tvec[space], pop["blocks"][space][0][:kp], unit, space)
                res[space] = (similarity(d.d2[0], self._sigma(unit, space, key)), d.completeness[0], d.quality[0])
            use_ext = (pop["blocks"]["extended"][1][:kp] & t_ext) if "extended" in pop["blocks"] else np.zeros(kp, dtype=bool)
            nan = np.full(kp, np.nan)
            pick = lambda j: np.where(use_ext, res["extended"][j] if "extended" in res else nan, res["base"][j] if "base" in res else nan)
            return pick(0), pick(1), pick(2)
        ac = arch_comp() if arch else None

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
            # which observations the search looks at
            if sname == "S1":
                sel = (obs_pid == tg.player_id) if is_player else (obs_team == tg.team)
            elif sname == "S2":
                sel = obs_def_team == tg.opponent
            elif sname in ("S3", "S4"):
                sel = (obs_pid != tg.player_id) & (obs_def_team != tg.opponent) if is_player else ((obs_team != tg.team) & (obs_def_team != tg.opponent))
            else:
                sel = np.ones(len(obs_row), dtype=bool)
            idx = np.flatnonzero(sel)
            if sname == "S5":
                prof, miss = self.profile_sims(g, o, k, key)
                comps_sel = {f"matchup_{'offense' if side == 'off' else 'defense'}": (side, tuple(a[obs_row[idx]] for a in prof[side])) for side in ("off", "def")}
                sim_off, sim_def = side_mean(comps_sel, ("off",)), side_mean(comps_sel, ("def",))
                combined = sim_off * sim_def
                if miss:
                    out[sname] = SearchResult(sname, self._summary(tg, sname, combined, comps_sel, None, False, "missing_required_ftn_features", sim_threshold, min_neff,
                                                                   best_override=float(np.nanmax(combined)) if np.isfinite(combined).any() else float("nan")))
                    continue
            else:
                allc = components(sname)
                comps_sel = {u: (kd, (c[0][idx], c[1][idx], c[2][idx])) for u, (kd, c) in allc.items()}
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
            out[sname] = SearchResult(sname, *self._finish(tg, sname, idx, obs_row, obs_pid, obs_team, combined, comps_sel, rec, cont, quality, completeness, final,
                                                            sim_threshold, min_neff, keep, sim_off, sim_def))
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
                sim_off=None, sim_def=None):
        ok = np.isfinite(combined) & np.isfinite(final)
        best = float(np.nanmax(combined)) if np.isfinite(combined).any() else float("nan")
        match = ok & (combined >= sim_threshold)
        w = final[match]
        n_eff = float(w.sum() ** 2 / (w ** 2).sum()) if match.any() and (w ** 2).sum() > 0 else 0.0
        reason = None
        if not np.isfinite(best):
            reason = "no_similarity"
        elif best < sim_threshold:
            reason = "best_similarity_below_threshold"
        elif n_eff < min_neff:
            reason = "n_eff_below_minimum"
        summary = self._summary(tg, sname, combined, comps, final, reason is None, reason, sim_threshold, min_neff, n_matches=int(match.sum()), n_eff=n_eff)
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
            cols.update(sim_combined=combined[sel], completeness_penalty=completeness[sel], recency_weight=rec[sel], continuity_weight=cont[sel],
                        quality_weight=quality[sel], final_weight=final[sel])
            matches = pl.DataFrame(cols, schema_overrides={"target_player_id": pl.String, "obs_player_id": pl.String}, strict=False).sort(["sim_combined", "obs_game_id", "obs_team"], descending=[True, False, False])
        return summary, matches


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


def run_search_log(pool: Pool, targets: list, which=("S1", "S2", "S3", "S4", "S5"), sim_threshold: float = cs.SIM_THRESHOLD, min_neff: float = cs.MIN_NEFF) -> pl.DataFrame:
    """One row per (target, search): n_matches, n_eff, best similarity, no_match, reason and the best similarity of every unit. No match rows are not
    stored here; call Pool.search(..., keep=True) for them."""
    rows = []
    last = None
    for tg in targets:
        if (tg.game_id, tg.team) != last:
            pool.clear_cache()
            last = (tg.game_id, tg.team)
        for sname, res in pool.search(tg, which, keep=False, sim_threshold=sim_threshold, min_neff=min_neff).items():
            r = dict(res.summary)
            ub = r.pop("unit_best")
            r["unit_best"] = json.dumps({k: (None if v != v else round(v, 6)) for k, v in sorted(ub.items())})
            rows.append(r)
    return pl.DataFrame(rows)


def aggregate_search_log(summary: pl.DataFrame, sim_threshold: float = cs.SIM_THRESHOLD) -> pl.DataFrame:
    """Per season, market, search: how often the search returns no_match (and why), and per unit how often the unit alone has no similarity at all or
    nothing above the threshold. Markets of a family are searched once and reported under each market."""
    s = summary.filter(pl.col("applicable"))
    rows = []
    for (season, market, search), g in s.group_by("season", "market", "search", maintain_order=True):
        fam = FAMILY_OF_MARKET[market]
        n = g.height
        base = dict(season=season, search=search, n_targets=n, no_match_rate=float(g["no_match"].mean()), median_best_similarity=float(g["best_similarity"].median() or float("nan")),
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

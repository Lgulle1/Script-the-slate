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
ARCHETYPE_POPULATION = {"rb_archetype": lambda d: (d["position"] == "RB") & ((d["c.carries"] > 0) | (d["c.targets"] > 0)),
                        "receiver_archetype": lambda d: d["position"].is_in(["WR", "TE", "RB"]) & (d["c.targets"] > 0),
                        "qb_archetype": lambda d: (d["position"] == "QB") & (d["c.dropbacks"] > 0)}
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


def build_queries(inp: Inputs) -> pl.DataFrame:
    """The (game, team, player) pairs whose window values are needed: every player-game in the ledger; the players of a team's previous six games
    who did not play (absent regulars: their absence is what the lineup correction measures); and the players the 4a layer lists for the game."""
    led = inp.player.select("game_id", "team", "player_id")
    g = inp.games.select("game_id", "team", "season", "week").with_row_index("_r")
    seq = inp.games.sort("team", "gameday").with_columns(seq=pl.int_range(pl.len()).over("team")).select("game_id", "team", "seq")
    lseq = led.join(seq, on=["game_id", "team"], how="inner")
    regs = pl.concat([lseq.select("team", "player_id", seq=pl.col("seq") + o) for o in range(1, 7)]).unique()
    absent = regs.join(seq, on=["team", "seq"], how="inner").select("game_id", "team", "player_id")
    parts = [led, absent]
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
    """Archetype vectors for every player-game in the archetype's population (rb: an RB with a carry or target; receiver: a WR / TE / RB with a
    target; qb: a QB with a dropback)."""
    K = cs.SHRINK_K
    led = inp.player.select("game_id", "team", "player_id", "position", "c.carries", "c.targets", "c.dropbacks")
    q = pw.queries.with_row_index("qi").select("qi", "game_id", "team", "player_id", "season", "week")
    cut = week_cutoffs(inp.games)
    m = led.join(q, on=["game_id", "team", "player_id"], how="inner").join(cut, on=["season", "week"], how="left")
    kidx = {k: i for i, k in enumerate(pw.keys)}
    out = []
    for unit, pop in ARCHETYPE_POPULATION.items():
        rows = m.filter(pop(m))
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

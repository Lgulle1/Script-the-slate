"""Role-context and played-hurt features for the projection models (yours, 2026-10-19: Allen, Coker).

The depth-chart role never changes when a starter is out: Allen is "RB2" in every game of 2026, including the three he started because Hall was out. So a
player's average over all his games mixes two jobs. These features compare the game being predicted with each past game by who was missing around the
player, using only injury-report information posted before the game:

  M (missing usage)   for a player at a game: the summed trailing usage (RB: carries + targets, WR/TE: targets, QB: pass attempts, recency-weighted over his
                      last 8 games) of his same-position teammates who were on the report that week, each times a chance of missing: Out 1.0, Doubtful 0.75,
                      Did Not Practice with no game status 0.5, Questionable 0.25. His own report entry is left out.
  ctx_<stat>          his recency-weighted mean of <stat> over past games, each past game also weighted by exp(-|M_then - M_now| / 6): games played under the
                      same teammate situation count, games under a different one barely do. ctx_mass, ctx_n: how much history that leaves.
  miss_t, miss_last   M now and M in his last game.
  hd_<stat>           his recency-weighted mean with every past game he played while Questionable or Doubtful counted at half weight; hurt_t is his own status
                      now (0 none, 1 Questionable, 2 Doubtful), hurt_last3 how many of his last 3 games he played hurt.

Switched on for a build by setting features.volume_features.ROLE_CTX to the result of build_tables (otherwise nothing changes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import polars as pl

import config

STATUS_W = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}
HURT_CODE = {"Questionable": 1, "Doubtful": 2}
USAGE = {"RB": ("carries", "targets"), "WR": ("targets",), "TE": ("targets",), "QB": ("attempts",)}
SCALE = 6.0          # usage units: about a third of a lead back's touches
HURT_W = 0.5
LAST_N, HALF = 8, 6.0


@dataclass
class RoleTables:
    M: dict = field(default_factory=dict)       # (player_id, game_id) -> missing teammate usage at that game
    H: dict = field(default_factory=dict)       # (player_id, game_id) -> 0 / 1 Questionable / 2 Doubtful
    mode: str = "both"                          # "ctx", "hurt" or "both"


def injury_lookup(raw_db=None) -> pl.DataFrame:
    """The latest report per (player, season, week), regular season, 2019 on."""
    con = duckdb.connect(str(raw_db or config.RAW_DUCKDB_PATH), read_only=True)
    d = con.execute("SELECT DISTINCT ON (gsis_id, season, week) gsis_id, season, week, team, report_status, practice_status FROM injuries "
                    "WHERE gsis_id IS NOT NULL AND game_type = 'REG' AND season >= 2019 ORDER BY gsis_id, season, week, pulled_at DESC").pl()
    con.close()
    return d.with_columns(season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32))


def miss_weight(status, practice) -> float:
    w = STATUS_W.get(status, 0.0)
    if status is None and practice and practice.startswith("Did Not"):
        w = 0.5
    return w


def _usage(df: pl.DataFrame) -> pl.Series:
    fam = df["family"].to_list()
    cols = {c: df[c].fill_null(0).to_numpy().astype(float) for c in ("carries", "targets", "attempts")}
    out = np.zeros(df.height)
    for i, f in enumerate(fam):
        out[i] = sum(cols[c][i] for c in USAGE.get(f, ()))
    return out


def build_tables(player_log: pl.DataFrame, week_cutoff: dict, inj: pl.DataFrame, mode: str = "both") -> RoleTables:
    """`week_cutoff` maps (season, week) to the week's first kickoff; `inj` is injury_lookup(). Pregame information only."""
    log = player_log.sort("player_id", "gameday")
    log = log.with_columns(usage=pl.Series(_usage(log)))
    by_pid = {}
    for k, g in log.partition_by("player_id", as_dict=True, maintain_order=True).items():
        pid = k[0] if isinstance(k, tuple) else k
        by_pid[pid] = (np.array([d.toordinal() for d in g["gameday"].to_list()]), g["usage"].to_numpy(), g["family"][-1])

    def u_pre(pid, cutoff):
        e = by_pid.get(pid)
        if e is None:
            return None, 0.0
        days, usage, fam = e
        j = int(np.searchsorted(days, cutoff, side="left"))
        if j == 0:
            return fam, 0.0
        lo = max(0, j - LAST_N)
        w = 0.5 ** ((j - 1 - np.arange(lo, j)) / HALF)
        return fam, float((w * usage[lo:j]).sum() / w.sum())

    total, own = {}, {}
    H = {}
    status_of = {}
    for r in inj.iter_rows(named=True):
        key = (r["gsis_id"], r["season"], r["week"])
        status_of[key] = r["report_status"]
        w = miss_weight(r["report_status"], r["practice_status"])
        if w <= 0 or (r["season"], r["week"]) not in week_cutoff:
            continue
        fam, u = u_pre(r["gsis_id"], week_cutoff[(r["season"], r["week"])].toordinal())
        if fam is None or u <= 0:
            continue
        c = w * u
        own[key] = c
        total[(r["team"], r["season"], r["week"], fam)] = total.get((r["team"], r["season"], r["week"], fam), 0.0) + c
    M = {}
    for r in log.select("player_id", "game_id", "team", "season", "week", "family").iter_rows(named=True):
        key = (r["player_id"], r["season"], r["week"])
        M[(r["player_id"], r["game_id"])] = max(0.0, total.get((r["team"], r["season"], r["week"], r["family"]), 0.0) - own.get(key, 0.0))
        H[(r["player_id"], r["game_id"])] = HURT_CODE.get(status_of.get(key), 0)
    return RoleTables(M=M, H=H, mode=mode)


def player_arrays(role: RoleTables, pid, game_ids) -> tuple:
    """(M, H) for every game of one player, in the player's game order."""
    return (np.array([role.M.get((pid, g), 0.0) for g in game_ids]), np.array([role.H.get((pid, g), 0) for g in game_ids]))


def row_features(role: RoleTables, Mv, Hv, w_rec, j, i, X, stats, P) -> dict:
    """Features for target game i of a player with j earlier games (`w_rec` their recency weights, X the stat matrix, P the pooled-ratio triples)."""
    r = {}
    if role.mode in ("ctx", "both"):
        Mi, Mp = Mv[i], Mv[:j]
        w_ctx = w_rec * np.exp(-np.abs(Mp - Mi) / SCALE)
        for c, s in enumerate(stats):
            r[f"ctx_{s}"] = _wavg(w_ctx, X[:j, c])
        for nm, pn, pd_ in P:
            mass = float((w_ctx * pd_[:j]).sum())
            r[f"ctx_{nm}"] = float((w_ctx * pn[:j]).sum() / mass) if mass > 0 else np.nan
        r.update(ctx_mass=float(w_ctx.sum()), ctx_n=float((np.abs(Mp - Mi) < SCALE).sum()), miss_t=float(Mi), miss_last=float(Mp[-1]))
    if role.mode in ("hurt", "both"):
        w_h = w_rec * np.where(Hv[:j] > 0, HURT_W, 1.0)
        for c, s in enumerate(stats):
            r[f"hd_{s}"] = _wavg(w_h, X[:j, c])
        for nm, pn, pd_ in P:
            mass = float((w_h * pd_[:j]).sum())
            r[f"hd_{nm}"] = float((w_h * pn[:j]).sum() / mass) if mass > 0 else np.nan
        r.update(hurt_t=float(Hv[i]), hurt_last3=float((Hv[max(0, j - 3):j] > 0).sum()))
    return r


def empty_features(role: RoleTables, stats, P) -> dict:
    """The same columns, NaN: a player with no history."""
    r = {}
    if role.mode in ("ctx", "both"):
        r.update({f"ctx_{s}": np.nan for s in stats} | {f"ctx_{nm}": np.nan for nm, _, _ in P} | dict(ctx_mass=0.0, ctx_n=0.0, miss_t=np.nan, miss_last=np.nan))
    if role.mode in ("hurt", "both"):
        r.update({f"hd_{s}": np.nan for s in stats} | {f"hd_{nm}": np.nan for nm, _, _ in P} | dict(hurt_t=np.nan, hurt_last3=np.nan))
    return r


def _wavg(w, x):
    m = ~np.isnan(x)
    ws = w[m].sum()
    return float((w[m] * x[m]).sum() / ws) if ws > 0 else np.nan

"""The searches: which past situations are relevant to tonight, and how did players do against their own expectation in them.

Every search answers with a shrunk, weighted mean of past `rel` (actual vs the comp-free expectation, as a fraction; clipped to [REL_LO, REL_HI]) and an
effective sample size. Only games strictly before the target's gameday are used. Each search also has a CONTROL with the similarity removed (recency only):
similarity earns its place only if it beats its own control.

  own  (S1) the player's past games, weighted by how much that game's DEFENSE resembled tonight's (the defense unit of the market).
  dfn  (S2) tonight's defense's past games against players of the same family, weighted by how much that OFFENSE resembled tonight's offense and how much
            that PLAYER resembled tonight's player (archetype).
  int  (S5) league-wide: past games of the same family and market where the (offense, defense) PAIR resembled tonight's pair, and the player resembled him.

A kernel is exp(-d2 / (2 h^2)), d2 = weighted mean squared z difference over the features present on both sides (feature weights per unit; uniform
unless `fw` is given). h = None is the control (every candidate weight 1 before recency).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

import config
from comps2 import archetypes as AR
from comps2 import fingerprints as FP

REL_LO, REL_HI = -1.0, 2.0
MARKET_UNITS = {  # market -> (offense stats, defense stats)
    "rush_att": (FP.UNITS["run"] + FP.UNITS["pace"], FP.UNITS["run"]),
    "rush_yds": (FP.UNITS["run"] + FP.UNITS["pace"], FP.UNITS["run"]),
    "qb_rush_att": (FP.UNITS["run"] + FP.UNITS["pace"], FP.UNITS["run"]),
    "qb_rush_yds": (FP.UNITS["run"] + FP.UNITS["pace"], FP.UNITS["run"]),
    "targets": (FP.UNITS["pass"] + FP.UNITS["pace"], FP.UNITS["pass"]),
    "rec": (FP.UNITS["pass"] + FP.UNITS["pace"], FP.UNITS["pass"]),
    "rec_yds": (FP.UNITS["pass"] + FP.UNITS["pace"], FP.UNITS["pass"]),
    "pass_att": (FP.UNITS["pass"] + FP.UNITS["pace"], FP.UNITS["pass"]),
    "pass_cmp": (FP.UNITS["pass"] + FP.UNITS["pace"], FP.UNITS["pass"]),
    "pass_yds": (FP.UNITS["pass"] + FP.UNITS["pace"], FP.UNITS["pass"]),
}
MARKET_UNITS = {m: (tuple(dict.fromkeys(o)), tuple(dict.fromkeys(d))) for m, (o, d) in MARKET_UNITS.items()}


@dataclass
class Params:
    h_def: float | None = 1.0        # own: kernel width on the defense fingerprint
    h_off: float | None = 1.0        # dfn: on the offense fingerprint
    h_arch: float | None = 1.0       # dfn / int: on the player archetype
    h_pair: float | None = 1.0       # int: on the (offense, defense) pair
    hl_own: float = 12.0             # own: half-life in the player's games
    hl_dfn: float = 16.0             # dfn: half-life in the defense's games
    hl_int_days: float = 365.0       # int: half-life in days
    int_window_days: int = 730
    k0: float = 2.0                  # shrinkage: sum(w) / (sum(w) + k0)
    fw: dict = field(default_factory=dict)   # learned feature weights {"<market>|o_<stat>" or "o_<stat>" / "d_<stat>" / "a_<stat>": weight}; missing -> 1


def _fw(P, market, key):
    return P.fw.get(f"{market}|{key}", P.fw.get(key, 1.0))


def load():
    e = pl.read_parquet(config.PROCESSED_DIR / "comps2_expectations.parquet").filter(pl.col("market").is_in(list(MARKET_UNITS)))
    e = e.with_columns(relc=pl.col("rel").clip(REL_LO, REL_HI), gd=pl.col("gameday").cast(pl.Date).cast(pl.Int32))
    fp = pl.read_parquet(config.PROCESSED_DIR / "comps2_fingerprints.parquet")
    ar = pl.read_parquet(config.PROCESSED_DIR / "comps2_archetypes.parquet")
    off = fp.filter(pl.col("side") == "off").select("game_id", "team", *[pl.col(s).alias(f"o_{s}") for s in FP.STATS])
    dfn = fp.filter(pl.col("side") == "def").select("game_id", pl.col("team").alias("opponent"), *[pl.col(s).alias(f"d_{s}") for s in FP.STATS])
    arc = ar.select("game_id", "player_id", *[pl.col(s).alias(f"a_{s}") for s in AR.ALL])
    e = e.join(off, on=["game_id", "team"], how="left").join(dfn, on=["game_id", "opponent"], how="left").join(arc, on=["game_id", "player_id"], how="left")
    return e.sort("gd", "game_id", "player_id", "market").with_row_index("rid")


def _kernel(X, x, h, w):
    """X (n, k), x (k,): weights exp(-d2 / 2h^2) over features present on both sides; h None -> ones."""
    if h is None:
        return np.ones(len(X))
    m = ~np.isnan(X) & ~np.isnan(x)[None, :]
    diff = np.where(m, X - x[None, :], 0.0)
    ww = m * w[None, :]
    den = ww.sum(1)
    d2 = np.where(den > 0, (ww * diff * diff).sum(1) / np.maximum(den, 1e-9), 4.0)
    return np.exp(-d2 / (2 * h * h))


def _summ(w, r, k0):
    sw = w.sum()
    if sw <= 0:
        return np.nan, 0.0, 0.0
    mean = (w * r).sum() / sw
    return mean * sw / (sw + k0), sw * sw / max((w * w).sum(), 1e-12), sw


def run(e: pl.DataFrame, P: Params, targets: np.ndarray | None = None, searches=("own", "dfn", "int")) -> pl.DataFrame:
    """Components for every row in `targets` (row ids; default: all rows from 2021)."""
    if targets is None:
        targets = e.filter(pl.col("season") >= 2021)["rid"].to_numpy()
    out = {f"{s}_{k}": np.full(len(e), np.nan) for s in searches for k in ("shift", "neff", "sw")}
    rel = e["relc"].to_numpy()
    gd = e["gd"].to_numpy()
    market = e["market"].to_numpy()
    fam = e["family"].to_numpy()
    player = e["player_id"].to_numpy()
    opp = e["opponent"].to_numpy()
    tset = np.zeros(len(e), bool)
    tset[targets] = True
    for m, (ostats, dstats) in MARKET_UNITS.items():
        mi = np.flatnonzero(market == m)
        if not len(mi):
            continue
        O = e[mi].select([f"o_{s}" for s in ostats]).to_numpy().astype(float)
        D = e[mi].select([f"d_{s}" for s in dstats]).to_numpy().astype(float)
        wo = np.array([_fw(P, m, f"o_{s}") for s in ostats])
        wd = np.array([_fw(P, m, f"d_{s}") for s in dstats])
        loc = {r: j for j, r in enumerate(mi)}
        # --- own: the player's past games against similar defenses
        if "own" in searches:
            for pid, idx in _groups(player[mi]).items():
                rows = mi[idx]
                for a, r in enumerate(rows):
                    if not tset[r] or a == 0:
                        continue
                    past = idx[:a][gd[mi[idx[:a]]] < gd[r]]
                    if not len(past):
                        continue
                    ago = len(past) - np.arange(len(past))
                    w = 0.5 ** (ago / P.hl_own) * _kernel(D[past], D[loc[r]], P.h_def, wd)
                    out["own_shift"][r], out["own_neff"][r], out["own_sw"][r] = _summ(w, rel[mi[past]], P.k0)
        # --- dfn: tonight's defense against similar offenses and similar players of the family
        for f_ in np.unique(fam[mi]):
            fi = np.flatnonzero(fam[mi] == f_)
            A = e[mi[fi]].select([f"a_{s}" for s in AR.FEATS.get(f_, ())]).to_numpy().astype(float) if f_ in AR.FEATS else None
            wa = np.array([_fw(P, m, f"a_{s}") for s in AR.FEATS[f_]]) if A is not None else None
            if "dfn" in searches:
                for d_, idx in _groups(opp[mi[fi]]).items():
                    rows = mi[fi[idx]]
                    for a, r in enumerate(rows):
                        if not tset[r]:
                            continue
                        past = idx[gd[mi[fi[idx]]] < gd[r]]
                        if not len(past):
                            continue
                        # defense games ago: rank of distinct gamedays back
                        days = gd[mi[fi[past]]]
                        ud = np.unique(days)
                        ago = len(ud) - np.searchsorted(ud, days)
                        w = 0.5 ** (ago / P.hl_dfn) * _kernel(O[fi[past]], O[loc[r]], P.h_off, wo)
                        if A is not None:
                            w = w * _kernel(A[past], A[_pos(fi, loc[r])], P.h_arch, wa)
                        out["dfn_shift"][r], out["dfn_neff"][r], out["dfn_sw"][r] = _summ(w, rel[mi[fi[past]]], P.k0)
            if "int" in searches:
                PAIR = np.hstack([O[fi], D[fi]])
                wp = np.concatenate([wo, wd])
                gdf = gd[mi[fi]]
                for day in np.unique(gdf):
                    tr = np.flatnonzero((gdf == day) & tset[mi[fi]])
                    if not len(tr):
                        continue
                    past = np.flatnonzero((gdf < day) & (gdf >= day - P.int_window_days))
                    if not len(past):
                        continue
                    rec = 0.5 ** ((day - gdf[past]) / P.hl_int_days)
                    for t in tr:
                        w = rec * _kernel(PAIR[past], PAIR[t], P.h_pair, wp)
                        if A is not None:
                            w = w * _kernel(A[past], A[t], P.h_arch, wa)
                        r = mi[fi[t]]
                        out["int_shift"][r], out["int_neff"][r], out["int_sw"][r] = _summ(w, rel[mi[fi[past]]], P.k0)
    return e.select("rid", "season", "week", "game_id", "player_id", "team", "opponent", "family", "market", "expected", "actual", "rel", "relc").with_columns(
        [pl.Series(k, v) for k, v in out.items()])


def _groups(keys):
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    cuts = np.flatnonzero(k[1:] != k[:-1]) + 1
    return {keys[s[0]]: s for s in np.split(order, cuts)} if len(k) else {}


def _pos(fi, j):
    return int(np.searchsorted(fi, j))

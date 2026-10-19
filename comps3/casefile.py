"""The readable case file for one prop: every chain's comps, with who, when, how similar and how that game went.

  python -m comps3.casefile 2026_06_NYJ_NE "Braelon Allen" rush_yds [relc|rline]

Same rules as comps3.chains (cosine, MIN_SIM, TOP_K, MIN_COMPS, weights); this only lists the comps instead of averaging them.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

from comps2 import expectations as EX
from comps3 import chains as CH
from comps3 import profile as PR


def _names():
    import duckdb
    import config
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    n = con.execute("SELECT DISTINCT gsis_id, display_name FROM players WHERE gsis_id IS NOT NULL").pl()
    con.close()
    return dict(n.iter_rows())


def case(x: pl.DataFrame, rid: int, outcome="relc", overlaps=None) -> dict:
    overlaps = overlaps if overlaps is not None else pl.read_parquet(PR.OVERLAP_OUT)
    r = x.filter(pl.col("rid") == rid).row(0, named=True)
    m, f_ = r["market"], r["family"]
    sub = x.filter((pl.col("market") == m) & (pl.col("family") == f_) & (pl.col("gd") < r["gd"]) & (pl.col("gd") >= r["gd"] - CH.WINDOW_DAYS)
                   & pl.col(outcome).is_not_null())
    both = pl.concat([x.filter(pl.col("rid") == rid), sub])
    ospec, dspec, kind = CH.vec_spec(m, f_)
    VO, VD, VA = CH._vecs(both, ospec, "o"), CH._vecs(both, dspec, "d"), CH._arch(both, f_)
    allm = x.filter((pl.col("market") == m) & (pl.col("family") == f_))
    FL = CH.floors(CH._vecs(allm, ospec, "o"), CH._vecs(allm, dspec, "d"), CH._arch(allm, f_))
    k = "ov_run" if kind == "run" else "ov_pass"
    so, sd, sp = VO[1:] @ VO[0], VD[1:] @ VD[0], VA[1:] @ VA[0]
    rec = 0.5 ** ((r["gd"] - sub["gd"].to_numpy()) / CH.HL_DAYS)
    delta = sub[outcome].to_numpy().astype(float)
    qb_miss = 0.85 if CH.GROUP[m] == "rush" else 0.35
    ovd = overlaps.filter(pl.col("game_id") == r["game_id"])
    ov_def = dict(ovd.filter((pl.col("team") == r["opponent"]) & (pl.col("side") == "def")).select("prev_game_id", k).iter_rows())
    ov_off = dict(ovd.filter((pl.col("team") == r["team"]) & (pl.col("side") == "off")).select("prev_game_id", k).iter_rows())
    gids = sub["game_id"].to_list()
    out = {}
    masks = {
        "own": (sub["player_id"].to_numpy() == r["player_id"], sd,
                np.where(sub["qb_id"].to_numpy() == r["qb_id"], 1.0, qb_miss) * np.where(sub["team"].to_numpy() == r["team"], 1.0, 0.7)),
        "dfn": (sub["opponent"].to_numpy() == r["opponent"], np.sqrt(np.clip(so, 0, None) * np.clip(sp, 0, None)), np.array([ov_def.get(g, 0.0) for g in gids])),
        "team": (sub["team"].to_numpy() == r["team"], sd, np.array([ov_off.get(g, 0.0) for g in gids])),
        "int": (np.ones(sub.height, bool), np.cbrt(np.clip(so, 0, None) * np.clip(sd, 0, None) * np.clip(sp, 0, None)), np.ones(sub.height)),
    }
    for ch, (mask, sim, ctx) in masks.items():
        c = np.flatnonzero(mask)
        eff, n, idx = CH._pick(sim[c], sim[c] * rec[c] * ctx[c], delta[c], FL[ch])
        rows = sub[c[idx]].select("season", "week", "game_id", "player_id", "team", "opponent", outcome).with_columns(
            sim=pl.Series(np.round(sim[c[idx]], 3)), weight=pl.Series(np.round((sim * rec * ctx)[c[idx]], 3)))
        out[ch] = dict(effect=eff, n=n, comps=rows, floor=FL[ch])
    return dict(target=r, chains=out)


def show(game_id, player_name, market, outcome="relc"):
    x = CH.frame()
    names = _names()
    inv = {v: k for k, v in names.items()}
    pid = inv.get(player_name, player_name)
    t = x.filter((pl.col("game_id") == game_id) & (pl.col("player_id") == pid) & (pl.col("market") == market))
    if not t.height:
        raise SystemExit("no such row (the player needs a stored projection for that game)")
    cf = case(x, t["rid"][0], outcome)
    r = cf["target"]
    print(f"{player_name} {market} {game_id}: expected {r['expected']:.1f}, line {r['line']}, actual {r['actual']}")
    for ch, v in cf["chains"].items():
        eff = "no reliable comp" if np.isnan(v["effect"]) else f"{v['effect']:+.0%}"
        print(f"\n  {ch}: {eff} ({v['n']} comps above the bar {v['floor']:.2f})")
        for c in v["comps"].iter_rows(named=True):
            print(f"     {c['season']} wk{c['week']:>2} {names.get(c['player_id'], c['player_id']):<22} {c['team']} vs {c['opponent']}  sim {c['sim']:.2f}  "
                  f"weight {c['weight']:.2f}  went {c[outcome]:+.0%}")


if __name__ == "__main__":
    a = sys.argv[1:]
    show(a[0], a[1], a[2], a[3] if len(a) > 3 else "relc")

"""Mark 4 situation rules, played forward (2026-10-19): bet exactly where FanDuel has been wrong, no picker, no seeds.

Before each season S (2023-2026), every situation tag x market group x side (mark4.spot_score.tag_matrix: the blind-spot situations plus the book-lag
game numbers) is scored on lines from seasons BEFORE S only: recency-weighted return shrunk toward the group's base (K bets of weight), the number of
bets, and how many past seasons were up. A rule is kept when its shrunk return >= THETA, it has >= MIN_BETS past bets, and most past seasons were up.
In season S every line matching a kept rule is bet once, 1 unit, on the rule's side (a line matching rules on both sides takes the side whose best rule is
stronger). Nothing from S is seen before betting S.

  python -m mark4.rules            # writes records/lab/rules_forward.csv
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from mark4 import blindspots as BS
from mark4 import spot_score as SS

K = 200.0
MIN_BETS = 100
THETAS = (0.0, 0.02, 0.04, 0.06)
OUT = config.ROOT / "records" / "lab" / "rules_forward.csv"


def main():
    d = BS.frame()
    k = ["game_id", "player_id", "market"]
    ov = d.filter(pl.col("side") == "over").unique(k).sort(k)
    un = d.filter(pl.col("side") == "under").unique(k).select(*k, under_profit="profit", under_dec="dec")
    ov = ov.join(un, on=k, how="left")
    M, names = SS.tag_matrix(ov)
    season = ov["season"].to_numpy()
    t_row = season + (ov["week"].to_numpy() - 1) / 18.0
    grp = np.array([SS.GROUP_OF.get(m, "other") for m in ov["market"].to_list()])
    prof = {"over": ov["profit"].to_numpy().astype(float), "under": ov["under_profit"].to_numpy().astype(float)}
    rows, picked = [], []
    for S in (2023, 2024, 2025, 2026):
        tr, te = season < S, season == S
        w = 0.5 ** ((S - t_row) / 1.0)
        for th in THETAS:
            best = {side: np.full(len(season), -np.inf) for side in prof}
            for g in np.unique(grp):
                gtr, gte = tr & (grp == g), te & (grp == g)
                if not gte.any():
                    continue
                for side, pr in prof.items():
                    ok = gtr & ~np.isnan(pr)
                    base = (w[ok] * pr[ok]).sum() / w[ok].sum()
                    for j, nm in enumerate(names):
                        m = ok & M[:, j]
                        if m.sum() < MIN_BETS:
                            continue
                        shr = ((w[m] * pr[m]).sum() + K * base) / (w[m].sum() + K)
                        ups = [pr[m & (season == y)].mean() > 0 for y in np.unique(season[m]) if (m & (season == y)).sum() >= 10]
                        if shr >= th and sum(ups) * 2 > len(ups):
                            hit = gte & M[:, j]
                            best[side][hit] = np.maximum(best[side][hit], shr)
                            if th == THETAS[1]:
                                picked.append(dict(season=S, group=g, situation=nm, side=side, past_shrunk=float(shr), past_bets=int(m.sum())))
            bu, bo = best["under"], best["over"]
            bet_u = te & np.isfinite(bu) & (bu >= bo)
            bet_o = te & np.isfinite(bo) & (bo > bu)
            pu, po = prof["under"][bet_u], prof["over"][bet_o]
            pu, po = pu[~np.isnan(pu)], po[~np.isnan(po)]
            n = len(pu) + len(po)
            rows.append(dict(season=S, theta=th, bets=n, units=float(pu.sum() + po.sum()), ret=float((pu.sum() + po.sum()) / n) if n else None,
                             unders=len(pu), overs=len(po)))
    t = pl.DataFrame(rows)
    t.write_csv(OUT)
    pl.DataFrame(picked).write_csv(OUT.with_name("rules_picked.csv"))
    rw = lambda s: 0.5 ** (2026 - s)
    for th in THETAS:
        x = t.filter(pl.col("theta") == th)
        st = sum(rw(r["season"]) * r["bets"] for r in x.iter_rows(named=True))
        un_ = sum(rw(r["season"]) * r["units"] for r in x.iter_rows(named=True))
        print(f"bar {th:.0%}: recency-weighted return {un_ / st if st else float('nan'):+.1%} | " + " | ".join(
            f"{r['season']}: {r['ret']:+.1%} on {r['bets']}" if r["bets"] else f"{r['season']}: no bets" for r in x.iter_rows(named=True)))


if __name__ == "__main__":
    main()

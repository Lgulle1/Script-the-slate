"""The betting-strategy tournament for Mark 4 (2026-10-20): every format, scored per season, played forward, for each Mark separately.

Probabilities come from the forward, no-randomness predictions of each version (records/lab/forward_p_*_0.npy): each season predicted only from lines
graded before its 6-week block. Versions: Mark 3 (frozen inputs), Mark 4h (comps vs the line + gentle recency, leak-free inputs), and a blend (the
average of the two probabilities).

PARLAY formats (one leg per game always; a normal FanDuel parlay cannot hold two legs of one game):
  legs 2 / 3 / 4 | parlays a week 1 / 2 / 4 / 8 (the first 4 take the 30% profit boost, one per boost window; "no boost" scores them all plain)
  leg pool: the top 6 / 8 / 12 / 16 legs of the week | pairing: rank (best with next best), snake (best with worst), spread (columns)
  markets in one parlay: any / all different | sides: unders only / either side (over or under, whichever has the bigger edge) | edge bar 2 / 4 / 6%
  markets: the 4 core (rec, rec yds, rush yds, pass yds) / all 7
STRAIGHT formats: top 3 / 6 / 10 / 20 a week | edge 0 / 2 / 4 / 6 / 8% | unders / overs / either | 4 / 7 markets

Each format: units, units bet and bets per season. Then, honestly, NESTED: before each season the format with the best shrunk recency-weighted return
over the earlier seasons is chosen and bet that season (shrunk toward 0 with SHRINK units of weight, so a format needs real volume to win).

  python -m mark4.tournament "Mark 4h clean"      # one version per run (they run in parallel); writes records/lab/tournament_<version>.parquet
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import polars as pl

from mark4 import forward as F

BOOST, N_BOOSTS = F.ST.BOOST, 4
SHRINK = 20.0


def load_p(version):
    if version == "Blend":
        return (np.load(F.OUT / "forward_p_Mark_3_det_0.npy") + np.load(F.OUT / "forward_p_Mark_4h_clean_0.npy")) / 2
    return np.load(F.OUT / f"forward_p_{version.replace(' ', '_')}_0.npy")


def week_pools(d, A, p, sides, markets, edge):
    """{(season, week): rows sorted by chance}, rows as tuples (chance, edge, dec, won, game_id, player_id, market)."""
    eo, eu = p - A.be_o, (1 - p) - A.be_u
    if sides == "under":
        over = np.zeros(len(p), bool)
    elif sides == "over":
        over = np.ones(len(p), bool)
    else:
        over = eo > eu
    ch = np.where(over, p, 1 - p)
    ed = np.where(over, eo, eu)
    dec = np.where(over, A.dec_o, A.dec_u)
    won = np.where(over, A.over_won, 1 - A.over_won).astype(float)
    mk = d["market"].to_numpy()
    ok = ~np.isnan(ch) & ~np.isnan(won) & (d["history"].to_numpy() >= F.CF.MIN_HISTORY) & (ed >= edge)
    if markets == "four":
        ok &= np.isin(mk, list(F.IB.FOUR))
    season, week, gid, pid = d["season"].to_numpy(), d["week"].to_numpy(), d["game_id"].to_numpy(), d["player_id"].to_numpy()
    W = {}
    for i in np.flatnonzero(ok):
        W.setdefault((int(season[i]), int(week[i])), []).append((ch[i], ed[i], dec[i], won[i], gid[i], pid[i], mk[i]))
    for k in W:
        W[k].sort(key=lambda r: -r[0])
    return W


def pool(rows, m):
    """Top m legs, one per player (a player can carry one leg a week across parlays)."""
    out, seen = [], set()
    for r in rows:
        if r[5] in seen:
            continue
        seen.add(r[5])
        out.append(r)
        if len(out) == m:
            break
    return out


def build(legs, k, c, pairing, mix):
    """Up to c parlays of k legs from `legs` (sorted), never two legs of one game; mix 'diff' = no two legs of one market."""
    used = set()
    out = []
    order = list(range(len(legs)))
    for j in range(c):
        free = [i for i in order if i not in used]
        if not free:
            break
        first = free[0]
        grp = [first]
        games, mkts = {legs[first][4]}, {legs[first][6]}
        cand = free[1:]
        if pairing == "snake":
            cand = cand[::-1]
        elif pairing == "spread":
            step = max(1, len(free) // k)
            cand = cand[step - 1:] + cand[:step - 1]
        for i in cand:
            if legs[i][4] in games or (mix == "diff" and legs[i][6] in mkts):
                continue
            grp.append(i)
            games.add(legs[i][4]); mkts.add(legs[i][6])
            if len(grp) == k:
                break
        if len(grp) < k:
            break
        used |= set(grp)
        out.append([legs[i] for i in grp])
    return out


def score_parlays(W, k, c, m, pairing, mix, boost):
    per = {}
    for (s, w), rows in W.items():
        pars = build(pool(rows, m), k, c, pairing, mix)
        if not pars:
            continue
        u = 0.0
        for n, par in enumerate(pars):
            win = all(r[3] == 1 for r in par)
            gain = float(np.prod([r[2] for r in par])) - 1
            u += gain * ((1 + BOOST) if (boost and n < N_BOOSTS) else 1.0) if win else -1.0
        v = per.setdefault(s, [0.0, 0.0, 0])
        v[0] += u; v[1] += len(pars); v[2] += 1
    return per


def score_straights(W, n):
    per = {}
    for (s, w), rows in W.items():
        picks = pool(rows, n)
        u = sum((r[2] - 1) if r[3] == 1 else -1.0 for r in picks)
        v = per.setdefault(s, [0.0, 0.0, 0])
        v[0] += u; v[1] += len(picks); v[2] += 1
    return per


def run(version):
    t0 = time.time()
    d, A, *_ = F.frame()
    p = load_p(version)
    rows = []
    pools = {}
    for sides, markets, edge in itertools.product(("under", "over", "both"), ("four", "all"), (0.0, 0.02, 0.04, 0.06, 0.08)):
        pools[(sides, markets, edge)] = week_pools(d, A, p, sides, markets, edge)
    print(f"{version}: pools built ({time.time() - t0:.0f}s)", flush=True)
    for (sides, markets, edge), W in pools.items():
        for n in (3, 6, 10, 20):
            for s, v in score_straights(W, n).items():
                rows.append(dict(kind="straights", fmt=f"top {n} straights | edge {edge:.0%} | {sides} | {markets} markets", season=s, units=v[0], staked=v[1], weeks=v[2]))
    print(f"{version}: straights done ({time.time() - t0:.0f}s)", flush=True)
    for sides, markets, edge in itertools.product(("under", "both"), ("four", "all"), (0.02, 0.04, 0.06)):
        W = pools[(sides, markets, edge)]
        for k, c, m, pairing, mix, boost in itertools.product((2, 3, 4), (1, 2, 4, 8), (6, 8, 12, 16), ("rank", "snake", "spread"), ("any", "diff"), (True, False)):
            if k * c > m:
                continue
            name = f"{c} x {k}-leg parlays{'' if boost else ' (no boost)'} | pool {m} | {pairing} | markets {mix} | edge {edge:.0%} | {sides} | {markets} markets"
            for s, v in score_parlays(W, k, c, m, pairing, mix, boost).items():
                rows.append(dict(kind="parlays" if boost else "parlays, no boost", fmt=name, season=s, units=v[0], staked=v[1], weeks=v[2]))
    t = pl.DataFrame(rows).with_columns(version=pl.lit(version))
    t.write_parquet(F.OUT / f"tournament_{version.replace(' ', '_')}.parquet")
    print(f"{version}: {t['fmt'].n_unique()} formats scored ({time.time() - t0:.0f}s)", flush=True)
    return t


_COMBOS = {}


def _combos(m, k):
    if (m, k) not in _COMBOS:
        _COMBOS[(m, k)] = np.array(list(itertools.combinations(range(m), k)), dtype=np.int32).reshape(-1, k)
    return _COMBOS[(m, k)]


def combo_ev(W, k, m, mix="any"):
    """ALL-COMBO scoring (2026-10-20): per week, the average result of every k-leg parlay that can be built from the top m legs (different games,
    and different markets if mix == 'diff'). That is the expected return of a parlay drawn from that pool, measured on what actually happened, with
    far less luck than the one or two parlays a rule happens to build. Per season: {season: [units plain, units boosted, weeks, combos per week]},
    one unit per week (the week's average parlay)."""
    per = {}
    for (s, w), rows in W.items():
        legs = pool(rows, m)
        if len(legs) < k:
            continue
        C = _combos(len(legs), k)
        dec = np.array([r[2] for r in legs]); won = np.array([r[3] for r in legs]); gm = np.array([r[4] for r in legs]); mk = np.array([r[6] for r in legs])
        ok = np.ones(len(C), bool)
        for a, b in itertools.combinations(range(k), 2):
            ok &= gm[C[:, a]] != gm[C[:, b]]
            if mix == "diff":
                ok &= mk[C[:, a]] != mk[C[:, b]]
        C = C[ok]
        if not len(C):
            continue
        hit = won[C].prod(1) == 1
        gain = dec[C].prod(1) - 1
        plain = np.where(hit, gain, -1.0).mean()
        boosted = np.where(hit, gain * (1 + BOOST), -1.0).mean()
        v = per.setdefault(s, [0.0, 0.0, 0, 0])
        v[0] += plain; v[1] += boosted; v[2] += 1; v[3] += len(C)
    return per


def run_combos(version):
    """The all-combo tournament: pool (sides x markets x edge x depth) x legs x market mix, plus straights per leg for comparison."""
    t0 = time.time()
    d, A, *_ = F.frame()
    p = load_p(version)
    rows = []
    for sides, markets, edge in itertools.product(("under", "over", "both"), ("four", "all"), (0.0, 0.02, 0.04, 0.06)):
        W = week_pools(d, A, p, sides, markets, edge)
        for m in (4, 6, 8, 12, 16):
            for s, v in score_straights(W, m).items():
                rows.append(dict(sides=sides, markets=markets, edge=edge, depth=m, legs=1, mix="any", season=s, plain=v[0], boosted=v[0], weeks=v[2], staked=v[1]))
            for k, mix in itertools.product((2, 3, 4, 5), ("any", "diff")):
                if k > m:
                    continue
                for s, v in combo_ev(W, k, m, mix).items():
                    rows.append(dict(sides=sides, markets=markets, edge=edge, depth=m, legs=k, mix=mix, season=s, plain=v[0], boosted=v[1], weeks=v[2], staked=float(v[2])))
        print(f"{version}: {sides} {markets} {edge} ({time.time() - t0:.0f}s)", flush=True)
    t = pl.DataFrame(rows).with_columns(version=pl.lit(version))
    t.write_parquet(F.OUT / f"combos_{version.replace(' ', '_')}.parquet")
    return t


def nested(t: pl.DataFrame, kind: str, seasons=(2023, 2024, 2025, 2026)):
    """Before each season: the format with the best shrunk recency-weighted return over earlier seasons; its result that season."""
    x = t.filter(pl.col("kind") == kind)
    out = []
    for S in seasons:
        past = x.filter(pl.col("season") < S).with_columns(w=0.5 ** (S - pl.col("season")))
        sc = past.group_by("fmt").agg(u=(pl.col("w") * pl.col("units")).sum(), st=(pl.col("w") * pl.col("staked")).sum(), n=pl.col("staked").sum())
        sc = sc.filter(pl.col("n") >= 30).with_columns(score=pl.col("u") / (pl.col("st") + SHRINK)).sort("score", descending=True)
        if not sc.height:
            continue
        best = sc["fmt"][0]
        cur = x.filter((pl.col("season") == S) & (pl.col("fmt") == best))
        u, st = (float(cur["units"][0]), float(cur["staked"][0])) if cur.height else (0.0, 0.0)
        out.append(dict(season=S, fmt=best, units=u, staked=st, ret=u / st if st else None, past_score=float(sc["score"][0])))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "combos":
        for v in sys.argv[2:]:
            run_combos(v)
    else:
        for v in sys.argv[1:]:
            run(v)

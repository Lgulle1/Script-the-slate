"""Tournament 3 (yours, 2026-10-15): does "run the tournament on every past season and bet the winner" actually make money on a season it has not seen?

For each season S of 2021-2026 (the held-out season):
  1. the picker is retrained without S: every other season T is predicted by a picker trained on the remaining seasons (not S, not T), and S itself by a picker
     trained on all five other seasons;
  2. every candidate setup is scored on the five other seasons and the winner is chosen (two ways: best total return per unit, and best worst season);
  3. that winner is bet on S, untouched.
The six held-out results are what the method would really have earned. Fold "ALL" tunes on all six seasons: that is the setup we would bet going forward, and
its return there is in-hindsight, not a forecast.

Knobs (everything tournament 2 had, plus which markets): picker (5 versions) x side (under / over / both) x edge cut-off (0-8 points) x history rule (none / 3+)
x ranking (picker chance / edge) x markets (all four, each one alone, receiving, all but pass yards) x how many (singles 1-10, parlays from the top 2-8)
x per-game limit (2 / 3 / none) x per-player limit (1 / none; parlays always 1). Singles and parlays are chosen separately, then the best mix
(singles only / parlays only / both) by return per unit. Stakes 1 unit; parlays are 2 legs with the 30% boost. FanDuel's four markets only.

  python scripts/tournament3.py      # writes records/tuning/tournament3_*.json / .parquet
"""
import importlib.util
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T2 = _load("tournament2")
TN, CF, ST = T2.TN, T2.CF, T2.ST
OUTD = config.ROOT / "records" / "tuning"
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
FOUR = ["rec", "rec_yds", "rush_yds", "pass_yds"]
MARKETS = {"all four": FOUR, "receptions": ["rec"], "receiving yards": ["rec_yds"], "rush yards": ["rush_yds"], "pass yards": ["pass_yds"],
           "receiving (rec + rec yds)": ["rec", "rec_yds"], "all but pass yards": ["rec", "rec_yds", "rush_yds"]}
SIDES, THETAS, HISTS, RANKS = ("under", "over", "both"), (0.0, 0.02, 0.04, 0.06, 0.08), (0, 3), ("chance", "edge")
SINGLE_GRID = list(itertools.product((1, 2, 3, 4, 5, 6, 8, 10), (2, 3, None), (1, None)))
PARLAY_GRID = list(itertools.product((2, 4, 6, 8), (2, 3, None)))
MIN_BETS = dict(singles=100, parlays=50)     # over the tuning seasons
MIN_SEASON = dict(singles=10, parlays=5)     # per tuning season, for the "best worst season" rule
LOCKED = dict(side="under", edge_min=0.04, history_min=3, rank_by="chance", markets="all four")
G = {}                                       # shared with workers (fork)


def fold_predictions(d, A, makers, layered):
    """{S: p_over} - rows of S predicted without S; every other season T predicted without S and T. Key "ALL": every season predicted without itself."""
    X = TN.feats(d, layered=layered)
    season = d["season"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    fit = lambda train, te: np.mean([m().fit(X[np.isin(season, train) & graded], A.over_won[np.isin(season, train) & graded]).predict_proba(X[te])[:, 1]
                                     for m in makers], axis=0)
    out = {"ALL": np.full(d.height, np.nan)}
    for S in SEASONS:
        out["ALL"][season == S] = fit([t for t in SEASONS if t != S], season == S)
    for S in SEASONS:
        p = out["ALL"].copy()
        for T in SEASONS:
            if T != S:
                p[season == T] = fit([u for u in SEASONS if u not in (S, T)], season == T)
        out[S] = p
    return out


def week_lists(p_over):
    """{(side, theta, hist, rank, markets): [week rows sorted best first]}; a row is (game, player, dec, won, season)."""
    A, base = G["A"], G["base"]
    eo, eu = p_over - A.be_o, (1 - p_over) - A.be_u
    out = {}
    for side in SIDES:
        if side == "under":
            ch, ed, dec, won = 1 - p_over, eu, A.dec_u, 1 - A.over_won
        elif side == "over":
            ch, ed, dec, won = p_over, eo, A.dec_o, A.over_won
        else:
            o = eo >= eu
            ch, ed, dec, won = np.where(o, p_over, 1 - p_over), np.maximum(eo, eu), np.where(o, A.dec_o, A.dec_u), np.where(o, A.over_won, 1 - A.over_won)
        ok = base["ok"] & ~np.isnan(won.astype(float)) & ~np.isnan(ch)
        rows = list(zip(base["game"].tolist(), base["player"].tolist(), dec.tolist(), won.tolist(), base["season"].tolist()))
        for theta, hist, rank, (mname, mk) in itertools.product(THETAS, HISTS, RANKS, MARKETS.items()):
            m = ok & (ed >= theta) & (base["history"] >= hist) & np.isin(base["market"], mk)
            key = ch if rank == "chance" else ed
            W = []
            for idx in base["weeks"]:
                i = idx[m[idx]]
                i = i[np.argsort(-key[i], kind="stable")]
                W.append([rows[j] for j in i.tolist()])
            out[(side, theta, hist, rank, mname)] = W
    return out


def top(rows, n, gcap, pcap):
    out, g, p = [], {}, {}
    for r in rows:
        if pcap and p.get(r[1], 0) >= pcap:
            continue
        if gcap and g.get(r[0], 0) >= gcap:
            continue
        out.append(r)
        g[r[0]] = g.get(r[0], 0) + 1
        p[r[1]] = p.get(r[1], 0) + 1
        if len(out) == n:
            break
    return out


def parlays(legs):
    used, res = set(), []
    for i, a in enumerate(legs):
        if i in used:
            continue
        for j in range(i + 1, len(legs)):
            if j not in used and legs[j][0] != a[0]:
                used |= {i, j}
                b = legs[j]
                res.append((a[4], (a[2] * b[2] - 1) * (1 + ST.BOOST) if (a[3] == 1 and b[3] == 1) else -1.0))
                break
    return res


def keys():
    """Candidate keys in a fixed order, singles then parlays."""
    ks = []
    for f in itertools.product(SIDES, THETAS, HISTS, RANKS, MARKETS):
        ks += [("singles",) + f + g for g in SINGLE_GRID]
        ks += [("parlays",) + f + (n, gc, 1) for n, gc in PARLAY_GRID]
    return ks


def run_job(job):
    """One (picker, fold): per candidate, units and bets in every season -> array (n_candidates, n_seasons, 2)."""
    pname, fold = job
    t0 = time.time()
    L = week_lists(G["preds"][pname][fold])
    si = {s: k for k, s in enumerate(SEASONS)}
    res = []
    for f in itertools.product(SIDES, THETAS, HISTS, RANKS, MARKETS):
        W = L[f]
        for n, gc, pc in SINGLE_GRID:
            a = np.zeros((len(SEASONS), 2))
            for rows in W:
                for r in top(rows, n, gc, pc):
                    a[si[r[4]], 0] += (r[2] - 1) if r[3] == 1 else -1.0
                    a[si[r[4]], 1] += 1
            res.append(a)
        for n, gc in PARLAY_GRID:
            a = np.zeros((len(SEASONS), 2))
            for rows in W:
                for s, u in parlays(top(rows, n, gc, 1)):
                    a[si[s], 0] += u
                    a[si[s], 1] += 1
            res.append(a)
    print(f"  {pname} | fold {fold}: {time.time() - t0:.0f}s", flush=True)
    return pname, fold, np.stack(res)


def describe(pname, k):
    kind, side, theta, hist, rank, mk, n, gc, pc = k
    return dict(kind=kind, picker=pname, side=side, edge_min=theta, history_min=hist, rank_by=rank, markets=mk, top_n=n,
                per_game=str(gc or "none"), per_player=str(pc or "none"))


def choose(R, K, tune_cols, rule):
    """Index of the winning singles and parlay candidate (or None) and the chosen mix, using only the tuning seasons."""
    u, b = R[:, tune_cols, 0], R[:, tune_cols, 1]
    tu, tb = u.sum(1), b.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        roi = tu / tb
        worst = np.where(b >= 0, u / np.where(b > 0, b, np.nan), np.nan)
    best = {}
    for kind in ("singles", "parlays"):
        isk = np.array([k[0][0] == kind for k in K])
        if rule == "total":
            score = np.where(isk & (tb >= MIN_BETS[kind]), roi, -np.inf)
        else:
            okseason = (b >= MIN_SEASON[kind]).all(1)
            score = np.where(isk & okseason & (tb >= MIN_BETS[kind]), np.nanmin(worst, axis=1), -np.inf)
            score = np.where(np.isnan(score), -np.inf, score)
        best[kind] = int(np.argmax(score)) if np.isfinite(score.max()) else None
    mixes = {"singles only": [best["singles"]], "parlays only": [best["parlays"]], "both": [best["singles"], best["parlays"]]}
    mixes = {m: ix for m, ix in mixes.items() if None not in ix}

    def mscore(ix):
        if rule == "total":
            return sum(tu[i] for i in ix) / sum(tb[i] for i in ix)
        su, sb = sum(u[i] for i in ix), sum(b[i] for i in ix)
        return np.nanmin(su / np.where(sb > 0, sb, np.nan))
    mix = max(mixes, key=lambda m: mscore(mixes[m]))
    return best, mix, mixes[mix]


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    sw = d.select("season", "week").to_numpy()
    weeks = [np.flatnonzero((sw[:, 0] == s) & (sw[:, 1] == w)) for s, w in sorted({tuple(x) for x in sw.tolist()})]
    G["A"] = A
    G["base"] = dict(ok=d["market"].is_in(FOUR).to_numpy() & d["season"].is_in(list(SEASONS)).to_numpy(), history=d["history"].to_numpy(),
                     market=d["market"].to_numpy(), game=d["game_id"].to_numpy(), player=d["player_id"].to_numpy(), season=d["season"].to_numpy(), weeks=weeks)
    G["preds"] = {}
    for pname, (makers, layered) in T2.pickers().items():
        G["preds"][pname] = fold_predictions(d, A, makers, layered)
        print(f"{pname}: fold predictions done ({time.time() - t0:.0f}s)", flush=True)
    folds = list(SEASONS) + ["ALL"]
    jobs = [(p, f) for p in G["preds"] for f in folds]
    with Pool(4) as pool:
        done = pool.map(run_job, jobs, chunksize=1)
    ks = keys()
    K = [(k, pn) for pn in G["preds"] for k in ks]                     # (key, picker) per candidate row
    R = {f: np.concatenate([r for pn in G["preds"] for (p, ff, r) in done if p == pn and ff == f]) for f in folds}
    print(f"candidates per fold: {len(K)} ({time.time() - t0:.0f}s)", flush=True)

    out = dict(candidates=len(K), singles_candidates=sum(k[0][0] == "singles" for k in K), parlay_candidates=sum(k[0][0] == "parlays" for k in K),
               seasons=list(SEASONS), held_out={}, all_seasons={}, rank_vs_heldout={}, locked={})
    lock_ix = {}
    for kind, n, gc, pc in (("singles", 2, 3, None), ("parlays", 2, 3, 1), ("singles", 8, 3, None), ("parlays", 8, 3, 1)):
        k = (kind, "under", 0.04, 3, "chance", "all four", n, gc, pc)
        lock_ix[(kind, n)] = K.index((k, "P1 six averaged + layers (locked)"))

    def pack(R_, ix, cols):
        return dict(units=float(R_[ix][cols, 0].sum()), bets=int(R_[ix][cols, 1].sum()))

    for rule in ("total", "worst season"):
        rows = []
        for f in SEASONS:
            h = SEASONS.index(f)
            tune = [i for i in range(len(SEASONS)) if i != h]
            best, mix, ix = choose(R[f], K, tune, rule)
            row = dict(season=f, mix=mix, setups=[describe(K[i][1], K[i][0]) for i in ix],
                       tune=[pack(R[f], i, tune) for i in ix], held_out=[pack(R[f], i, [h]) for i in ix])
            row["held_out_units"] = sum(x["units"] for x in row["held_out"])
            row["held_out_bets"] = sum(x["bets"] for x in row["held_out"])
            row["tune_units"] = sum(x["units"] for x in row["tune"])
            row["tune_bets"] = sum(x["bets"] for x in row["tune"])
            rows.append(row)
        out["held_out"][rule] = rows
        best, mix, ix = choose(R["ALL"], K, list(range(len(SEASONS))), rule)
        out["all_seasons"][rule] = dict(mix=mix, setups=[describe(K[i][1], K[i][0]) for i in ix],
                                        per_season=[{str(s): pack(R["ALL"], i, [j]) for j, s in enumerate(SEASONS)} for i in ix])
    # locked rules, judged on each held-out season (picker trained without that season)
    for (kind, n), i in lock_ix.items():
        out["locked"][f"{kind} top {n}"] = {str(s): pack(R[s], i, [j]) for j, s in enumerate(SEASONS)}
    # does a high tuning rank predict the held-out season?
    for kind in ("singles", "parlays"):
        isk = np.array([k[0][0] == kind for k in K])
        bands = {}
        for f in SEASONS:
            h = SEASONS.index(f)
            tune = [i for i in range(len(SEASONS)) if i != h]
            tu, tb = R[f][:, tune, 0].sum(1), R[f][:, tune, 1].sum(1)
            hu, hb = R[f][:, h, 0], R[f][:, h, 1]
            ok = isk & (tb >= MIN_BETS[kind]) & (hb > 0)
            order = np.argsort(-(tu[ok] / tb[ok]))
            hu_, hb_ = hu[ok][order], hb[ok][order]
            N = len(order)
            for nm, lo, hi in (("top 0.1%", 0, .001), ("top 1%", 0, .01), ("top 10%", 0, .1), ("middle 10%", .45, .55), ("bottom 10%", .9, 1.0), ("all", 0, 1.0)):
                a, b = int(lo * N), max(int(hi * N), int(lo * N) + 1)
                bands.setdefault(nm, {})[str(f)] = float(hu_[a:b].sum() / hb_[a:b].sum())
        out["rank_vs_heldout"][kind] = bands
    (OUTD / "tournament3_summary.json").write_text(json.dumps(out, indent=1, default=str))
    # every candidate, every fold: per-season units and bets
    rec = []
    for f in folds:
        for i, (k, pn) in enumerate(K):
            r = describe(pn, k)
            r["fold"] = str(f)
            for j, s in enumerate(SEASONS):
                r[f"u{s}"], r[f"b{s}"] = float(R[f][i, j, 0]), int(R[f][i, j, 1])
            rec.append(r)
    pl.DataFrame(rec).write_parquet(OUTD / "tournament3_all.parquet")
    for rule, rows in out["held_out"].items():
        U, B = sum(r["held_out_units"] for r in rows), sum(r["held_out_bets"] for r in rows)
        print(f"\nrule '{rule}': held-out total {U:+.1f}u on {B} bets ({U / B:+.1%})")
        for r in rows:
            print(f"  {r['season']}: {r['mix']:<13} tuned {r['tune_units'] / r['tune_bets']:+.1%} -> held out {r['held_out_units']:+.1f}u on {r['held_out_bets']} "
                  f"({r['held_out_units'] / max(r['held_out_bets'], 1):+.1%})  {[(x['kind'], x['picker'][:2], x['side'], x['markets'], x['edge_min'], x['top_n'], x['per_game']) for x in r['setups']]}")
    print(json.dumps(out["rank_vs_heldout"], indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

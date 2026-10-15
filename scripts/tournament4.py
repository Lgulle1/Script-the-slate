"""Tournament 4 (yours, 2026-10-15): retrain the projection model itself, then rerun everything on top of it.

For every projection-model variant (scripts/proj_variants.py: model settings, training window, recency, feature groups on/off), walk-forward over 2020-2024:
  1. the line study is rebuilt on that variant's projections (chances for every graded FanDuel line, 2021-2026; 2025-2026 projections are the stored
     full-stack ones, so a variant only changes their neighbour pools),
  2. the picker (locked six-model average, and the single model) is retrained per held-out season exactly as in tournament 3,
  3. every betting rule of tournament 3 (126,000 per variant counting both pickers... see tournament3.py for the knobs) is scored per season.
Judged on 2021-2024, the seasons the variant actually changes:
  - accuracy of each variant's projections (average miss vs the base model, same rows),
  - the same fixed rules on every variant (locked parlays, locked singles, receiving-yards unders) on held-out seasons,
  - the fully honest test: for each held-out season 2021-2024, choose the best (variant x betting rule) on the other three seasons and bet it on the held-out one.

  python scripts/tournament4.py            # needs data/processed/proj_variants/*_wf.parquet; writes records/tuning/tournament4_summary.json
"""
import importlib.util
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
    sys.modules[name] = m            # worker processes find run_job by module name
    spec.loader.exec_module(m)
    return m


T3 = _load("tournament3")
PV = _load("proj_variants")
T2, TN = T3.T2, T3.TN
LS, TS = TN.LS, TN.TM.T
VD = PV.OUTD
OUTD = config.ROOT / "records" / "tuning"
JUDGE = (2021, 2022, 2023, 2024)
PICKERS = ("P1 six averaged + layers (locked)", "P3 one model + layers (tournament 1 winner)")


def build_sel(name):
    """Line study on one variant's projections -> proj_variants/<name>_sel.parquet."""
    out = VD / f"{name}_sel.parquet"
    if out.exists():
        return name
    LS.BASE_2024, LS.OUT = VD / f"{name}_wf.parquet", out
    t0 = time.time()
    LS.main_build()
    print(f"  line study {name}: {time.time() - t0:.0f}s", flush=True)
    return name


def accuracy(name, base):
    """Average miss of the variant's projections vs base on the same player rows, by market (2021-2024)."""
    v = pl.read_parquet(VD / f"{name}_wf.parquet").filter((pl.col("kind") == "player") & pl.col("season").is_in(JUDGE))
    j = base.join(v.select("market", "season", "week", "entity", pv="prediction"), on=["market", "season", "week", "entity"], how="inner").drop_nulls(["pb", "pv", "actual"])
    return {m: dict(n=g.height, base=float((g["pb"] - g["actual"]).abs().mean()), variant=float((g["pv"] - g["actual"]).abs().mean()))
            for (m,), g in j.group_by("market") if m in T3.FOUR}


def run_variant(name):
    TS.SEL = VD / f"{name}_sel.parquet"
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    sw = d.select("season", "week").to_numpy()
    T3.G["A"] = A
    T3.G["base"] = dict(ok=d["market"].is_in(T3.FOUR).to_numpy() & d["season"].is_in(list(T3.SEASONS)).to_numpy(), history=d["history"].to_numpy(),
                        market=d["market"].to_numpy(), game=d["game_id"].to_numpy(), player=d["player_id"].to_numpy(), season=d["season"].to_numpy(),
                        weeks=[np.flatnonzero((sw[:, 0] == s) & (sw[:, 1] == w)) for s, w in sorted({tuple(x) for x in sw.tolist()})])
    pk = T2.pickers()
    T3.G["preds"] = {p: T3.fold_predictions(d, A, *pk[p]) for p in PICKERS}
    jobs = [(p, f) for p in PICKERS for f in JUDGE]
    with Pool(4) as pool:
        done = pool.map(T3.run_job, jobs, chunksize=1)
    return {f: np.concatenate([r for p in PICKERS for (pp, ff, r) in done if pp == p and ff == f]) for f in JUDGE}


def main():
    t0 = time.time()
    names = [n for n in PV.VARIANTS if (VD / f"{n}_wf.parquet").exists()]
    print("variants:", names, flush=True)
    with Pool(4, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(build_sel, names))
    base = pl.read_parquet(VD / "base_wf.parquet").filter((pl.col("kind") == "player") & pl.col("season").is_in(JUDGE)).select("market", "season", "week", "entity", "actual", pb="prediction")
    ks = T3.keys()
    K = [(k, p) for p in PICKERS for k in ks]
    si = {s: i for i, s in enumerate(T3.SEASONS)}
    J = [si[s] for s in JUDGE]
    out = dict(variants={}, seasons=list(JUDGE), candidates_per_variant=len(K))
    R = {}
    fixed = {"locked parlays, top 8 (4 a week)": ("parlays", "under", 0.04, 3, "chance", "all four", 8, 3, 1),
             "locked parlays, top 2 (1 a week)": ("parlays", "under", 0.04, 3, "chance", "all four", 2, 3, 1),
             "locked singles, top 2": ("singles", "under", 0.04, 3, "chance", "all four", 2, 3, None),
             "locked singles, top 8": ("singles", "under", 0.04, 3, "chance", "all four", 8, 3, None),
             "unders all four, 4+ edge, up to 10": ("singles", "under", 0.04, 3, "chance", "all four", 10, None, None),
             "receiving yards unders, 4+ edge, up to 10": ("singles", "under", 0.04, 3, "chance", "receiving yards", 10, None, None),
             "receiving yards unders, any edge, up to 10": ("singles", "under", 0.0, 3, "chance", "receiving yards", 10, None, None)}
    fix_ix = {nm: K.index((k, PICKERS[0])) for nm, k in fixed.items()}
    for n in names:
        tv = time.time()
        R[n] = run_variant(n)
        acc = accuracy(n, base)
        held = {nm: {str(s): dict(units=float(R[n][s][i, si[s], 0]), bets=int(R[n][s][i, si[s], 1])) for s in JUDGE} for nm, i in fix_ix.items()}
        out["variants"][n] = dict(accuracy=acc, fixed_rules=held)
        line = " | ".join(f"{nm.split(',')[0][:22]} {sum(x['units'] for x in h.values()) / max(sum(x['bets'] for x in h.values()), 1):+.1%}" for nm, h in held.items())
        print(f"{n}: {time.time() - tv:.0f}s  miss vs base " + " ".join(f"{m} {a['variant'] - a['base']:+.3f}" for m, a in sorted(acc.items())) + f"\n   {line}", flush=True)
    # fully honest: best (variant x rule) on the other three seasons, bet on the held-out one
    res = {}
    for rule in ("total", "worst season"):
        rows = []
        for s in JUDGE:
            tune = [si[t] for t in JUDGE if t != s]
            best = None
            for n in names:
                b, mix, ix = T3.choose(R[n][s], K, tune, rule)
                tu = sum(R[n][s][i, tune, 0].sum() for i in ix) / sum(R[n][s][i, tune, 1].sum() for i in ix)
                if best is None or tu > best[0]:
                    best = (tu, n, mix, ix)
            tu, n, mix, ix = best
            hu, hb = sum(R[n][s][i, si[s], 0] for i in ix), sum(R[n][s][i, si[s], 1] for i in ix)
            rows.append(dict(season=s, variant=n, mix=mix, setups=[T3.describe(K[i][1], K[i][0]) for i in ix], tune_roi=float(tu), held_units=float(hu), held_bets=int(hb)))
        res[rule] = rows
        U, B = sum(r["held_units"] for r in rows), sum(r["held_bets"] for r in rows)
        print(f"\nhonest variant x rule tournament, rule '{rule}': held-out {U:+.1f}u on {B} ({U / max(B, 1):+.1%})")
        for r in rows:
            x = r["setups"]
            print(f"  {r['season']}: {r['variant']:<15} {r['mix']:<13} tuned {r['tune_roi']:+.1%} -> {r['held_units']:+.1f}u on {r['held_bets']} "
                  f"({r['held_units'] / max(r['held_bets'], 1):+.1%})  {[(z['kind'], z['side'], z['markets'], z['edge_min'], z['top_n']) for z in x]}")
    out["honest"] = res
    # per variant: share of straight-bet setups positive on held-out seasons (pooled 2021-2024, 150+ bets), and median
    for n in names:
        isk = np.array([k[0][0] == "singles" for k in K])
        u = sum(R[n][s][:, si[s], 0] for s in JUDGE)
        b = sum(R[n][s][:, si[s], 1] for s in JUDGE)
        ok = isk & (b >= 150)
        roi = u[ok] / b[ok]
        out["variants"][n]["singles_heldout"] = dict(setups=int(ok.sum()), share_positive=float((roi > 0).mean()), median=float(np.median(roi)), best=float(roi.max()))
        isp = ~isk & (b >= 60)
        rp = u[isp] / b[isp]
        out["variants"][n]["parlays_heldout"] = dict(setups=int(isp.sum()), share_positive=float((rp > 0).mean()), median=float(np.median(rp)))
    (OUTD / "tournament4_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nstraight bets on held-out seasons, per variant (setups with 150+ bets): share positive / median / best")
    for n in names:
        x, y = out["variants"][n]["singles_heldout"], out["variants"][n]["parlays_heldout"]
        print(f"  {n:<16} singles {x['share_positive']:.0%} / {x['median']:+.1%} / {x['best']:+.1%}   parlays {y['share_positive']:.0%} / {y['median']:+.1%}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

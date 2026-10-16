"""Mark 2 vs Mark 3, head to head (yours, 2026-10-16: "is Mark 3 better than Mark 2? should we make it active?").

Same weeks, same rule (up to 4 boosted 2-leg parlays from the top 8; straight bets = top 8, for reference), two pickers: Mark 2 (locked) and Mark 3 (plus schedule
inputs). Two tests:
  hidden   2021-2026, each season's picks from a picker trained on the other five seasons (as tournament 3 / ideas batch 1)
  replay   2024 week 4 - 2026 week 5, the picker retrained every week on earlier weeks only (the replay pools)
For each: return per unit, and a paired week bootstrap (weeks resampled, both marks together) for the chance Mark 3's return per unit is higher.

  python scripts/compare_marks.py      # writes records/tuning/mark_compare.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


IB = _load("ideas_batch")
TN, CF, T2, ST = IB.TN, IB.CF, IB.T2, IB.ST
FOUR = IB.FOUR


def weekly(W, key="chance", dec="dec", won="won", player="player_id"):
    """{(season, week): (parlay units, parlays, single units, singles)} under the Mark 2 rule."""
    out = {}
    for sw, rows in W.items():
        legs = ST.top(rows, 8, ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], chance=key, player=player)
        pu = pn = 0.0
        for a, b in ST.pairs(legs):
            pu += (a[dec] * b[dec] - 1) * (1 + ST.BOOST) if a[won] == 1 and b[won] == 1 else -1.0
            pn += 1
        su = sn = 0.0
        for r in ST.top(rows, 8, ST.SINGLES_CAPS["per_game"], None, chance=key, player=player):
            su += (r[dec] - 1) if r[won] == 1 else -1.0
            sn += 1
        out[sw] = (pu, pn, su, sn)
    return out


def compare(a, b, kind, n_boot=4000, seed=7):
    """Return per unit of each, and the paired-week bootstrap chance that b beats a."""
    i = 0 if kind == "parlays" else 2
    weeks = sorted(set(a) | set(b))
    A = np.array([[a.get(w, (0, 0, 0, 0))[i], a.get(w, (0, 0, 0, 0))[i + 1]] for w in weeks])
    B = np.array([[b.get(w, (0, 0, 0, 0))[i], b.get(w, (0, 0, 0, 0))[i + 1]] for w in weeks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(weeks), (n_boot, len(weeks)))
    ra = A[idx, 0].sum(1) / np.maximum(A[idx, 1].sum(1), 1)
    rb = B[idx, 0].sum(1) / np.maximum(B[idx, 1].sum(1), 1)
    diff = rb - ra
    seasons = {}
    for w in weeks:
        s = str(w[0])
        x = seasons.setdefault(s, [0, 0, 0, 0])
        x[0] += A[weeks.index(w), 0]; x[1] += A[weeks.index(w), 1]; x[2] += B[weeks.index(w), 0]; x[3] += B[weeks.index(w), 1]
    return dict(mark2=dict(units=float(A[:, 0].sum()), bets=int(A[:, 1].sum()), roi=float(A[:, 0].sum() / max(A[:, 1].sum(), 1))),
                mark3=dict(units=float(B[:, 0].sum()), bets=int(B[:, 1].sum()), roi=float(B[:, 0].sum() / max(B[:, 1].sum(), 1))),
                p_mark3_better=float((diff > 0).mean()), diff_ci90=[float(np.percentile(diff, 5)), float(np.percentile(diff, 95))],
                seasons={s: dict(mark2=v[0] / max(v[1], 1), mark3=v[2] / max(v[3], 1), mark2_bets=v[1], mark3_bets=v[3]) for s, v in sorted(seasons.items())},
                weeks=len(weeks))


def main():
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X = TN.feats(d, layered=True)
    sched, _ = IB.schedule_feats(d)
    makers = CF.ensemble_makers()
    h2 = weekly(IB.week_rows(d, A, IB.hidden_preds(d, A, X, makers)))
    h3 = weekly(IB.week_rows(d, A, IB.hidden_preds(d, A, np.hstack([X, sched]), makers)))
    out = {"hidden": {k: compare(h2, h3, k) for k in ("parlays", "singles")}}

    def pool_weeks(f):
        p = pl.read_parquet(config.PROCESSED_DIR / f).filter(pl.col("market").is_in(FOUR))
        W = {}
        for r in p.sort("picker_under_chance", descending=True).iter_rows(named=True):
            W.setdefault((r["season"], r["week"]), []).append(r)
        return W
    r2 = weekly(pool_weeks("replay_pool_uncapped.parquet"), key="picker_under_chance", dec="dec_u")
    r3 = weekly(pool_weeks("replay_pool_schedule.parquet"), key="picker_under_chance", dec="dec_u")
    out["replay"] = {k: compare(r2, r3, k) for k in ("parlays", "singles")}
    (config.ROOT / "records" / "tuning" / "mark_compare.json").write_text(json.dumps(out, indent=1))
    for t, v in out.items():
        for k, c in v.items():
            print(f"{t:<7}{k:<8} Mark 2 {c['mark2']['roi']:+.1%} on {c['mark2']['bets']}  Mark 3 {c['mark3']['roi']:+.1%} on {c['mark3']['bets']}  "
                  f"P(Mark 3 better) {c['p_mark3_better']:.0%}  diff 90% range {c['diff_ci90'][0]:+.0%} to {c['diff_ci90'][1]:+.0%}")


if __name__ == "__main__":
    main()

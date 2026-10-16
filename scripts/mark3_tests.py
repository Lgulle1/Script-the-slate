"""Mark 2 vs Mark 3, more tests, and ways to improve Mark 3 (yours, 2026-10-16: "run more tests to see who's better, and see if there are other improvements
to Mark 3"). Every test uses the hidden-season protocol (each season 2021-2026 picked by a picker trained on the other five) unless it says otherwise.

Who is better:
  seeds     the picker's random seed changes which validation rows its early stopping uses; 5 seeds per picker, so luck in one fit cannot decide it
  rules     the same comparison under 18 nearby betting rules (edge cut-off 2/4/6 points, top 6/8/10, 2-leg or 3-leg parlays): does Mark 3 win under most?
  blind     2021-2023 picked by a picker trained only on LATER seasons (as the original blind tests)
  markets   Mark 3 vs Mark 2 by market (straight bets, top 8)
Improving Mark 3 (each against Mark 3 as it stands):
  drop one  remove one schedule input at a time (rest days / short week / division game / primetime / turf): which ones carry it
  travel    add time-zone change for the visitors, west-coast team in an early eastern kickoff, coming off a bye, neutral site (London etc.)
  weather   add temperature and wind; movement: add FanDuel price movement; 3-leg: Mark 3's picks as 3-leg parlays

  python scripts/mark3_tests.py      # writes records/tuning/mark3_tests.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
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


IB, CM = _load("ideas_batch"), _load("compare_marks")
TN, CF, T2, ST = IB.TN, IB.CF, IB.T2, IB.ST
SEASONS = IB.SEASONS
TZ = {**{t: 0 for t in "BUF MIA NE NYJ NYG PHI WAS BAL CIN CLE PIT IND JAX TB ATL CAR DET".split()},
      **{t: -1 for t in "TEN HOU KC CHI GB MIN NO DAL".split()}, **{t: -2 for t in "DEN ARI".split()}, **{t: -3 for t in "LV LAC LA SF SEA".split()}}
SCHED_GROUPS = {"rest days": [0, 1, 2], "short week": [3], "division game": [4], "primetime": [5], "turf/grass": [6]}


def makers(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [lambda d=d, l=l: HistGradientBoostingClassifier(max_depth=d, learning_rate=0.1, max_iter=200, min_samples_leaf=l, l2_regularization=1.0, random_state=seed)
            for d in (2, 3, 4) for l in (50, 150)]


def travel_feats(d):
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT DISTINCT ON (game_id) game_id, gametime, location, away_rest, home_rest, home_team FROM schedules ORDER BY game_id, pulled_at DESC").pl()
    con.close()
    x = d.select("game_id", "team", "is_home").join(sc, on="game_id", how="left")
    h = x["is_home"].fill_null(0).to_numpy() == 1
    team_tz = np.array([TZ.get(t, 0) for t in x["team"].to_list()], dtype=float)
    game_tz = np.array([TZ.get(t, 0) for t in x["home_team"].fill_null("").to_list()], dtype=float)
    neutral = (x["location"] == "Neutral").fill_null(False).to_numpy()
    shift = np.where(h | neutral, 0.0, game_tz - team_tz)
    hour = np.array([int(str(t).split(":")[0]) if t else 13 for t in x["gametime"].to_list()])
    early_west = ((team_tz <= -2) & (hour <= 13) & (game_tz >= -1) & ~h).astype(float)
    rest = np.where(h, x["home_rest"].to_numpy(), x["away_rest"].to_numpy()).astype(float)
    return np.column_stack([shift, early_west, (rest >= 13).astype(float), neutral.astype(float)])


def blind_preds(d, A, X, mk):
    """2021-2023 each picked by a picker trained only on later seasons; other seasons left empty."""
    season = d["season"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    p = np.full(d.height, np.nan)
    for S in (2021, 2022, 2023):
        tr = (season > S) & graded
        te = season == S
        p[te] = np.mean([m().fit(X[tr], A.over_won[tr]).predict_proba(X[te])[:, 1] for m in mk], axis=0)
    return p


def rule_roi(W, edge=0.04, n=8, k=2):
    rows = {sw: [r for r in rr if r["edge"] >= edge] for sw, rr in W.items()}
    s = IB.summary(IB.run_rule(rows, n=n, k=k))
    return s["roi"], s["stake"]


def main():
    t0 = time.time()
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X = TN.feats(d, layered=True)
    sched, wx = IB.schedule_feats(d)
    trav = travel_feats(d)
    mv = IB.movement_feats(d)
    X3 = np.hstack([X, sched])
    out = {}
    log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

    def evaluate(Xv, mk):
        """(picks with 2+ points of edge for the rule grid, weekly results and summary under the locked 4-point rule)."""
        W0 = IB.week_rows(d, A, IB.hidden_preds(d, A, Xv, mk), min_edge=0.02)
        W = {sw: [r for r in rr if r["edge"] >= 0.04] for sw, rr in W0.items()}
        W = {sw: rr for sw, rr in W.items() if rr}
        wk = CM.weekly(W)
        r = IB.summary(IB.run_rule(W))
        return W0, wk, dict(parlays=r["roi"], parlay_stake=r["stake"], seasons={s: v["roi"] for s, v in r["seasons"].items()}, singles=IB.singles(W))

    # 1. seeds
    seeds = {}
    keep = {}
    for seed in (7, 1, 2, 3, 4):
        W2, wk2, r2 = evaluate(X, makers(seed))
        W3, wk3, r3 = evaluate(X3, makers(seed))
        c = CM.compare(wk2, wk3, "parlays")
        cs = CM.compare(wk2, wk3, "singles")
        seeds[seed] = dict(mark2=r2, mark3=r3, p_parlays=c["p_mark3_better"], p_singles=cs["p_mark3_better"])
        if seed == 7:
            keep = dict(W2=W2, W3=W3, wk3=wk3, r3=r3)
        log(f"seed {seed}: parlays Mark 2 {r2['parlays']:+.1%} Mark 3 {r3['parlays']:+.1%} (P {c['p_mark3_better']:.0%})  singles {r2['singles']:+.1%} vs {r3['singles']:+.1%} (P {cs['p_mark3_better']:.0%})")
    out["seeds"] = seeds
    # 2. rules
    rules = {}
    for edge in (0.02, 0.04, 0.06):
        for n in (6, 8, 10):
            for k in (2, 3):
                if k == 3 and n == 8:
                    n3 = 9
                else:
                    n3 = n
                a, sa = rule_roi(keep["W2"], edge, n3, k)
                b, sb = rule_roi(keep["W3"], edge, n3, k)
                rules[f"edge {int(edge * 100)}, top {n3}, {k}-leg"] = dict(mark2=a, mark3=b, mark2_stake=sa, mark3_stake=sb)
    out["rules"] = rules
    log(f"rules: Mark 3 ahead in {sum(v['mark3'] is not None and v['mark2'] is not None and v['mark3'] > v['mark2'] for v in rules.values())} of {len(rules)}")
    # 3. blind 2021-2023, picker trained on later seasons only
    bl = {}
    for name, Xv in (("mark2", X), ("mark3", X3)):
        W = IB.week_rows(d, A, blind_preds(d, A, Xv, makers(7)))
        W = {sw: r for sw, r in W.items() if sw[0] <= 2023}
        r = IB.summary(IB.run_rule(W))
        bl[name] = dict(parlays=r["roi"], stake=r["stake"], seasons={s: v["roi"] for s, v in r["seasons"].items()}, singles=IB.singles(W))
    out["blind"] = bl
    log(f"blind 2021-23: parlays Mark 2 {bl['mark2']['parlays']:+.1%} Mark 3 {bl['mark3']['parlays']:+.1%}; singles {bl['mark2']['singles']:+.1%} vs {bl['mark3']['singles']:+.1%}")
    # 4. markets (straight bets, top 8)
    mk_out = {}
    for name, W in (("mark2", keep["W2"]), ("mark3", keep["W3"])):
        for m in IB.FOUR:
            U = B = 0
            for rows in W.values():
                for r in ST.top([x for x in rows if x["market"] == m and x["edge"] >= 0.04], 8, ST.SINGLES_CAPS["per_game"], None, chance="chance", player="player_id"):
                    U += (r["dec"] - 1) if r["won"] == 1 else -1.0
                    B += 1
            mk_out.setdefault(m, {})[name] = dict(roi=U / B if B else None, bets=B)
    out["markets"] = mk_out
    # 5. improving Mark 3
    imp = {"Mark 3 as it stands": keep["r3"]}
    for g, cols in SCHED_GROUPS.items():
        keepcols = [c for c in range(sched.shape[1]) if c not in cols]
        _, wk, r = evaluate(np.hstack([X, sched[:, keepcols]]), makers(7))
        r["p_better_than_mark3"] = CM.compare(keep["wk3"], wk, "parlays")["p_mark3_better"]
        imp[f"without {g}"] = r
        log(f"without {g}: parlays {r['parlays']:+.1%} singles {r['singles']:+.1%}")
    for name, extra in (("plus travel", trav), ("plus weather", wx), ("plus price movement", mv), ("plus travel + weather", np.hstack([trav, wx]))):
        _, wk, r = evaluate(np.hstack([X3, extra]), makers(7))
        r["p_better_than_mark3"] = CM.compare(keep["wk3"], wk, "parlays")["p_mark3_better"]
        imp[name] = r
        log(f"{name}: parlays {r['parlays']:+.1%} singles {r['singles']:+.1%} (P better than Mark 3 {r['p_better_than_mark3']:.0%})")
    r3leg = IB.summary(IB.run_rule({sw: [r for r in rr if r["edge"] >= 0.04] for sw, rr in keep["W3"].items()}, n=9, k=3))
    imp["3-leg parlays from the top 9"] = dict(parlays=r3leg["roi"], parlay_stake=r3leg["stake"], seasons={s: v["roi"] for s, v in r3leg["seasons"].items()})
    base_seasons = keep["r3"]["seasons"]
    for k, v in imp.items():
        v["seasons_beating_mark3"] = sum(1 for s, x in v["seasons"].items() if x is not None and base_seasons.get(s) is not None and x > base_seasons[s])
    out["improve_mark3"] = imp
    (config.ROOT / "records" / "tuning" / "mark3_tests.json").write_text(json.dumps(out, indent=1, default=str))
    log("done")


if __name__ == "__main__":
    main()

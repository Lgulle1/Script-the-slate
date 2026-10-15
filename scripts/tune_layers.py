"""Everything at once, per market: the model's own layer weights (injuries, opponent defence, recent form, team offence, role), the model/book mix, the shift on the over's
chance, the edge cut-off, the side and one extra filter. Judged on the half it was not tuned on: tuned on 2025 weeks 1-9 and scored on 2025 weeks 10-18 + 2026 weeks 1-5,
and the reverse. 2024 is not used here: its predictions are the base model only and have no layer breakdown.

  python scripts/tune_layers.py
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import calibrate as C

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP, TM, LS = _load("layer_parts"), _load("tune_more"), _load("line_study")
G = LP.GROUPS
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0)
WS, BS, THS, SIDES = (0.3, 0.5, 0.7, 1.0), (-0.08, -0.04, 0.0, 0.04), (0.04, 0.06, 0.08, 0.10), ("both", "over", "under")
MIN_BETS = 40
OUTD = config.ROOT / "records" / "tuning"
FILTERS = [("model 5%+ under the line", lambda d: d["gap"] <= -0.05), ("model 10%+ under the line", lambda d: d["gap"] <= -0.10),
           ("model 5%+ over the line", lambda d: d["gap"] >= 0.05), ("game total <= 42.5", lambda d: d["total_line"] <= 42.5), ("game total >= 47.5", lambda d: d["total_line"] >= 47.5),
           ("team favoured by 3+", lambda d: d["team_spread"] >= 3), ("team underdog by 3+", lambda d: d["team_spread"] <= -3), ("home", lambda d: d["is_home"] == 1),
           ("away", lambda d: d["is_home"] == 0), ("dome", lambda d: d["dome"] == 1), ("outdoors", lambda d: d["dome"] == 0)]


def assemble() -> dict:
    """Per market: arrays of everything a setup needs, one row per priced line of 2025-26."""
    d = TM.data().filter(pl.col("season") >= 2025)
    sp = LP.stored_preds()
    p = LP.parts()
    d = d.join(sp.select("game_id", "player_id", "market", "stack"), on=["game_id", "player_id", "market"], how="inner")
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    out = {}
    for m, qs in LP.QTY.items():
        x = d.filter(pl.col("market") == m).sort("season", "week", "game_id", "player_id", "line")
        comps = []
        for q in qs:
            pq = p.filter(pl.col("quantity") == q).select("game_id", "player_id", "stack", "baseline", *G)
            j = x.select("game_id", "player_id", "stack").join(pq, on=["game_id", "player_id", "stack"], how="left", maintain_order="left")
            comps.append(np.column_stack([j["baseline"].fill_null(0).to_numpy()] + [j[g].fill_null(0).to_numpy() for g in G]))
        v = cal.markets[m]
        a = {c: x[c].to_numpy() for c in ("season", "week", "line", "fair_o", "be_o", "be_u", "dec_o", "dec_u", "over_won", "gap", "total_line", "team_spread", "is_home", "dome", "mean")}
        a.update(game_id=x["game_id"].to_numpy(), player=x["player_name"].to_numpy(), team=x["team"].to_numpy(), gameday=x["gameday"].cast(pl.String).to_numpy(),
                 comps=comps, pool=(v["pred"], v["act"], v["c"] if v["use_width"] else 1.0), n=x.height)
        out[m] = a
    return out


def sub(a: dict, mask: np.ndarray) -> dict:
    b = {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == (a["n"],) else v) for k, v in a.items()}
    b["comps"] = [c[mask] for c in a["comps"]]
    b["n"] = int(mask.sum())
    return b


def mean_of(a, alpha):
    w = np.array([1.0, *alpha])
    return np.prod([c @ w for c in a["comps"]], axis=0)


def p_over(a, alpha):
    pred, act, c0 = a["pool"]
    return LS.chances(pred, act, c0, mean_of(a, alpha), a["line"], 600, 1.0)


def bets(a, po, w, b, th, side, filt=None):
    p = 1 / (1 + np.exp(-(w * LS.logit(po) + (1 - w) * LS.logit(a["fair_o"])))) + b
    eo, eu = p - a["be_o"], (1 - p) - a["be_u"]
    over = eo >= eu
    m = np.maximum(eo, eu) >= th
    if side != "both":
        m &= over if side == "over" else ~over
    if filt is not None:
        m &= np.asarray(filt(a))
    won = np.where(over, a["over_won"], 1 - a["over_won"])
    dec = np.where(over, a["dec_o"], a["dec_u"])
    return m, over, np.where(won == 1, dec - 1, -1.0), won, dec


def best_betting(a, po, filt=None):
    best = None
    for w, b, th, sd in itertools.product(WS, BS, THS, SIDES):
        m, _, prof, _, _ = bets(a, po, w, b, th, sd, filt)
        if m.sum() >= MIN_BETS:
            r = prof[m].mean()
            if best is None or r > best[0]:
                best = (r, dict(w=w, b=b, th=th, side=sd))
    return best


def tune_market(a, layers=True, filters=True):
    alpha = [1.0] * len(G)
    po = p_over(a, alpha)
    best = best_betting(a, po)
    if best is None:
        return None
    if layers:
        for _ in range(2):
            for gi in range(len(G)):
                for v in ALPHAS:
                    trial = alpha.copy()
                    trial[gi] = v
                    bb = best_betting(a, p_over(a, trial))
                    if bb and bb[0] > best[0] + 1e-9:
                        best, alpha = bb, trial
    st = dict(best[1], alpha=alpha, filt=None, filt_name="none", roi=best[0])
    if filters:
        po = p_over(a, alpha)
        for nm, f in FILTERS:
            bb = best_betting(a, po, f)
            if bb and bb[0] > st["roi"] + 0.01:
                st = dict(bb[1], alpha=alpha, filt=f, filt_name=nm, roi=bb[0])
    return st


def describe(m, st):
    lay = ", ".join(f"{g} x{v:g}" for g, v in zip(G, st["alpha"]) if v != 1.0) or "all layers x1"
    return f"{m}: {lay}; model {st['w']:.0%} / book {1 - st['w']:.0%}; shift {st['b']:+.0%}; edge >= {st['th']:.0%}; {st['side']}; filter: {st['filt_name']}"


VARIANTS = {"per-market betting only": dict(layers=False, filters=False), "per-market betting + layer weights": dict(layers=True, filters=False),
            "EVERYTHING per market (layers + mix + shift + edge + side + filter)": dict(layers=True, filters=True)}


def run():
    A = assemble()
    halves = {"2025 wk 1-9": lambda a: (a["season"] == 2025) & (a["week"] <= 9), "2025 wk 10-18 + 2026": lambda a: (a["season"] == 2026) | (a["week"] >= 10)}
    rows, allbets, final = [], [], {}
    for vname, opt in VARIANTS.items():
        res = {}
        for tr, te in (("2025 wk 1-9", "2025 wk 10-18 + 2026"), ("2025 wk 10-18 + 2026", "2025 wk 1-9"), ("all", "all")):
            P, W, U, GID = [], [], [], []
            descs = []
            for m, a in A.items():
                atr = a if tr == "all" else sub(a, halves[tr](a))
                ate = a if te == "all" else sub(a, halves[te](a))
                st = tune_market(atr, **opt)
                if st is None or st["roi"] <= 0:
                    continue
                descs.append(describe(m, st))
                mk, over, prof, won, dec = bets(ate, p_over(ate, st["alpha"]), st["w"], st["b"], st["th"], st["side"], st["filt"])
                P.append(prof[mk]); W.append(won[mk]); GID.append(ate["game_id"][mk]); U.append(mk.sum())
                if tr != "all":
                    allbets.append(pl.DataFrame(dict(setup=vname, tuned_on=tr, scored_on=te, season=ate["season"][mk], week=ate["week"][mk], gameday=ate["gameday"][mk],
                                                     player=ate["player"][mk], team=ate["team"][mk], market=m, line=ate["line"][mk], bet=np.where(over[mk], "OVER", "UNDER"),
                                                     price=np.where(dec[mk] >= 2, np.round((dec[mk] - 1) * 100), np.round(-100 / (dec[mk] - 1))),
                                                     model_prediction_reweighted=np.round(mean_of(ate, st["alpha"])[mk], 2), result=np.where(won[mk] == 1, "WIN", "LOSS"),
                                                     profit_units=np.round(prof[mk], 3), rule=describe(m, st))))
                if tr == "all" and vname.startswith("EVERYTHING"):
                    final[m] = describe(m, st)
            p, w, g = (np.concatenate(P), np.concatenate(W), np.concatenate(GID)) if P else (np.array([]), np.array([]), np.array([]))
            res[(tr, te)] = (p, w, g, descs)
        pi, wi, _, di = res[("all", "all")]
        pt = np.concatenate([res[k][0] for k in res if k[0] != "all"])
        wt = np.concatenate([res[k][1] for k in res if k[0] != "all"])
        gt = np.concatenate([res[k][2] for k in res if k[0] != "all"])
        lo, hi = TM.boot(pt, gt) if len(pt) > 20 else (np.nan, np.nan)
        rows.append(dict(setup=vname, insample_bets=len(pi), insample_hit=float(wi.mean()), insample_roi=float(pi.mean()), test_bets=len(pt), test_hit=float(wt.mean()) if len(wt) else None,
                         test_roi=float(pt.mean()) if len(pt) else None, test_lo=lo, test_hi=hi, test_units=float(pt.sum())))
        r = rows[-1]
        print(f"{vname}\n   same data: {r['insample_bets']} bets, hit {r['insample_hit']:.1%}, ROI {r['insample_roi']:+.1%}  |  TEST: {r['test_bets']} bets, hit {r['test_hit']:.1%}, ROI {r['test_roi']:+.1%} [{lo:+.1%}, {hi:+.1%}]")
        for k in res:
            if k[0] != "all":
                p, w = res[k][0], res[k][1]
                print(f"     tuned {k[0]} -> scored {k[1]}: {len(p)} bets, hit {w.mean() if len(w) else 0:.1%}, ROI {p.mean() if len(p) else 0:+.1%}")
    print("\nEVERYTHING per market, tuned on all of 2025-26 (the setup itself):")
    for m, s in final.items():
        print("  ", s)
    OUTD.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTD / "layers_summary.csv")
    pl.concat(allbets).write_csv(OUTD / "layers_test_bets.csv")
    (OUTD / "layers_everything_setup.txt").write_text("\n".join(final.values()) + "\n")


if __name__ == "__main__":
    run()

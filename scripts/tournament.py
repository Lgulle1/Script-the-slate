"""The tournament (yours, 2026-10-15): tune on TRAINING = 2024 + 2025 weeks 1-9, take the best candidates of every family, run them on TEST A = 2025 weeks 10-18
and TEST B = 2026 weeks 1-5, keep refining around the leaders, and log every candidate that was tried (records/tuning/tournament_*).

Families (every one scored the same way, on the same lines, with the same money rules):
  L  layer weights tuned for ACCURACY per market on the 2025 training weeks (every increment kept), giving a second, "layered" model chance for 2025-26
  F1 one global setup: model chance source (9 pool/width versions + layered) x model/book mix x shift x edge cut-off x side x every subset of the 7 markets
  F2 the best F1 setups plus one or two extra filters (gap, game total, spread, home/away, dome, line size, price)
  F3 machine-learning pickers (gradient boosting, several sizes) and F4 line-fed logistic stackers; training score is out-of-fold
  F5 per-market: each market takes its own best setup
  R  refinement: finer grids around the overall leaders

  python scripts/tournament.py
"""
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import calibrate as C

HERE = Path(__file__).resolve().parent
OUTD = config.ROOT / "records" / "tuning"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TM, LP, LS = _load("tune_more"), _load("layer_parts"), _load("line_study")
MARKETS = ["pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "rec", "rec_yds"]
SEGS = ["train", "A", "B"]
SUBSETS = np.array([[(i >> j) & 1 for j in range(7)] for i in range(1, 128)], float)       # 127 x 7
SUBSET_NAMES = [",".join(MARKETS[j] for j in range(7) if (i >> j) & 1) for i in range(1, 128)]
THETAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
SIDES = ("both", "over", "under")
MIN_TRAIN = 300
LOG = []


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.append(msg)


# ------------------------------------------------------------------ data and the layered chance

def tune_layers_for_accuracy(train_mask_2025: pl.DataFrame):
    """Per market: layer weights (0-2 in steps of 0.25, two passes) that minimise the average miss on the 2025 training weeks; returns weights and the miss before/after per segment."""
    from live import results as RS
    p, sp = LP.parts(), LP.stored_preds()
    act = pl.concat([RS.played(s)[0] for s in (2025, 2026)]).filter(pl.col("player_id").is_not_null()).select("game_id", "player_id", "market", "actual")
    seg = lambda df: df.with_columns(seg=pl.when(pl.col("game_id").str.slice(0, 4) == "2026").then(pl.lit("B"))
                                     .when(pl.col("game_id").str.slice(5, 2).cast(pl.Int32) >= 10).then(pl.lit("A")).otherwise(pl.lit("train")))
    grid = np.arange(0, 2.01, 0.25)
    weights, rows = {}, []
    for m in MARKETS:
        spm = sp.filter(pl.col("market") == m)
        base = seg(LP.market_means(p, spm).join(act, on=["game_id", "player_id", "market"]).drop_nulls("actual"))

        def mae(wd, s):
            x = seg(LP.market_means(p, spm, wd).join(act, on=["game_id", "player_id", "market"]).drop_nulls("actual")).filter(pl.col("seg") == s)
            return float((x["mean_w"] - x["actual"]).abs().mean())
        wd = {g: 1.0 for g in LP.GROUPS}
        best = mae(wd, "train")
        for _ in range(2):
            for g in LP.GROUPS:
                for v in grid:
                    t = dict(wd, **{g: float(v)})
                    e = mae(t, "train")
                    if e < best - 1e-9:
                        best, wd = e, t
        weights[m] = wd
        r = dict(market=m, weights=", ".join(f"{g} x{v:g}" for g, v in wd.items() if v != 1.0) or "unchanged")
        for s in SEGS:
            b = base.filter(pl.col("seg") == s)
            r[f"miss_{s}_before"] = float((b["mean_w"] - b["actual"]).abs().mean()) if b.height else None
            r[f"miss_{s}_after"] = mae(wd, s) if b.height else None
        rows.append(r)
    means = pl.concat([LP.market_means(p, sp.filter(pl.col("market") == m), weights[m]) for m in MARKETS])
    return weights, pl.DataFrame(rows), means


def build():
    d = TM.data()
    d = d.with_columns(seg=pl.when(pl.col("season") == 2026).then(pl.lit("B")).when((pl.col("season") == 2025) & (pl.col("week") >= 10)).then(pl.lit("A")).otherwise(pl.lit("train")))
    say(f"lines: {d.height} (training {d.filter(pl.col('seg') == 'train').height}, test A {d.filter(pl.col('seg') == 'A').height}, test B {d.filter(pl.col('seg') == 'B').height})")
    for s in SEGS:
        say(f"   {s} by market:", d.filter(pl.col("seg") == s).group_by("market").len().sort("market").rows())
    w, tab, means = tune_layers_for_accuracy(d)
    say("\n[L] layer weights tuned for accuracy on the 2025 training weeks (average miss before -> after):")
    for r in tab.iter_rows(named=True):
        f = lambda s: f"{r[f'miss_{s}_before']:.3f}->{r[f'miss_{s}_after']:.3f}" if r[f"miss_{s}_before"] is not None else "n/a"
        say(f"   {r['market']:9s} {r['weights']:60s} train {f('train')}  A {f('A')}  B {f('B')}")
    d = d.join(means.rename({"mean_w": "mean_layered"}), on=["game_id", "player_id", "market"], how="left").with_columns(mean_layered=pl.coalesce("mean_layered", "mean"))
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    pl_ = np.full(d.height, np.nan)
    mk = d["market"].to_numpy()
    for m in MARKETS:
        i = np.flatnonzero(mk == m)
        v = cal.markets[m]
        pl_[i] = LS.chances(v["pred"], v["act"], v["c"] if v["use_width"] else 1.0, d["mean_layered"].to_numpy()[i], d["line"].to_numpy()[i], 600, 1.0)
    # 2024 keeps its own strictly-earlier-pool chance (no layer breakdown exists for it)
    d = d.with_columns(p_over_layered=pl.when(pl.col("season") == 2024).then(pl.col("p_over_k600_c100")).otherwise(pl.Series(pl_)))
    tab.write_csv(OUTD / "tournament_layer_weights.csv")
    return d


# ------------------------------------------------------------------ the common scorer

class Arr:
    def __init__(self, d):
        self.d = d
        self.mi = np.array([MARKETS.index(x) for x in d["market"].to_list()])
        self.si = np.array([SEGS.index(x) for x in d["seg"].to_list()])
        self.key = self.mi * 3 + self.si
        for c in ("fair_o", "be_o", "be_u", "dec_o", "dec_u", "over_won"):
            setattr(self, c, d[c].to_numpy())


def settle(A, p_over):
    eo, eu = p_over - A.be_o, (1 - p_over) - A.be_u
    over = eo >= eu
    won = np.where(over, A.over_won, 1 - A.over_won)
    dec = np.where(over, A.dec_o, A.dec_u)
    return over, np.maximum(eo, eu), np.where(won == 1, dec - 1, -1.0), won


def grid_stats(A, over, edge, prof, won, extra=None):
    """For every (theta, side): per market and segment -> bets, profit, wins. Shape (7 thetas, 3 sides, 7 markets, 3 segs, 3)."""
    out = np.zeros((len(THETAS), 3, 7, 3, 3))
    for ti, th in enumerate(THETAS):
        base = edge >= th
        if extra is not None:
            base = base & extra
        for si, sd in enumerate(SIDES):
            m = base & (over if sd == "over" else (~over if sd == "under" else True))
            k = A.key[m]
            out[ti, si, :, :, 0] = np.bincount(k, minlength=21).reshape(7, 3)
            out[ti, si, :, :, 1] = np.bincount(k, weights=prof[m], minlength=21).reshape(7, 3)
            out[ti, si, :, :, 2] = np.bincount(k, weights=won[m], minlength=21).reshape(7, 3)
    return out


def subsets(G):
    """(7, 3, 7, 3, 3) -> (7 thetas, 3 sides, 127 subsets, 3 segs, 3)."""
    return np.einsum("um,tsmgk->tsugk", SUBSETS, G)


def rows_from(fam, desc_fn, S, min_train=MIN_TRAIN):
    """Candidate rows with at least min_train training bets."""
    n_tr = S[..., 0, 0]
    idx = np.argwhere(n_tr >= min_train)
    out = []
    for ti, si, ui in idx:
        v = S[ti, si, ui]
        out.append(dict(family=fam, setup=desc_fn(THETAS[ti], SIDES[si], SUBSET_NAMES[ui]),
                        **{f"{s}_bets": int(v[g, 0]) for g, s in enumerate(SEGS)}, **{f"{s}_units": float(v[g, 1]) for g, s in enumerate(SEGS)},
                        **{f"{s}_wins": int(v[g, 2]) for g, s in enumerate(SEGS)}))
    return out


def finish(df: pl.DataFrame) -> pl.DataFrame:
    for s in SEGS:
        df = df.with_columns((pl.col(f"{s}_units") / pl.col(f"{s}_bets")).alias(f"{s}_roi"), (pl.col(f"{s}_wins") / pl.col(f"{s}_bets")).alias(f"{s}_hit"))
    df = df.with_columns(test_bets=pl.col("A_bets") + pl.col("B_bets"), test_units=pl.col("A_units") + pl.col("B_units"), test_wins=pl.col("A_wins") + pl.col("B_wins"))
    df = df.with_columns(test_roi=pl.col("test_units") / pl.col("test_bets"), test_hit=pl.col("test_wins") / pl.col("test_bets"),
                         all_bets=pl.col("train_bets") + pl.col("test_bets"), all_units=pl.col("train_units") + pl.col("test_units"))
    return df.with_columns(all_roi=pl.col("all_units") / pl.col("all_bets"))


# ------------------------------------------------------------------ families

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


SOURCES = [f"p_over_k{k}_c{c}" for k in (300, 600, 1200) for c in (85, 100, 114)] + ["p_over_layered"]
WS = np.round(np.arange(0, 1.01, 0.1), 2)
BS = (-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04)


def family_F1(A, d):
    rows = []
    t0 = time.time()
    for src in SOURCES:
        pm = d[src].to_numpy()
        lf, lm = logit(A.fair_o), logit(pm)
        for w in WS:
            for b in BS:
                p = 1 / (1 + np.exp(-(w * lm + (1 - w) * lf))) + b
                S = subsets(grid_stats(A, *settle(A, p)))
                rows += rows_from("F1 global setup", lambda th, sd, ms, src=src, w=w, b=b: f"chance={src.replace('p_over_', '')} | model {w:.0%}/book {1 - w:.0%} | shift {b:+.0%} | edge>={th:.0%} | {sd} | {ms}", S)
    say(f"[F1] {len(SOURCES) * len(WS) * len(BS) * len(THETAS) * 3 * 127:,} global setups tried ({len(rows):,} with >= {MIN_TRAIN} training bets) in {time.time() - t0:.0f}s")
    return rows


def parse_F1(setup):
    parts = [x.strip() for x in setup.split("|")]
    src = "p_over_" + parts[0].split("=")[1]
    w = float(parts[1].split("model ")[1].split("%")[0]) / 100
    b = float(parts[2].split("shift ")[1].rstrip("%")) / 100
    th = float(parts[3].split(">=")[1].rstrip("%")) / 100
    return src, w, b, th, parts[4], parts[5].split(",")


def mask_of(A, d, setup):
    src, w, b, th, sd, ms = parse_F1(setup)
    p = 1 / (1 + np.exp(-(w * logit(d[src].to_numpy()) + (1 - w) * logit(A.fair_o)))) + b
    over, edge, prof, won = settle(A, p)
    m = (edge >= th) & np.isin(A.mi, [MARKETS.index(x) for x in ms])
    if sd != "both":
        m &= over if sd == "over" else ~over
    return m, over, prof, won


def filters(d, A, over):
    sidebe = np.where(over, A.be_o, A.be_u)
    med = {"pass_att": 33.5, "pass_cmp": 21.5, "pass_yds": 230.5, "rush_att": 13.5, "rush_yds": 55.5, "rec": 3.5, "rec_yds": 40.5}
    below = d["line"].to_numpy() < np.array([med[m] for m in d["market"].to_list()])
    g, tot, spd, home, dome = (d[c].to_numpy() for c in ("gap", "total_line", "team_spread", "is_home", "dome"))
    return {"model 5%+ under line": g <= -0.05, "model 10%+ under line": g <= -0.10, "model 5%+ over line": g >= 0.05, "model 10%+ over line": g >= 0.10,
            "game total <=42.5": tot <= 42.5, "game total 43-47": (tot > 42.5) & (tot < 47.5), "game total >=47.5": tot >= 47.5,
            "team fav 3+": spd >= 3, "team dog 3+": spd <= -3, "spread within 3": np.abs(spd) < 3, "home": home == 1, "away": home == 0, "dome": dome == 1, "outdoors": dome == 0,
            "line below market median": below, "line above market median": ~below, "price -115 or better": sidebe <= 0.535, "plus money": sidebe <= 0.5, "price worse than -115": sidebe > 0.535}


def family_F2(A, d, top: pl.DataFrame):
    rows = []
    for setup in top["setup"].to_list():
        m, over, prof, won = mask_of(A, d, setup)
        F = filters(d, A, over)
        names = list(F)
        combos = [(n,) for n in names] + list(itertools.combinations(names, 2))
        for c in combos:
            k = m.copy()
            for n in c:
                k &= F[n]
            v = np.zeros((3, 3))
            for g in range(3):
                kk = k & (A.si == g)
                v[g] = (kk.sum(), prof[kk].sum(), won[kk].sum())
            if v[0, 0] >= 150:
                rows.append(dict(family="F2 setup + filters", setup=setup + " | filters: " + " & ".join(c), **{f"{s}_bets": int(v[i, 0]) for i, s in enumerate(SEGS)},
                                 **{f"{s}_units": float(v[i, 1]) for i, s in enumerate(SEGS)}, **{f"{s}_wins": int(v[i, 2]) for i, s in enumerate(SEGS)}))
    say(f"[F2] {len(top) * (19 + 171):,} filtered setups tried on the {len(top)} best F1 setups ({len(rows):,} with >= 150 training bets)")
    return rows


def feats(d, layered=True):
    mk = [(d["market"] == m).to_numpy().astype(float) for m in MARKETS]
    cols = [logit(d["fair_o"].to_numpy()), logit(d["p_over_k600_c100"].to_numpy()), logit(d["p_over_k300_c100"].to_numpy()), logit(d["p_over_k1200_c100"].to_numpy()),
            d["gap"].to_numpy(), d["line"].to_numpy(), d["be_o"].to_numpy(), d["be_u"].to_numpy(), d["total_line"].to_numpy(), d["team_spread"].to_numpy(),
            d["is_home"].fill_null(0.5).to_numpy(), d["week"].to_numpy(), d["dome"].fill_null(0).to_numpy(), *mk]
    if layered:
        cols.insert(2, logit(d["p_over_layered"].to_numpy()))
    return np.column_stack(cols)


def oof_and_fit(make, X, y, tr_idx, blocks):
    """Out-of-fold predictions on the training rows (5 folds by week blocks) and a model fit on all training rows for the test rows."""
    p = np.full(len(y), np.nan)
    ub = np.unique(blocks[tr_idx])
    folds = np.array_split(np.random.default_rng(7).permutation(ub), 5)
    for f in folds:
        te = tr_idx[np.isin(blocks[tr_idx], f)]
        trn = tr_idx[~np.isin(blocks[tr_idx], f)]
        p[te] = make().fit(X[trn], y[trn]).predict_proba(X[te])[:, 1]
    rest = np.setdiff1d(np.arange(len(y)), tr_idx)
    p[rest] = make().fit(X[tr_idx], y[tr_idx]).predict_proba(X[rest])[:, 1]
    return p


def family_F3_F4(A, d):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = A.over_won
    tr_idx = np.flatnonzero(A.si == 0)
    blocks = (d["season"] * 100 + d["week"]).to_numpy()
    rows, n = [], 0
    makers = {}
    for depth, lr, leaf in itertools.product((2, 3, 4), (0.03, 0.1), (50, 150)):
        makers[f"F3 boosted trees depth {depth} rate {lr} leaf {leaf}"] = lambda depth=depth, lr=lr, leaf=leaf: HistGradientBoostingClassifier(
            max_depth=depth, learning_rate=lr, max_iter=200, min_samples_leaf=leaf, l2_regularization=1.0, random_state=7)
    for c in (0.01, 0.1, 1.0):
        makers[f"F4 logistic stacker C={c}"] = lambda c=c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    for name, mk in makers.items():
        for layered in (False, True):
            X = feats(d, layered)
            p = oof_and_fit(mk, X, y, tr_idx, blocks)
            S = subsets(grid_stats(A, *settle(A, p)))
            fam = name.split(" ")[0] + (" ML picker" if name.startswith("F3") else " stacker")
            rows += rows_from(fam, lambda th, sd, ms, name=name, layered=layered: f"{name}{' +layers' if layered else ''} | edge>={th:.0%} | {sd} | {ms}", S)
            n += len(THETAS) * 3 * 127
    say(f"[F3/F4] {n:,} machine-learning / stacker setups tried ({len(makers) * 2} models x cut-offs x sides x market sets); training scores are out-of-fold")
    return rows


def family_F5(all_rows: pl.DataFrame):
    """Per market: the best single-market setup by training (>= 60 training bets); combined over markets that are profitable in training."""
    single = all_rows.filter(pl.col("family") == "F1 global setup")
    out, picks = [], []
    for m in MARKETS:
        s = single.filter(pl.col("setup").str.ends_with(f"| {m}") & (pl.col("train_bets") >= 60)).sort("train_roi", descending=True).head(3)
        if s.height and s["train_roi"][0] > 0:
            picks.append(s)
    if picks:
        for k in range(3):
            chosen = [p.row(min(k, p.height - 1), named=True) for p in picks]
            r = dict(family="F5 per-market", setup=f"rank-{k + 1} setup per market: " + " || ".join(c["setup"] for c in chosen))
            for col in ("train_bets", "train_units", "train_wins", "A_bets", "A_units", "A_wins", "B_bets", "B_units", "B_wins"):
                r[col] = sum(c[col] for c in chosen)
            out.append(r)
    say(f"[F5] {len(out)} per-market combinations")
    return out


def family_refine(A, d, leaders: pl.DataFrame):
    """Finer grids around each leader's chance source, mix, shift and cut-off; same markets and side."""
    rows = []
    for setup in leaders["setup"].to_list():
        src, w0, b0, th0, sd, ms = parse_F1(setup.split(" | filters")[0])
        for w in np.clip(np.round(np.arange(w0 - 0.1, w0 + 0.101, 0.025), 3), 0, 1):
            for b in np.round(np.arange(b0 - 0.02, b0 + 0.021, 0.005), 3):
                for th in np.round(np.arange(max(th0 - 0.02, 0), th0 + 0.021, 0.005), 3):
                    st = f"chance={src.replace('p_over_', '')} | model {w:.1%}/book {1 - w:.1%} | shift {b:+.1%} | edge>={th:.1%} | {sd} | {','.join(ms)}"
                    p = 1 / (1 + np.exp(-(w * logit(d[src].to_numpy()) + (1 - w) * logit(A.fair_o)))) + b
                    over, edge, prof, won = settle(A, p)
                    m = (edge >= th) & np.isin(A.mi, [MARKETS.index(x) for x in ms])
                    if sd != "both":
                        m &= over if sd == "over" else ~over
                    v = [(int((m & (A.si == g)).sum()), float(prof[m & (A.si == g)].sum()), int(won[m & (A.si == g)].sum())) for g in range(3)]
                    if v[0][0] >= MIN_TRAIN:
                        rows.append(dict(family="R refinement", setup=st, **{f"{s}_bets": v[i][0] for i, s in enumerate(SEGS)}, **{f"{s}_units": v[i][1] for i, s in enumerate(SEGS)},
                                         **{f"{s}_wins": v[i][2] for i, s in enumerate(SEGS)}))
    say(f"[R] {len(rows):,} refined setups around {leaders.height} leaders")
    return rows


# ------------------------------------------------------------------ report

def show(df, title, n=10):
    say(f"\n{title}")
    say(f"   {'train bets/hit/ROI':>24s} | {'test A (2025 wk10-18)':>24s} | {'test B (2026 wk1-5)':>22s} | setup")
    for r in df.head(n).iter_rows(named=True):
        f = lambda s: f"{r[f'{s}_bets']:5d} {r[f'{s}_hit'] or 0:5.1%} {r[f'{s}_roi'] or 0:+6.1%}" if r[f"{s}_bets"] else f"{0:5d}     -       -"
        say(f"   {f('train'):>24s} | {f('A'):>24s} | {f('B'):>22s} | {r['setup'][:150]}")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    d = build()
    A = Arr(d)
    rows = family_F1(A, d)
    F1 = finish(pl.DataFrame(rows))
    top_f1 = F1.sort("train_roi", descending=True).head(100)
    rows += family_F2(A, d, top_f1)
    rows += family_F3_F4(A, d)
    allr = finish(pl.DataFrame(rows, infer_schema_length=None))
    rows += family_F5(allr)
    allr = finish(pl.DataFrame(rows, infer_schema_length=None))

    say(f"\nTOTAL candidates tried: {sum(1 for _ in rows):,} with enough training bets (the grids themselves are larger; counts per family above)")
    # round 1: rank on training only, top 10 per family, see what they do on the tests
    for fam in allr["family"].unique().sort().to_list():
        show(allr.filter(pl.col("family") == fam).sort("train_roi", descending=True), f"ROUND 1 - {fam}: top 10 by TRAINING return, and what they did on the two tests")
    # does training rank predict test? (the question that decides whether searching works)
    big = allr.filter(pl.col("A_bets") >= 30)
    q = big.with_columns(r=pl.col("train_roi").rank(descending=True) / big.height)
    for lo, hi, nm in ((0, 0.001, "top 0.1% by training"), (0, 0.01, "top 1%"), (0, 0.1, "top 10%"), (0.45, 0.55, "middle"), (0.9, 1.0, "bottom 10%")):
        z = q.filter((pl.col("r") > lo) & (pl.col("r") <= hi))
        say(f"   {nm:24s}: {z.height:7d} setups, average test-A return {z['A_roi'].mean():+.2%}, share positive on A {(z['A_roi'] > 0).mean():.0%}")
    # round 2: keep the ones positive on training AND both tests, rank by everything combined, refine around the top 10, repeat once
    def leaders_of(df):
        return df.filter((pl.col("train_roi") > 0) & (pl.col("A_bets") >= 40) & (pl.col("A_roi") > 0) & ((pl.col("B_bets") == 0) | (pl.col("B_roi") > 0)) & (pl.col("all_bets") >= 500)
                         ).sort("all_units", descending=True)
    lead = leaders_of(allr)
    say(f"\nROUND 2: {lead.height:,} setups are positive on training AND test A AND test B (or have no B bets)")
    show(lead, "ROUND 2 leaders by total units over all three segments")
    f1lead = lead.filter(pl.col("family").is_in(["F1 global setup", "F2 setup + filters"])).head(10)
    rr = family_refine(A, d, f1lead)
    R = finish(pl.DataFrame(rr)) if rr else None
    if R is not None:
        rows += rr
        allr = finish(pl.DataFrame(rows, infer_schema_length=None))
        lead = leaders_of(allr)
        show(lead, "ROUND 3 after refinement: leaders by total units over all three segments")
    win = lead.head(1)
    if win.height:
        r = win.row(0, named=True)
        say(f"\nWINNER (most units over training + both tests, positive in each): {r['setup']}")
        say(f"   training {r['train_bets']} bets {r['train_hit']:.1%} {r['train_roi']:+.1%} | A {r['A_bets']} bets {r['A_hit']:.1%} {r['A_roi']:+.1%} | B {r['B_bets']} bets "
            f"{(r['B_hit'] or 0):.1%} {(r['B_roi'] or 0):+.1%} | all {r['all_bets']} bets {r['all_roi']:+.1%}, {r['all_units']:+.1f} units")
    allr.sort("train_roi", descending=True).write_parquet(OUTD / "tournament_all_candidates.parquet")
    allr.sort("all_units", descending=True).head(20000).write_csv(OUTD / "tournament_top20000.csv")
    lead.head(500).write_csv(OUTD / "tournament_leaders.csv")
    (OUTD / "tournament_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

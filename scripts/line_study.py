"""Line study: the model against every FanDuel main line (all seven markets), with and without the late-stamped rows, plus the walk-forward search, the blend test and the parlay-with-boost arithmetic.

Design: docs/overnight_plan.md, "Line study on all markets (2024-2026)". Descriptive and tracked; it sets no model weight and recommends no bet.

  python scripts/line_study.py build            # grade every priced line, write data/processed/line_study_sel.parquet
  python scripts/line_study.py report           # the with/without table and the reference numbers
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import calibrate as C
from live import store as ST

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("price_check", HERE / "price_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
OUT = config.PROCESSED_DIR / "line_study_sel.parquet"
LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
MARKETS = ["pass_att", "pass_cmp", "pass_yds", "qb_rush_att", "qb_rush_yds", "rush_att", "rush_yds", "rec", "rec_yds"]     # the two QB-rushing markets drop out of the study: the model has no distribution for them
STORES = {2025: config.ROOT / "records" / "holdout_2025" / "predictions", 2026: config.ROOT / "records" / "retro_2026" / "predictions"}
LIVE = config.ROOT / "records" / "predictions"          # live runs: used for 2026 weeks the retro replay does not cover (week 6 on), never for weeks it does
LIVE_FROM = (2026, 6)
BASE_2024 = config.PROCESSED_DIR / "vol_x_eff_2024_walkforward.parquet"      # regenerated base-model walk-forward predictions (2020-2024)
GRACE = 300
KS = (300, 600, 1200)                                                         # design weight: size of the neighbour pool
CS = (0.85, 1.0, 1.15)                                                        # design weight: width scaling on top of the calibrator's own
TIERS = ["pre", "grace", "late"]


def tier_of(late_s):
    return pl.when(late_s.is_null()).then(pl.lit("pre")).when(late_s <= GRACE).then(pl.lit("grace")).otherwise(pl.lit("late"))


def quoted_lines(season: int) -> pl.DataFrame:
    """One row per (game, player, market, line): last over/under prices, the latest stamp's lateness (s after kickoff), the open prices at the same line if the file gave them."""
    d = pl.read_csv(LOG, infer_schema_length=0).filter((pl.col("source") == "bettingpros_import") & pl.col("game_id").str.starts_with(f"{season}_"))
    d = d.with_columns(line=pl.col("line").cast(pl.Float64), price=pl.col("price").cast(pl.Float64, strict=False),
                       late_s=pl.col("note").str.extract(r"stamp (\d+) s after kickoff", 1).cast(pl.Float64), is_open=pl.col("note").str.starts_with("open"))
    k = ["game_id", "player_id", "market"]
    # the log drops a LAST row whose line and price equal the OPENING row's (dedup key has no time), so the latest row of either kind is the quote; an opening row is pregame ("pre")
    last = d.sort("captured_at")
    cur = last.group_by([*k, "side"], maintain_order=True).agg(line=pl.col("line").last())
    last = last.join(cur, on=[*k, "side", "line"], how="semi")
    kk = [*k, "line"]
    o = last.filter(pl.col("side") == "over").group_by(kk, maintain_order=True).last().select(*kk, over_price="price", late_o="late_s", captured_at="captured_at")
    u = last.filter(pl.col("side") == "under").group_by(kk, maintain_order=True).last().select(*kk, under_price="price", late_u="late_s")
    q = o.join(u, on=kk, how="inner").drop_nulls(["over_price", "under_price"])
    q = q.with_columns(late_s=pl.max_horizontal("late_o", "late_u")).drop("late_o", "late_u")
    op = d.filter(pl.col("is_open"))
    oo = op.filter(pl.col("side") == "over").group_by(kk, maintain_order=True).last().select(*kk, open_over="price")
    ou = op.filter(pl.col("side") == "under").group_by(kk, maintain_order=True).last().select(*kk, open_under="price")
    q = q.join(oo, on=kk, how="left").join(ou, on=kk, how="left")
    return q.with_columns(tier=tier_of(pl.col("late_s")), week=pl.col("game_id").str.split("_").list.get(1).cast(pl.Int64), season=pl.lit(season))


def pool_for(season: int, week: int, cal: C.Calibrator, market: str, wf: pl.DataFrame | None):
    """(sorted prediction, actual, c) the neighbour pool of a prediction at (season, week). 2025+: the live calibrator (all of 2020-2024). 2024: strictly earlier walk-forward rows only."""
    if season >= config.HOLDOUT_SEASON:
        v = cal.markets[market]
        return v["pred"], v["act"], (v["c"] if v["use_width"] else 1.0)
    d = wf.filter((pl.col("market") == market) & ((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))).drop_nulls(["prediction", "actual"])
    o = np.argsort(d["prediction"].to_numpy(), kind="stable")
    v = cal.markets[market]
    return d["prediction"].to_numpy()[o], d["actual"].to_numpy()[o], (v["c"] if v["use_width"] else 1.0)


def chances(pred, act, c0, means, lines, k, cmul):
    """P(actual > line) for each (mean, line): the k past cases nearest in prediction rank, spread about their median scaled by c0 * cmul."""
    if len(pred) < k:
        return np.full(len(means), np.nan)
    idx = C._window(pred, means, k)
    s = act[idx].astype(float)
    med = np.median(s, axis=1, keepdims=True)
    s = med + c0 * cmul * (s - med)
    return (s > lines[:, None]).mean(axis=1)


def predictions(season: int) -> pl.DataFrame:
    if season in STORES:
        from app import data as D
        p = D.latest(ST.PredictionStore(STORES[season]).load()).filter(pl.col("player_id").is_not_null()).select("game_id", "player_id", "market", "mean")
        p = p.with_columns(week=pl.col("game_id").str.split("_").list.get(1).cast(pl.Int64))
        if season == LIVE_FROM[0] and LIVE.exists():
            lv = D.latest(ST.PredictionStore(LIVE).load()).filter(pl.col("player_id").is_not_null()).select("game_id", "player_id", "market", "mean")
            lv = lv.with_columns(week=pl.col("game_id").str.split("_").list.get(1).cast(pl.Int64)).filter(pl.col("game_id").str.starts_with(f"{season}_") & (pl.col("week") >= LIVE_FROM[1]))
            p = pl.concat([p.filter(pl.col("week") < LIVE_FROM[1]), lv])
        return p
    wf = pl.read_parquet(BASE_2024).filter((pl.col("season") == season) & (pl.col("kind") == "player")).select("week", player_id="entity", market="market", mean="prediction")
    return wf.drop_nulls("mean")


def build(season: int, cal: C.Calibrator, wf_all: pl.DataFrame | None) -> pl.DataFrame:
    from live import results as RS
    q = quoted_lines(season)
    p = predictions(season)
    x = q.join(p.drop("game_id") if "game_id" in p.columns and season not in STORES else p, on=(["game_id", "player_id", "market"] if season in STORES else ["week", "player_id", "market"]), how="inner")
    x = x.filter(pl.col("market").is_in(MARKETS)).sort("market", "week", "game_id", "player_id", "line")
    cols = {}
    for m in x["market"].unique().sort().to_list():
        sub = x.filter(pl.col("market") == m)
        res = np.full((len(KS) * len(CS), sub.height), np.nan)
        for wk in sub["week"].unique().to_list():
            msk = (sub["week"] == wk).to_numpy()
            pred, act, c0 = pool_for(season, wk, cal, m, wf_all)
            for i, (k, cm) in enumerate((k, cm) for k in KS for cm in CS):
                res[i, msk] = chances(pred, act, c0, sub["mean"].to_numpy()[msk], sub["line"].to_numpy()[msk], k, cm)
        cols[m] = res
    po = np.full((len(KS) * len(CS), x.height), np.nan)
    pos = 0
    for m in x["market"].unique().sort().to_list():
        n = (x["market"] == m).sum()
        po[:, pos:pos + n] = cols[m]
        pos += n
    for i, (k, cm) in enumerate((k, cm) for k in KS for cm in CS):
        x = x.with_columns(pl.Series(f"p_over_k{k}_c{int(cm * 100)}", po[i]))
    # no model chance without a calibrated pool (the QB-rushing pools are below the calibrator's minimum, so the model gives those markets no distribution): such lines are left out, never guessed
    x = x.with_columns(p_over=pl.col("p_over_k600_c100")).drop_nulls("p_over").filter(~pl.col("p_over").is_nan())
    # outcome
    act, _, _ = RS.played(season)
    act = act.filter(pl.col("player_id").is_not_null())
    # a played game: a line whose player has no result is void and a push is void; a game not played yet stays in, ungraded (actual empty)
    played = act.select("game_id").unique()
    x = x.join(act, on=["game_id", "player_id", "market"], how="left").with_columns(game_played=pl.col("game_id").is_in(played["game_id"].implode()))
    x = x.filter((~pl.col("game_played")) | (pl.col("actual").is_not_null() & (pl.col("actual") != pl.col("line")))).drop("game_played")
    return x


def long(x: pl.DataFrame) -> pl.DataFrame:
    """One row per line and side, at the LAST quoted price; the open price (if any) rides along as a second price source."""
    pcols = [c for c in x.columns if c.startswith("p_over")]
    base = ["season", "week", "game_id", "player_id", "market", "line", "tier", "late_s", "mean", "actual", *pcols]
    out = []
    for side in ("over", "under"):
        o = "under" if side == "over" else "over"
        d = x.select(*base, price=f"{side}_price", other=f"{o}_price", open_price=f"open_{side}", open_other=f"open_{o}")
        d = d.with_columns(side=pl.lit(side))
        out.append(d)
    s = pl.concat(out)
    be = lambda c: pl.when(pl.col(c) > 0).then(100.0 / (pl.col(c) + 100.0)).otherwise(-pl.col(c) / (-pl.col(c) + 100.0))
    s = s.with_columns(be=be("price"), be_other=be("other"))
    s = s.with_columns(fair=pl.col("be") / (pl.col("be") + pl.col("be_other")), vig=pl.col("be") + pl.col("be_other") - 1.0)
    dec = pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100.0).otherwise(1 + 100.0 / -pl.col("price"))
    won = pl.when(pl.col("side") == "over").then(pl.col("actual") > pl.col("line")).otherwise(pl.col("actual") < pl.col("line"))
    s = s.with_columns(dec=dec, won=won.cast(pl.Float64))
    s = s.with_columns(profit=pl.when(pl.col("won").is_null()).then(None).when(pl.col("won") == 1.0).then(pl.col("dec") - 1.0).otherwise(-1.0))
    for c in [c for c in s.columns if c.startswith("p_over")]:
        s = s.with_columns(pl.when(pl.col("side") == "over").then(pl.col(c)).otherwise(1 - pl.col(c)).alias(c.replace("p_over", "chance")))
    return s.with_columns(chance=pl.col("chance_k600_c100"), edge=pl.col("chance_k600_c100") - pl.col("be"))


def pick(s: pl.DataFrame, edge_col="edge") -> pl.DataFrame:
    """One side per line: the larger edge (ties to the over)."""
    return s.sort(["game_id", "player_id", "market", "line", edge_col, "side"], descending=[False, False, False, False, True, True]).unique(["game_id", "player_id", "market", "line"], keep="first", maintain_order=True)


def main_build():
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    wf = pl.read_parquet(BASE_2024).filter(pl.col("kind") == "player") if BASE_2024.exists() else None
    parts = []
    for season in (2022, 2023, 2024, 2025, 2026):
        if season <= 2024 and wf is None:
            print("2024: no regenerated base predictions; skipped")
            continue
        x = build(season, cal, wf)
        s = long(x)
        print(season, "lines graded", x.height, "tier", x.group_by("tier").len().sort("tier").rows())
        parts.append(s)
    pl.concat(parts, how="diagonal_relaxed").write_parquet(OUT)
    print("wrote", OUT)





# ---------------------------------------------------------------- reports

def boot(v, cl, draws=2000, seed=7):
    return pc.boot(np.asarray(v, float), np.asarray(cl))


def fair_roi(t: pl.DataFrame) -> float:
    """Return of a bettor with no skill facing these prices: the vig-free chance of the side at the quoted payout."""
    return float((t["fair"] * t["dec"] - 1.0).mean())


def line_row(name, t):
    if t.height < 10:
        return dict(name=name, n=t.height)
    roi, lo, hi = boot(t["profit"].to_numpy(), t["game_id"].to_numpy())
    prem = t["profit"].to_numpy() - (t["fair"] * t["dec"] - 1.0).to_numpy()
    pm, plo, phi = boot(prem, t["game_id"].to_numpy())
    return dict(name=name, n=t.height, hit=float(t["won"].mean()), be=float(t["be"].mean()), roi=roi, lo=lo, hi=hi, no_skill=fair_roi(t), premium=pm, p_lo=plo, p_hi=phi)


def tier_sets(s: pl.DataFrame) -> dict:
    return {"clean (pre + <=300 s)": s.filter(pl.col("tier") != "late"), "pregame only": s.filter(pl.col("tier") == "pre"),
            "late only (>300 s)": s.filter(pl.col("tier") == "late"), "all rows": s}


def main_report():
    s = pl.read_parquet(OUT)
    print("lines by season, tier:", s.filter(pl.col("side") == "over").group_by("season", "tier").len().sort("season", "tier").rows())
    print("lines by market:", s.filter(pl.col("side") == "over").group_by("market").len().sort("market").rows())
    rows = []
    for per, sub in (("2024", s.filter(pl.col("season") == 2024)), ("2025", s.filter(pl.col("season") == 2025)), ("2026", s.filter(pl.col("season") == 2026)), ("all", s)):
        if sub.height == 0:
            continue
        for nm, t in tier_sets(sub).items():
            rows.append(dict(period=per, **line_row(nm, pick(t))))
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3, tbl_width_chars=220):
        print("\nlarger-edge side on every line (no threshold):")
        print(pl.DataFrame(rows, infer_schema_length=None))
        rows = []
        for m in MARKETS:
            for nm, t in tier_sets(s.filter(pl.col("market") == m)).items():
                if nm in ("clean (pre + <=300 s)", "late only (>300 s)"):
                    rows.append(dict(market=m, **line_row(nm, pick(t))))
        print("\nby market:")
        print(pl.DataFrame(rows, infer_schema_length=None))
        rows = []
        rng = np.random.default_rng(1)
        for nm, sub in (("clean", s.filter(pl.col("tier") != "late")), ("all rows", s)):
            rows.append(dict(set=nm, **line_row("model: larger-edge side", pick(sub))))
            rows.append(dict(set=nm, **line_row("no model: better-priced side", pick(sub.with_columns(neg_be=-pl.col("be")), "neg_be"))))
            rows.append(dict(set=nm, **line_row("no model: random side", pick(sub.with_columns(r=pl.Series(rng.random(sub.height))), "r"))))
        print("\nwhat the model adds over a model-free choice of side (the premium over the vig-free price is partly the price asymmetry of the two sides):")
        print(pl.DataFrame(rows, infer_schema_length=None))


# ---------------------------------------------------------------- walk-forward search

BANDS = [("all", 0.0, 1.01), ("<50", 0.0, 0.5), ("50-60", 0.5, 0.6), ("60-70", 0.6, 0.7), (">=70", 0.7, 1.01)]
THETAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
GROUPS = {"all": MARKETS, "yards": ["pass_yds", "qb_rush_yds", "rush_yds", "rec_yds"], "counts": ["pass_att", "pass_cmp", "qb_rush_att", "rush_att", "rec"], "passing": ["pass_att", "pass_cmp", "pass_yds"],
          "rushing": ["qb_rush_att", "qb_rush_yds", "rush_att", "rush_yds"], "receiving": ["rec", "rec_yds"], **{m: [m] for m in MARKETS}}
MIN_HIST = 200


def rule_tables(s: pl.DataFrame):
    """For every design variant the larger-edge pick per line, and for every rule (variant, market group, side, band, theta) its bets-per-period and profit-per-period."""
    keys = sorted((int(a), int(b)) for a, b in s.select("season", "week").unique().rows())
    pidx = {k: i for i, k in enumerate(keys)}
    rules, N, S, picks = [], [], [], {}
    for k in KS:
        for cm in CS:
            col = f"chance_k{k}_c{int(cm * 100)}"
            t = s.drop_nulls(col).filter(~pl.col(col).is_nan()).with_columns(edge_v=pl.col(col) - pl.col("be"))
            t = pick(t, "edge_v").with_columns(ch=pl.col(col))
            per = np.array([pidx[(a, b)] for a, b in zip(t["season"].to_list(), t["week"].to_list())])
            mk, sd, ch, ed, pr = t["market"].to_numpy(), t["side"].to_numpy(), t["ch"].to_numpy(), t["edge_v"].to_numpy(), t["profit"].to_numpy()
            picks[(k, cm)] = t
            for g, ms in GROUPS.items():
                gm = np.isin(mk, ms)
                for d in ("both", "over", "under"):
                    dm = gm & ((sd == d) if d != "both" else True)
                    for bn, lo, hi in BANDS:
                        bm = dm & (ch >= lo) & (ch < hi)
                        for th in THETAS:
                            m = bm & (ed >= th)
                            rules.append((k, cm, g, d, bn, th, m))
                            N.append(np.bincount(per[m], minlength=len(keys)))
                            S.append(np.bincount(per[m], weights=pr[m], minlength=len(keys)))
    return keys, rules, np.array(N, float), np.array(S, float), picks


def walk_forward_search(s: pl.DataFrame, crit="roi", min_hist=MIN_HIST):
    keys, rules, N, S, picks = rule_tables(s)
    T = len(keys)
    chosen, rows = [], []
    for t in range(T):
        hn, hs = N[:, :t].sum(1), S[:, :t].sum(1)
        ok = hn >= min_hist
        if not ok.any() or N[:, t].sum() == 0:
            continue
        roi = np.where(ok, hs / np.maximum(hn, 1), -9)
        if crit == "lcb":
            roi = np.where(ok, roi - 1.0 / np.sqrt(np.maximum(hn, 1)), -9)
        r = int(np.argmax(roi))
        if N[r, t] == 0:
            continue
        elig = ok & (N[:, t] > 0)
        k, cm, g, d, bn, th, m = rules[r]
        tbl = picks[(k, cm)]
        sel = tbl.filter(pl.col("season") == keys[t][0]).filter(pl.col("week") == keys[t][1])
        sub_mask = m[((tbl["season"] == keys[t][0]) & (tbl["week"] == keys[t][1])).to_numpy()]
        bets = sel.filter(pl.Series(sub_mask))
        chosen.append(bets.with_columns(rule=pl.lit(f"K{k} c{cm} {g} {d} {bn} th{th}"), period=pl.lit(keys[t][0] * 100 + keys[t][1])))
        rows.append(dict(period=keys[t][0] * 100 + keys[t][1], rule=f"K{k} c{cm} {g} {d} {bn} th{th}", hist_n=int(hn[r]), hist_roi=float(hs[r] / hn[r]), n=int(N[r, t]), roi=float(S[r, t] / N[r, t]),
                         avg_eligible_rule_roi=float(np.mean(S[elig, t] / N[elig, t])), best_in_hindsight=float(np.max(np.where(N[:, t] >= 10, S[:, t] / np.maximum(N[:, t], 1), -9)))))
    return pl.DataFrame(rows), (pl.concat(chosen) if chosen else None), len(rules)


def main_search():
    s0 = pl.read_parquet(OUT)
    for label, s in (("clean rows (pre + <=300 s)", s0.filter(pl.col("tier") != "late")), ("all rows incl. late (>300 s)", s0)):
        for crit in ("roi", "lcb"):
            log, bets, nrules = walk_forward_search(s, crit)
            print(f"\n== walk-forward, {label}, choose by {crit}; {nrules} rules, >= {MIN_HIST} past bets to be eligible")
            if bets is None:
                print("  no bets")
                continue
            w = log["n"].to_numpy()
            roi, lo, hi = boot(bets["profit"].to_numpy(), bets["game_id"].to_numpy())
            base = pick(s)
            base = base.filter(pl.col("season") * 100 + pl.col("week") >= int(bets["period"].min()))
            b_roi, b_lo, b_hi = boot(base["profit"].to_numpy(), base["game_id"].to_numpy())
            avg = float(np.sum(log["avg_eligible_rule_roi"].to_numpy() * w) / w.sum())
            print(f"  chosen-each-week rule: {bets.height} bets, ROI {roi:+.3f} [{lo:+.3f}, {hi:+.3f}], hit {bets['won'].mean():.3f}; no-skill return at those prices {fair_roi(bets):+.3f}")
            print(f"  an average eligible rule, same weeks: {avg:+.3f}; larger-edge side on every line, same weeks ({base.height} bets): {b_roi:+.3f} [{b_lo:+.3f}, {b_hi:+.3f}]")
            for sea in sorted(bets["season"].unique().to_list()):
                b = bets.filter(pl.col("season") == sea)
                print(f"    {sea}: {b.height} bets, ROI {b['profit'].mean():+.3f}")
            print("  most-chosen rules:", log.group_by("rule").len().sort("len", descending=True).head(4).rows())




# ---------------------------------------------------------------- blend with the line, stacker, parlays

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def over_table(s: pl.DataFrame) -> pl.DataFrame:
    """One row per line, from the over side, with the book's vig-free chance of the over and the outcome."""
    o = s.filter(pl.col("side") == "over")
    return o.with_columns(fair_over=pl.col("fair"), over_won=pl.col("won"), p_model=pl.col("p_over_k600_c100"), period=pl.col("season") * 100 + pl.col("week"))


def blend_test(s: pl.DataFrame, min_train=150):
    """Walk-forward: does the model's mean carry information the line does not?  (1) stat scale, per market: out-of-sample absolute error of the line, the model mean and a fitted blend;
    (2) probability scale: log loss of the book's vig-free chance, the model's chance and a logistic blend, fitted only on earlier weeks."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    o = over_table(s).sort("period")
    periods = sorted(o["period"].unique().to_list())
    print("\n== blend test (walk-forward; each week fitted only on earlier weeks)")
    rows = []
    for m in MARKETS:
        d = o.filter(pl.col("market") == m)
        pe = {"line": [], "model": [], "blend": [], "w": []}
        for t in periods:
            tr, te = d.filter(pl.col("period") < t), d.filter(pl.col("period") == t)
            if tr.height < min_train or te.height == 0:
                continue
            X = np.column_stack([tr["line"].to_numpy(), tr["mean"].to_numpy()])
            lr = LinearRegression().fit(X, tr["actual"].to_numpy())
            Xt = np.column_stack([te["line"].to_numpy(), te["mean"].to_numpy()])
            a = te["actual"].to_numpy()
            pe["line"] += list(np.abs(a - te["line"].to_numpy())); pe["model"] += list(np.abs(a - te["mean"].to_numpy())); pe["blend"] += list(np.abs(a - lr.predict(Xt)))
            pe["w"].append(lr.coef_[1] / (lr.coef_.sum() if lr.coef_.sum() else 1))
        if pe["line"]:
            rows.append(dict(market=m, n=len(pe["line"]), mae_line=np.mean(pe["line"]), mae_model=np.mean(pe["model"]), mae_blend=np.mean(pe["blend"]), model_weight=float(np.mean(pe["w"]))))
    with pl.Config(tbl_rows=20, float_precision=3, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    ll = {"book": [], "model": [], "blend": []}
    coef = []
    for t in periods:
        tr, te = o.filter(pl.col("period") < t), o.filter(pl.col("period") == t)
        if tr.height < 500 or te.height == 0:
            continue
        f = lambda d: np.column_stack([logit(d["fair_over"].to_numpy()), logit(d["p_model"].to_numpy())])
        lr = LogisticRegression(C=1.0, max_iter=500).fit(f(tr), tr["over_won"].to_numpy())
        y = te["over_won"].to_numpy()
        for nm, p in (("book", te["fair_over"].to_numpy()), ("model", te["p_model"].to_numpy()), ("blend", lr.predict_proba(f(te))[:, 1])):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            ll[nm] += list(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        coef.append(lr.coef_[0])
    c = np.mean(coef, axis=0)
    print(f"probability scale (all markets pooled, {len(ll['book'])} lines): log loss book {np.mean(ll['book']):.4f}, model {np.mean(ll['model']):.4f}, blend {np.mean(ll['blend']):.4f}; mean blend weights on logit(book) {c[0]:.2f}, logit(model) {c[1]:.2f}")
    d = np.array(ll["book"]) - np.array(ll["blend"])
    print(f"  blend beats the book by {d.mean():.5f} log loss per line (se {d.std() / np.sqrt(len(d)):.5f})")


def stacker(s: pl.DataFrame, theta_list=(0.0, 0.02, 0.04, 0.06), min_train=1000):
    """A model trained to beat the line with the line as an input: walk-forward logistic regression of the over on [logit book, logit model, (mean - line) scaled, market dummies]; bet the larger edge of the stacker's chance over the quoted break-even."""
    from sklearn.linear_model import LogisticRegression
    o = over_table(s).sort("period")
    sd = o.group_by("market").agg(sd=pl.col("actual").std()).to_dict(as_series=False)
    sdm = dict(zip(sd["market"], sd["sd"]))
    feats = lambda d: np.column_stack([logit(d["fair_over"].to_numpy()), logit(d["p_model"].to_numpy()), (d["mean"].to_numpy() - d["line"].to_numpy()) / np.array([sdm[m] for m in d["market"].to_list()]),
                                       *[(d["market"].to_numpy() == m).astype(float) for m in MARKETS]])
    out = []
    for t in sorted(o["period"].unique().to_list()):
        tr, te = o.filter(pl.col("period") < t), o.filter(pl.col("period") == t)
        if tr.height < min_train or te.height == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(tr), tr["over_won"].to_numpy())
        out.append(te.with_columns(p_stack=pl.Series(lr.predict_proba(feats(te))[:, 1])))
    if not out:
        print("stacker: not enough history")
        return None
    z = pl.concat(out)
    be_o = z["be"].to_numpy()
    # be of the under = be_other of the over row
    be_u = z["be_other"].to_numpy()
    e_o, e_u = z["p_stack"].to_numpy() - be_o, (1 - z["p_stack"].to_numpy()) - be_u
    side = np.where(e_o >= e_u, "over", "under")
    edge = np.maximum(e_o, e_u)
    dec_o = z["dec"].to_numpy()
    dec_u = np.where(be_u >= 0.5, 1 / be_u, 1 / be_u)                      # decimal price of the under = 1 / its break-even
    won = np.where(side == "over", z["over_won"].to_numpy(), 1 - z["over_won"].to_numpy())
    dec = np.where(side == "over", dec_o, dec_u)
    prof = np.where(won == 1, dec - 1, -1.0)
    fair_side = np.where(side == "over", z["fair_over"].to_numpy(), 1 - z["fair_over"].to_numpy())
    z = z.with_columns(side_s=pl.Series(side), edge_s=pl.Series(edge), prof_s=pl.Series(prof), fair_s=pl.Series(fair_side), dec_s=pl.Series(dec))
    print(f"\n== stacker (line as input), walk-forward: {z.height} lines scored")
    for th in theta_list:
        t = z.filter(pl.col("edge_s") >= th)
        if t.height < 30:
            continue
        roi, lo, hi = boot(t["prof_s"].to_numpy(), t["game_id"].to_numpy())
        ns = float((t["fair_s"] * t["dec_s"] - 1).mean())
        print(f"  theta {th:.2f}: {t.height} bets, ROI {roi:+.3f} [{lo:+.3f}, {hi:+.3f}], no-skill {ns:+.3f}, premium {roi - ns:+.3f}")
    return z


def parlays(s: pl.DataFrame, boost=0.30, legs_list=(2, 3, 4)):
    """Parlay-with-boost arithmetic on all legs, from the data.  Every k-subset of the week's bets in DIFFERENT games is one parlay; its price is the product of the legs' decimal prices (what the book
    pays, margin included), a profit boost multiplies the profit of a winning parlay.  The mean over all such parlays is an unbiased estimate of the return of a parlay built from legs picked this way.
    Legs: the larger-edge side of every line with edge >= 0.04 (the model's picks), against the same number of random sides of random lines (no skill)."""
    rng = np.random.default_rng(7)
    picks = pick(s)
    print(f"\n== parlays with a {int(boost * 100)}% profit boost (legs in different games, one leg per player-market line)")
    rs = np.random.default_rng(3)
    cheap = pick(s.with_columns(neg_be=-pl.col("be")), "neg_be")
    rand = pick(s.with_columns(r=pl.Series(rs.random(s.height))), "r")
    for nm, legs in (("model picks (edge >= 0.04)", picks.filter(pl.col("edge") >= 0.04)), ("larger-edge side of every line", picks),
                     ("NO MODEL: better-priced side of every line", cheap), ("NO MODEL: random side of every line", rand)):
        for k in legs_list:
            tot, tot_b, tot_ns, tot_nsb, cnt = 0.0, 0.0, 0.0, 0.0, 0
            wk_res = []
            for (sea, wk), g in legs.group_by("season", "week"):
                dec, won, gid, fair = g["dec"].to_numpy(), g["won"].to_numpy(), g["game_id"].to_numpy(), g["fair"].to_numpy()
                n = len(dec)
                if n < k:
                    continue
                # sample parlays: random k-subsets with distinct games (exhaustive is huge)
                m = min(4000, n * 20)
                res = []
                for _ in range(m):
                    ix = rng.choice(n, size=k, replace=False)
                    if len(set(gid[ix])) < k:
                        continue
                    d = np.prod(dec[ix])
                    w = won[ix].prod()
                    res.append((w * (d - 1) - (1 - w), w * (boost * (d - 1) + (d - 1)) - (1 - w), np.prod(fair[ix]) * d - 1, np.prod(fair[ix]) * (1 + (1 + boost) * (d - 1)) - 1))
                if res:
                    r = np.array(res)
                    wk_res.append((len(r), r.sum(0)))
            if not wk_res:
                continue
            ntot = sum(a for a, _ in wk_res)
            r = sum(b for _, b in wk_res) / ntot
            # week-cluster bootstrap on the boosted return
            ws = np.array([b[1] / a for a, b in wk_res]); wn = np.array([a for a, _ in wk_res], float)
            bs = [np.average(ws[i], weights=wn[i]) for i in (rng.integers(0, len(ws), len(ws)) for _ in range(2000))]
            print(f"  {nm}, {k}-leg: {ntot} parlays sampled; no boost {r[0]:+.3f}; boosted {r[1]:+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]; no-skill with boost {r[3]:+.3f}, no-skill without {r[2]:+.3f}")


def main_extra():
    s0 = pl.read_parquet(OUT)
    for label, s in (("clean rows", s0.filter(pl.col("tier") != "late")), ("all rows incl. late", s0)):
        print(f"\n######## {label}")
        blend_test(s)
        stacker(s)
        parlays(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "report", "search", "extra"])
    a = ap.parse_args()
    {"build": main_build, "report": main_report, "search": main_search, "extra": main_extra}[a.cmd]()

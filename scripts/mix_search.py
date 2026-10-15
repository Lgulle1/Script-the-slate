"""The search for a profitable mix on the 2025 lines (docs/overnight_plan.md, "Search for a profitable mix on the 2025 lines"). Design fixed before any 2025 line was opened.

FIND = 2025 weeks 1-9, VERIFY = 2025 weeks 10-18, CONFIRM = 2026 weeks 1-5. Rules are scored on FIND only; the chosen rule is opened once on VERIFY, then on CONFIRM.
  python scripts/mix_search.py
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
from live import store as ST

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("price_check", HERE / "price_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
spec2 = importlib.util.spec_from_file_location("gap_check", HERE / "gap_check.py")
gc = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(gc)
OUTD = config.PROCESSED_DIR
MARKETS = ["pass_yds", "rush_yds", "rec", "rec_yds"]
THETAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
BANDS = [("all", 0.0, 1.01), ("below 50%", 0.0, 0.5), ("50-60%", 0.5, 0.6), ("60-70%", 0.6, 0.7), ("70% and up", 0.7, 1.01)]
MIN_BETS = 300
SIMS = 200
SEED = 7
STORES = {2025: config.ROOT / "records" / "holdout_2025" / "predictions", 2026: config.ROOT / "records" / "retro_2026" / "predictions"}


def build(season: int, cal: C.Calibrator) -> pl.DataFrame:
    q = pc.quoted(pl.read_csv(pc.LOG, infer_schema_length=0)).filter(pl.col("market").is_in(MARKETS) & pl.col("game_id").str.starts_with(f"{season}_"))
    preds = ST.PredictionStore(STORES[season]).load()
    s = gc.selections(q, preds, cal).sort("book", *pc.KEY, "line", "side")
    # line-centred chance of the over (the same past cases centred on the book's line) and the line's margin
    lc, vig = {}, {}
    for r in s.filter(pl.col("side") == "over").iter_rows(named=True):
        smp = cal.sample(r["market"], r["line"])
        lc[(r["game_id"], r["player_id"], r["market"], r["line"])] = float((smp > r["line"]).mean()) if smp is not None else 0.5
    s = s.with_columns(p_lc_over=pl.struct("game_id", "player_id", "market", "line").map_elements(lambda r: lc.get((r["game_id"], r["player_id"], r["market"], r["line"]), 0.5), return_dtype=pl.Float64))
    s = s.with_columns(p_lc=pl.when(pl.col("side") == "over").then(pl.col("p_lc_over")).otherwise(1 - pl.col("p_lc_over")))
    g = pc.grade(s, season).sort("game_id", "player_id", "market", "line", "side")
    pair = g.group_by("game_id", "player_id", "market", "line", maintain_order=True).agg(vig=(pl.col("break_even").sum() - 1.0))
    return g.join(pair, on=["game_id", "player_id", "market", "line"], how="left").with_columns(season=pl.lit(season))


def pick(g: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """One selection per line: the side with the larger score (ties to the over)."""
    return g.sort(["game_id", "player_id", "market", "line", score_col, "side"], descending=[False, False, False, False, True, True]).unique(["game_id", "player_id", "market", "line"], keep="first", maintain_order=True)


def apply(g: pl.DataFrame, market: str, direction: str, band: tuple, theta: float, edge_col: str = "edge", chance_col: str = "model_chance") -> pl.DataFrame:
    t = pick(g, edge_col)
    if market != "all":
        t = t.filter(pl.col("market") == market)
    if direction != "both":
        t = t.filter(pl.col("side") == direction)
    return t.filter((pl.col(chance_col) >= band[1]) & (pl.col(chance_col) < band[2]) & (pl.col(edge_col) >= theta))


def roi(t: pl.DataFrame) -> tuple:
    if t.height == 0:
        return 0, None
    return t.height, float(np.mean(t["profit"].to_numpy()))


def rule_grid() -> list:
    return [dict(market=m, direction=d, band=b, theta=th) for m in ["all", *MARKETS] for d in ("both", "over", "under") for b in BANDS for th in THETAS]


def mix_features(g: pl.DataFrame):
    mk = np.stack([(g["market"].to_numpy() == m).astype(float) for m in MARKETS], 1)
    return np.column_stack([g["model_chance"].to_numpy(), g["p_lc"].to_numpy(), g["edge"].to_numpy(), np.log1p(g["line"].to_numpy()), (g["side"].to_numpy() == "over").astype(float), g["vig"].to_numpy(), mk])


def main():
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    g25, g26 = build(2025, cal), build(2026, cal)
    for g in (g25, g26):
        pass
    find, verify, confirm = g25.filter(pl.col("week") <= 9), g25.filter(pl.col("week") >= 10), g26.filter(pl.col("week") <= 5)
    print(f"selections: FIND {find.height}, VERIFY {verify.height}, CONFIRM {confirm.height} (both sides of {find.height // 2 + verify.height // 2 + confirm.height // 2} lines)")
    # the fitted mix: L2 logistic regression (C = 1) on FIND only
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(mix_features(find))
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(mix_features(find)), find["won"].to_numpy())
    def with_mix(g):
        p = lr.predict_proba(sc.transform(mix_features(g)))[:, 1]
        return g.with_columns(mix_chance=pl.Series(p), mix_edge=pl.Series(p) - g["break_even"])
    find, verify, confirm = with_mix(find), with_mix(verify), with_mix(confirm)
    rows = []
    for r in rule_grid():
        n, v = roi(apply(find, r["market"], r["direction"], r["band"], r["theta"]))
        rows.append(dict(kind="edge", market=r["market"], direction=r["direction"], band=r["band"][0], theta=r["theta"], n_find=n, roi_find=v))
    for th in THETAS:
        n, v = roi(apply(find, "all", "both", BANDS[0], th, edge_col="mix_edge", chance_col="mix_chance"))
        rows.append(dict(kind="mix", market="all", direction="both", band="all", theta=th, n_find=n, roi_find=v))
    R = pl.DataFrame(rows, infer_schema_length=None)
    elig = R.filter(pl.col("n_find") >= MIN_BETS)
    best = elig.sort("roi_find", descending=True).head(1).row(0, named=True)
    pos = elig.filter(pl.col("roi_find") > 0).height
    # what chance alone would give: outcomes redrawn at the book's no-vig chance, same rules, same FIND selections
    rng = np.random.default_rng(SEED)
    # fair chance of the over = be_over / (be_over + be_under); the under mirrors it
    be = find["break_even"].to_numpy()
    side = find["side"].to_numpy()
    key = (find["game_id"] + "|" + find["player_id"] + "|" + find["market"] + "|" + find["line"].cast(pl.Utf8)).to_numpy()
    uniq, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=be)
    fair_side = be / tot[inv]
    dec = np.where(find["price"].to_numpy() > 0, 1 + find["price"].to_numpy() / 100.0, 1 + 100.0 / -find["price"].to_numpy())
    null_pos, null_best = [], []
    keep_rules = [r for r in rule_grid()]
    elig_set = {(r["market"], r["direction"], r["band"], r["theta"]) for r in elig.filter(pl.col("kind") == "edge").iter_rows(named=True)}
    for _ in range(SIMS):
        u = rng.random(len(uniq))
        over_wins = u[inv] < (np.where(side == "over", fair_side, 1 - fair_side))                      # one uniform per line decides the over
        won = np.where(side == "over", over_wins, ~over_wins).astype(float)
        sim = find.with_columns(profit=pl.Series(np.where(won == 1.0, dec - 1.0, -1.0)))
        vals = []
        for r in keep_rules:
            if (r["market"], r["direction"], r["band"][0], r["theta"]) not in elig_set:
                continue
            n, v = roi(apply(sim, r["market"], r["direction"], r["band"], r["theta"]))
            if n >= MIN_BETS:
                vals.append(v)
        null_pos.append(sum(1 for v in vals if v > 0))
        null_best.append(max(vals) if vals else np.nan)
    out = dict(rules_scanned=int(elig.height), find_positive=int(pos), chance_positive_mean=float(np.mean(null_pos)), chance_best_mean=float(np.nanmean(null_best)), chance_best_p95=float(np.nanquantile(null_best, 0.95)))
    sel = lambda g: (apply(g, best["market"], best["direction"], next(b for b in BANDS if b[0] == best["band"]), best["theta"]) if best["kind"] == "edge"
                     else apply(g, "all", "both", BANDS[0], best["theta"], edge_col="mix_edge", chance_col="mix_chance"))
    res = {}
    for name, g in (("FIND", find), ("VERIFY", verify), ("CONFIRM", confirm)):
        t = sel(g)
        n, v = roi(t)
        lo, hi = (None, None)
        if n >= 10:
            _, lo, hi = pc.boot(t["profit"].to_numpy(), t["game_id"].to_numpy())
        res[name] = dict(n=n, roi=v, lo=lo, hi=hi, hit=float(np.mean(t["won"].to_numpy())) if n else None, be=float(np.mean(t["break_even"].to_numpy())) if n else None)
    # every simple whole-sample reference
    ref = {}
    for name, g in (("FIND", find), ("VERIFY", verify), ("CONFIRM", confirm)):
        ref[name] = dict(edge_ge_0=roi(apply(g, "all", "both", BANDS[0], 0.0)), mix_ge_0=roi(apply(g, "all", "both", BANDS[0], 0.0, edge_col="mix_edge", chance_col="mix_chance")))
    R.write_parquet(OUTD / "mix_search_rules.parquet")
    pl.DataFrame([dict(period=k, **v) for k, v in res.items()]).write_parquet(OUTD / "mix_search_chosen.parquet")
    pl.DataFrame([out]).write_parquet(OUTD / "mix_search_summary.parquet")
    with pl.Config(tbl_rows=30, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        print("scan:", out)
        print("chosen rule:", best)
        print(pl.DataFrame([dict(period=k, **v) for k, v in res.items()]))
        print("references (n, roi):", ref)
        print(R.filter(pl.col("n_find") >= MIN_BETS).sort("roi_find", descending=True).head(10))
    return res


if __name__ == "__main__":
    main()

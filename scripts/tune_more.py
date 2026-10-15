"""More ways to tune the betting setup, every one judged the same way: tuned on 2024 and scored on 2025-26, tuned on 2025-26 and scored on 2024 (the test data),
plus tuned-and-scored on everything (the flattering number). Writes every test bet to records/tuning/test_bets.csv so each one can be read.

  python scripts/tune_more.py
"""
import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tune_setup", HERE / "tune_setup.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
OUTD = config.ROOT / "records" / "tuning"
FILTER_MIN = 150


def data() -> pl.DataFrame:
    d = T.lines().filter(pl.col("tier") != "late")
    s = pl.read_parquet(T.SEL).filter(pl.col("side") == "over").select("game_id", "player_id", "market", "line", "mean", "actual")
    d = d.join(s, on=["game_id", "player_id", "market", "line"], how="left")
    log = pl.read_csv(config.ROOT / "records" / "lines" / "prop_lines_log.csv", infer_schema_length=0).select("game_id", "player_id", "player_name", "team").unique(["game_id", "player_id"])
    d = d.join(log, on=["game_id", "player_id"], how="left")
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT DISTINCT ON (game_id) game_id, gameday, home_team, spread_line, total_line, roof FROM schedules ORDER BY game_id, pulled_at DESC").pl()
    con.close()
    d = d.join(sc, on="game_id", how="left").with_columns(is_home=(pl.col("team") == pl.col("home_team")).cast(pl.Float64))
    # nflverse spread_line > 0 means the home team is favoured by that many points; team_spread > 0 = the player's team is favoured
    d = d.with_columns(team_spread=pl.when(pl.col("is_home") == 1).then(pl.col("spread_line")).otherwise(-pl.col("spread_line")),
                       gap=(pl.col("mean") - pl.col("line")) / pl.col("line"), dome=(pl.col("roof").is_in(["dome", "closed"])).cast(pl.Float64))
    return d.with_columns(pl.col("total_line").fill_null(44.0), pl.col("team_spread").fill_null(0.0))


def L_of(d):
    return T.as_dict(d), np.array([T.MARKETS.index(x) for x in d["market"].to_list()])


# ---- procedures: train frame -> (rule description, function frame -> (mask, side_is_over, profit, won, dec))

def proc_free(tr):
    L, M = L_of(tr)
    st = T.tune(L, M)
    return T.describe(st), lambda d: _apply_st(d, st)


def proc_model_only(tr):
    keep = T.WS
    T.WS = np.array([1.0])
    try:
        L, M = L_of(tr)
        st = T.tune(L, M)
    finally:
        T.WS = keep
    return "100% MODEL: " + T.describe(st), lambda d: _apply_st(d, st)


def _apply_st(d, st, extra=None):
    L, M = L_of(d)
    m, prof, won, dec = T.apply(L, M, st)
    over, _, _, _, _ = T.settle(L, st["w"], st["b"], st["kc"])
    if extra is not None:
        m = m & extra(d)
    return m, over, prof, won, dec


def proc_per_market(tr, min_bets=60):
    """Each market gets its own weight, shift, pool, cut-off and side."""
    keep = T.MIN_BETS
    T.MIN_BETS = min_bets
    sts = {}
    try:
        for mk in T.MARKETS:
            sub = tr.filter(pl.col("market") == mk)
            if sub.height < min_bets:
                continue
            L, M = L_of(sub)
            st = T.tune(L, M)
            if st and st["roi"] > 0 and st["markets"]:
                sts[mk] = st
    finally:
        T.MIN_BETS = keep

    def f(d):
        n = d.height
        m, over, prof, won, dec = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n), np.zeros(n), np.ones(n)
        for mk, st in sts.items():
            idx = np.flatnonzero((d["market"] == mk).to_numpy())
            if len(idx) == 0:
                continue
            mm, oo, pp, ww, dd = _apply_st(d[idx], st)
            m[idx], over[idx], prof[idx], won[idx], dec[idx] = mm, oo, pp, ww, dd
        return m, over, prof, won, dec
    desc = "PER MARKET: " + " | ".join(f"{mk}: model {st['w']:.0%}, shift {st['b']:+.0%}, edge>={st['theta']:.0%}, {st['side']}" for mk, st in sts.items())
    return desc, f


FILTERS = ([("gap <= " + str(-g), lambda d, g=g: (d["gap"] <= -g).to_numpy()) for g in (0.0, 0.05, 0.10, 0.15)]
           + [("gap >= " + str(g), lambda d, g=g: (d["gap"] >= g).to_numpy()) for g in (0.0, 0.05, 0.10)]
           + [("price no worse than -115", lambda d: _sidebe(d) <= 0.535), ("plus money / even", lambda d: _sidebe(d) <= 0.5)]
           + [("game total <= 42.5", lambda d: (d["total_line"] <= 42.5).to_numpy()), ("game total >= 47.5", lambda d: (d["total_line"] >= 47.5).to_numpy()),
              ("player's team favoured by 3+", lambda d: (d["team_spread"] >= 3).to_numpy()), ("player's team underdog by 3+", lambda d: (d["team_spread"] <= -3).to_numpy()),
              ("home", lambda d: (d["is_home"] == 1).to_numpy()), ("away", lambda d: (d["is_home"] == 0).to_numpy()), ("dome", lambda d: (d["dome"] == 1).to_numpy()),
              ("weeks 1-8", lambda d: (d["week"] <= 8).to_numpy()), ("weeks 9-18", lambda d: (d["week"] >= 9).to_numpy()),
              ("line below market median", lambda d: _below_median(d)), ("line above market median", lambda d: ~_below_median(d))])
_SIDE = {}


def _sidebe(d):
    return np.where(_SIDE["over"](d), d["be_o"].to_numpy(), d["be_u"].to_numpy())


def _below_median(d):
    med = {"pass_att": 33.5, "pass_cmp": 21.5, "pass_yds": 230.5, "rush_att": 13.5, "rush_yds": 55.5, "rec": 3.5, "rec_yds": 40.5}
    return d["line"].to_numpy() < np.array([med[m] for m in d["market"].to_list()])


def proc_filters(tr, max_filters=3):
    """The free setup, then up to three extra filters added one at a time, each the one that raises the training return most (at least 150 bets left)."""
    L, M = L_of(tr)
    st = T.tune(L, M)
    _SIDE["over"] = lambda d: T.settle(L_of(d)[0], st["w"], st["b"], st["kc"])[0]
    chosen = []
    cur = lambda d: np.ones(d.height, bool)
    m, over, prof, won, dec = _apply_st(tr, st)
    best_roi = prof[m].mean()
    for _ in range(max_filters):
        cand = None
        for nm, f in FILTERS:
            if nm in [c for c, _ in chosen]:
                continue
            k = m & cur(tr) & f(tr)
            if k.sum() >= FILTER_MIN and prof[k].mean() > best_roi + 0.005:
                if cand is None or prof[k].mean() > cand[2]:
                    cand = (nm, f, prof[k].mean())
        if cand is None:
            break
        chosen.append((cand[0], cand[1]))
        best_roi = cand[2]
        fs = [f for _, f in chosen]
        cur = lambda d, fs=fs: np.logical_and.reduce([f(d) for f in fs])
    fs = [f for _, f in chosen]
    extra = (lambda d: np.logical_and.reduce([f(d) for f in fs])) if fs else None
    return T.describe(st) + " + filters: " + (", ".join(c for c, _ in chosen) or "none"), lambda d: _apply_st(d, st, extra)


def _feats(d):
    lg = T.logit
    mk = [(d["market"] == m).to_numpy().astype(float) for m in T.MARKETS]
    return np.column_stack([lg(d["fair_o"].to_numpy()), lg(d["p_over_k600_c100"].to_numpy()), lg(d["p_over_k300_c100"].to_numpy()), lg(d["p_over_k1200_c100"].to_numpy()),
                            d["gap"].to_numpy(), d["line"].to_numpy(), d["be_o"].to_numpy(), d["be_u"].to_numpy(), d["total_line"].to_numpy(), d["team_spread"].to_numpy(),
                            d["is_home"].fill_null(0.5).to_numpy(), d["week"].to_numpy(), *mk])


def proc_ml(tr, theta=0.06):
    """A gradient-boosted classifier of the over from 19 inputs (book chance, model chances, gap, line, prices, game total, spread, home, week, market); bet the side whose edge >= theta."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, min_samples_leaf=80, random_state=7).fit(_feats(tr), tr["over_won"].to_numpy())

    def f(d):
        p = clf.predict_proba(_feats(d))[:, 1]
        eo, eu = p - d["be_o"].to_numpy(), (1 - p) - d["be_u"].to_numpy()
        over = eo >= eu
        won = np.where(over, d["over_won"].to_numpy(), 1 - d["over_won"].to_numpy())
        dec = np.where(over, d["dec_o"].to_numpy(), d["dec_u"].to_numpy())
        return np.maximum(eo, eu) >= theta, over, np.where(won == 1, dec - 1, -1.0), won, dec
    return f"MACHINE LEARNING picker, edge >= {theta:.0%}", f


def proc_agreed(tr):
    st = dict(w=0.5, b=-0.06, kc=(600, 100), theta=0.10, side="under", markets=["pass_att", "rush_att"])
    return "LAST TIME'S SETUP (fixed, not re-tuned): " + T.describe(st), lambda d: _apply_st(d, st)


PROCS = {"last time's setup": proc_agreed, "free tune (model + book)": proc_free, "100% model": proc_model_only, "per-market settings": proc_per_market,
         "free tune + extra filters": proc_filters, "machine learning 4%": lambda tr: proc_ml(tr, 0.04), "machine learning 8%": lambda tr: proc_ml(tr, 0.08)}


def boot(prof, g, rng=None):
    rng = rng or np.random.default_rng(7)
    u, inv = np.unique(g, return_inverse=True)
    s, n = np.bincount(inv, weights=prof), np.bincount(inv).astype(float)
    b = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    d = data()
    parts = {"2024": d.filter(pl.col("season") == 2024), "2025-26": d.filter(pl.col("season") >= 2025), "all": d}
    rows, bets = [], []
    for name, proc in PROCS.items():
        res = {}
        for tr, te in (("all", "all"), ("2024", "2025-26"), ("2025-26", "2024")):
            desc, f = proc(parts[tr])
            m, over, prof, won, dec = f(parts[te])
            res[(tr, te)] = (m, prof, won)
            if tr != "all":
                x = parts[te].filter(pl.Series(m)).with_columns(setup=pl.lit(name), tuned_on=pl.lit(tr), bet=pl.Series(np.where(over[m], "OVER", "UNDER")),
                                                                 price=pl.Series(np.where(over[m], d_price(parts[te], "o")[m], d_price(parts[te], "u")[m])),
                                                                 result=pl.Series(np.where(won[m] == 1, "WIN", "LOSS")), profit_units=pl.Series(np.round(prof[m], 3)), rule=pl.lit(desc))
                bets.append(x)
            if tr == "all":
                all_desc = desc
        mi, pi, wi = res[("all", "all")]
        ma, pa, wa = res[("2024", "2025-26")]
        mb, pb, wb = res[("2025-26", "2024")]
        oos_p = np.concatenate([pa[ma], pb[mb]])
        oos_w = np.concatenate([wa[ma], wb[mb]])
        oos_g = np.concatenate([parts["2025-26"]["game_id"].to_numpy()[ma], parts["2024"]["game_id"].to_numpy()[mb]])
        lo, hi = boot(oos_p, oos_g) if len(oos_p) > 20 else (None, None)
        rows.append(dict(setup=name, insample_bets=int(mi.sum()), insample_hit=float(wi[mi].mean()) if mi.sum() else None, insample_roi=float(pi[mi].mean()) if mi.sum() else None,
                         test_bets=len(oos_p), test_hit=float(oos_w.mean()) if len(oos_w) else None, test_roi=float(oos_p.mean()) if len(oos_p) else None, test_lo=lo, test_hi=hi,
                         test_units=float(oos_p.sum()), rule_on_all_data=all_desc))
        r = rows[-1]
        print(f"{name:28s} in-sample {r['insample_bets']:5d} bets hit {r['insample_hit'] or 0:.3f} roi {r['insample_roi'] or 0:+.3f} | TEST {r['test_bets']:5d} bets hit {r['test_hit'] or 0:.3f} "
              f"roi {r['test_roi'] or 0:+.3f} [{lo if lo is not None else 0:+.3f}, {hi if hi is not None else 0:+.3f}] units {r['test_units']:+.1f}\n      {all_desc}")
    OUTD.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUTD / "tuning_summary.csv")
    cols = ["setup", "tuned_on", "season", "week", "gameday", "player_name", "team", "market", "line", "bet", "price", "mean", "actual", "result", "profit_units", "rule"]
    pl.concat(bets, how="diagonal_relaxed").select(cols).rename({"mean": "model_prediction"}).sort("setup", "season", "week", "player_name").write_csv(OUTD / "test_bets.csv")
    print("wrote", OUTD / "tuning_summary.csv", "and", OUTD / "test_bets.csv")


def d_price(d, side):
    dec = d["dec_o" if side == "o" else "dec_u"].to_numpy()
    return np.where(dec >= 2, np.round((dec - 1) * 100), np.round(-100 / (dec - 1)))


if __name__ == "__main__":
    main()

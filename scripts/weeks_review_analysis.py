"""What weeks 1-5 of the 2026 reconstruction teach (docs/overnight_plan.md, "What weeks 1-5 teach"). Descriptive; sets no weight, threshold or gate.

  python scripts/weeks_review_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import calibrate as C
from live import history as H

GRADED = config.ROOT / "data" / "processed" / "reconstruct_2026_graded.parquet"
LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
OUTD = config.ROOT / "data" / "processed"
MARKETS = ["pass_yds", "rush_yds", "rec", "rec_yds", "qb_rush_yds"]
LEVELS = (0.9, 0.8, 0.7, 0.6, 0.5)
DRAWS, SEED = 2000, 7
KEY = ["game_id", "player_id", "market"]


def imp(price):
    return np.where(price > 0, 100.0 / (price + 100.0), -price / (-price + 100.0))


def lines() -> pl.DataFrame:
    d = pl.read_csv(LOG, infer_schema_length=0).filter((pl.col("source") == "bettingpros_import") & pl.col("market").is_in(MARKETS))
    d = d.with_row_index("_i").with_columns(line=pl.col("line").cast(pl.Float64), price=pl.col("price").cast(pl.Float64, strict=False)).sort("captured_at", "_i")      # file order breaks ties
    o = d.filter(pl.col("side") == "over").group_by(KEY, maintain_order=True).last().select(*KEY, line="line", over_price="price")
    u = d.filter(pl.col("side") == "under").group_by(KEY, maintain_order=True).last().select(*KEY, under_line="line", under_price="price")
    return o.join(u, on=KEY, how="left").with_columns(under_price=pl.when(pl.col("under_line") == pl.col("line")).then(pl.col("under_price"))).drop("under_line")


def sample_for(cal, market: str, p: float):
    v = cal.markets.get(market)
    if v is None or len(v["pred"]) < cal.cfg.min_pool[market]:
        return None
    s = v["act"][C._window(v["pred"], np.array([p]), cal.cfg.k[market])[0]].astype(float)
    c = v["c"] if v["use_width"] else 1.0
    med = float(np.quantile(s, 0.5))
    return med + c * (s - med)


def boot_mean(per_game_sum: np.ndarray, per_game_n: np.ndarray) -> tuple:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_game_sum), size=(DRAWS, len(per_game_sum)))
    b = per_game_sum[idx].sum(1) / per_game_n[idx].sum(1)
    return float(per_game_sum.sum() / per_game_n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def cluster_mean(df: pl.DataFrame, col: str) -> tuple:
    d = df.select("game_id", col).sort("game_id")                                # numpy reductions in a fixed order: byte-stable
    g = d["game_id"].to_numpy()
    v = d[col].to_numpy().astype(float)
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    return boot_mean(np.add.reduceat(v, starts), np.diff(np.r_[starts, len(v)]).astype(float))


def mean_of(df: pl.DataFrame, col: str) -> float:
    return float(np.mean(df[col].to_numpy().astype(float)))


def icc(z: np.ndarray, grp: np.ndarray) -> float:
    """One-way intraclass correlation of z within groups (unbalanced)."""
    u, inv = np.unique(grp, return_inverse=True)
    n = np.bincount(inv).astype(float)
    k = len(u)
    N = len(z)
    mean_g = np.bincount(inv, weights=z) / n
    grand = z.mean()
    msb = (n * (mean_g - grand) ** 2).sum() / (k - 1)
    msw = ((z - mean_g[inv]) ** 2).sum() / (N - k)
    k0 = (N - (n ** 2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def main():
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    g = pl.read_parquet(GRADED)
    team_of = {}
    played, _ = H.read_played_logs(2026)
    for r in played.select("game_id", "player_id", "team").iter_rows():
        team_of[(r[0], r[1])] = r[2]
    x = g.filter(pl.col("market").is_in(MARKETS) & ~pl.col("did_not_play")).select(*KEY, "week", model="mean", actual="actual").join(lines(), on=KEY, how="inner")
    x = x.with_columns(team=pl.struct("game_id", "player_id").map_elements(lambda r: team_of.get((r["game_id"], r["player_id"])), return_dtype=pl.String))
    x = x.sort(KEY)                                                              # joins do not fix row order; every statistic below is computed in this order
    scale = pl.DataFrame([dict(market=m[0], sc=float(np.mean(np.abs(t["actual"].to_numpy() - t["model"].to_numpy())))) for m, t in sorted(x.partition_by("market", as_dict=True).items())])
    x = x.join(scale, on="market").with_columns(zm=(pl.col("actual") - pl.col("model")) / pl.col("sc"), zl=(pl.col("actual") - pl.col("line")) / pl.col("sc"))
    x = x.with_columns(gain=(pl.col("line") - pl.col("actual")).abs() - (pl.col("model") - pl.col("actual")).abs(), mc=(pl.col("model") - pl.col("actual")).abs() < (pl.col("line") - pl.col("actual")).abs())
    print(f"rows {x.height}")
    # 1. consistency
    cons = pl.DataFrame([dict(market=m, week=w, n=t.height, gain=float(np.mean(t["gain"].to_numpy())), model_closer=float(np.mean(t["mc"].to_numpy().astype(float))))
                         for (m, w), t in sorted(x.partition_by("market", "week", as_dict=True).items())])                    # numpy means in a fixed row order: byte-stable
    cons.write_parquet(OUTD / "weeks_review_consistency.parquet")
    wk = cons.group_by("market").agg(weeks_model_better=(pl.col("gain") > 0).sum(), weeks=pl.len()).sort("market")
    # 2. probabilities
    rows = []
    for r in x.sort(KEY).iter_rows(named=True):
        s = sample_for(cal, r["market"], r["model"])
        if s is None or r["over_price"] is None or r["under_price"] is None or r["actual"] == r["line"]:
            continue
        po = imp(np.array([r["over_price"]]))[0]
        pu = imp(np.array([r["under_price"]]))[0]
        p_book = po / (po + pu)
        p_model = float((s > r["line"]).mean())
        y = float(r["actual"] > r["line"])
        fl = {f"f{int(l * 100)}": float(np.floor(np.quantile(s, 1 - l, method="lower"))) for l in LEVELS}
        rows.append(dict(game_id=r["game_id"], player_id=r["player_id"], market=r["market"], week=r["week"], line=r["line"], model=r["model"], actual=r["actual"], p_model=p_model, p_book=p_book, y=y, **fl))
    P = pl.DataFrame(rows).with_columns(
        bs_model=(pl.col("p_model") - pl.col("y")) ** 2, bs_book=(pl.col("p_book") - pl.col("y")) ** 2,
        ll_model=-(pl.col("y") * pl.col("p_model").clip(0.01, 0.99).log() + (1 - pl.col("y")) * (1 - pl.col("p_model")).clip(0.01, 0.99).log()),
        ll_book=-(pl.col("y") * pl.col("p_book").clip(0.01, 0.99).log() + (1 - pl.col("y")) * (1 - pl.col("p_book")).clip(0.01, 0.99).log()))
    P = P.with_columns(d_bs=pl.col("bs_book") - pl.col("bs_model"), d_ll=pl.col("ll_book") - pl.col("ll_model"))
    out = []
    for m in [*sorted(P["market"].unique().to_list()), "ALL"]:
        s = P if m == "ALL" else P.filter(pl.col("market") == m)
        bs = cluster_mean(s, "d_bs"); ll = cluster_mean(s, "d_ll")
        out.append(dict(market=m, n=s.height, brier_model=mean_of(s, "bs_model"), brier_book=mean_of(s, "bs_book"), d_brier=bs[0], d_brier_lo=bs[1], d_brier_hi=bs[2],
                        logloss_model=mean_of(s, "ll_model"), logloss_book=mean_of(s, "ll_book"), d_ll=ll[0], d_ll_lo=ll[1], d_ll_hi=ll[2],
                        mean_p_model=mean_of(s, "p_model"), mean_p_book=mean_of(s, "p_book"), over_rate=mean_of(s, "y")))
    prob = pl.DataFrame(out)
    prob.write_parquet(OUTD / "weeks_review_probability.parquet")
    # 3. bars
    bars = []
    for m in [*sorted(P["market"].unique().to_list()), "ALL"]:
        s = P if m == "ALL" else P.filter(pl.col("market") == m)
        for l in LEVELS:
            f = s.with_columns(hit=(pl.col("actual") >= pl.col(f"f{int(l * 100)}")).cast(pl.Float64))
            mu = cluster_mean(f, "hit")
            bars.append(dict(market=m, stated=l, n=f.height, hit=mu[0], lo=mu[1], hi=mu[2]))
    bars = pl.DataFrame(bars)
    bars.write_parquet(OUTD / "weeks_review_bars.parquet")
    P.sort(KEY).write_parquet(OUTD / "weeks_review_rows.parquet")
    # right side of the line: the model's lean at the line against what happened (ties and pushes removed)
    L = x.filter((pl.col("model") != pl.col("line")) & (pl.col("actual") != pl.col("line"))).with_columns(lean_over=pl.col("model") > pl.col("line"), went_over=pl.col("actual") > pl.col("line")).with_columns(right=(pl.col("lean_over") == pl.col("went_over")).cast(pl.Float64))
    side = []
    for m in [*sorted(L["market"].unique().to_list()), "ALL"]:
        s_ = L if m == "ALL" else L.filter(pl.col("market") == m)
        mu = cluster_mean(s_, "right")
        ov, un = s_.filter(pl.col("lean_over")), s_.filter(~pl.col("lean_over"))
        side.append(dict(market=m, scope="all", n=s_.height, right=mu[0], lo=mu[1], hi=mu[2], n_over=ov.height, right_over=mean_of(ov, "right") if ov.height else None, n_under=un.height, right_under=mean_of(un, "right") if un.height else None))
    for w in sorted(L["week"].unique().to_list()):
        s_ = L.filter(pl.col("week") == w)
        mu = cluster_mean(s_, "right")
        side.append(dict(market="ALL", scope=f"week {w}", n=s_.height, right=mu[0], lo=mu[1], hi=mu[2], n_over=None, right_over=None, n_under=None, right_under=None))
    side = pl.DataFrame(side)
    side.write_parquet(OUTD / "weeks_review_side.parquet")
    wrong = P.with_columns(lean_over=pl.col("model") > pl.col("line"), went_over=pl.col("y") > 0.5).filter(pl.col("lean_over") != pl.col("went_over"))
    save = wrong.with_columns(hit=(pl.col("actual") >= pl.col("f80")).cast(pl.Float64))
    wrong_share = mean_of(save, "hit")
    # 4. disagreement thirds
    gap = x.with_columns(gap=(pl.col("model") - pl.col("line")).abs() / pl.col("sc"))
    thirds = []
    for m in sorted(gap["market"].unique().to_list()):
        s = gap.filter(pl.col("market") == m).sort("gap")
        n = s.height
        if n < 9:
            continue
        for i, name in enumerate(("small gap", "middle", "large gap")):
            t = s[i * n // 3:(i + 1) * n // 3 if i < 2 else n]
            thirds.append(dict(market=m, third=name, n=t.height, model_closer=mean_of(t.with_columns(mcf=pl.col("mc").cast(pl.Float64)), "mcf"), gain=mean_of(t, "gain")))
    thirds = pl.DataFrame(thirds)
    thirds.write_parquet(OUTD / "weeks_review_gap.parquet")
    # 5. game environment
    games_g = pl.read_parquet(GRADED).filter(pl.col("market").is_in(["total"])).select("game_id", total_model="mean", total_actual="actual")
    import duckdb
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT game_id, total_line, home_team, away_team, home_score, away_score FROM schedules WHERE season = 2026 QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY pulled_at DESC) = 1").pl()
    con.close()
    qbp = pl.read_parquet(GRADED).filter((pl.col("market") == "pass_att") & ~pl.col("did_not_play")).select("game_id", "player_id", pa_model="mean", pa_actual="actual")
    qbp = qbp.with_columns(d_pa=pl.col("pa_actual") - pl.col("pa_model")).group_by("game_id").agg(d_pass_att=pl.col("d_pa").sum())
    per = pl.DataFrame([dict(game_id=gk[0], week=gk[1], n=t.height, nmae_model=float(np.mean(np.abs(t["zm"].to_numpy()))), nmae_line=float(np.mean(np.abs(t["zl"].to_numpy()))),
                             model_closer=float(np.mean(t["mc"].to_numpy().astype(float)))) for gk, t in sorted(x.partition_by("game_id", "week", as_dict=True).items())])
    per = per.join(games_g, on="game_id", how="left").join(
        sched.select("game_id", "total_line", "home_team", "away_team", "home_score", "away_score"), on="game_id", how="left").join(qbp, on="game_id", how="left").with_columns(
        surprise_model=pl.col("total_actual") - pl.col("total_model"), surprise_book=pl.col("total_actual") - pl.col("total_line")).sort("game_id")
    per.write_parquet(OUTD / "weeks_review_games.parquet")
    mm = x.drop_nulls("team")
    grp = (mm["game_id"] + "|" + mm["team"]).to_numpy()
    gids = mm["game_id"].to_numpy()
    ug = np.unique(gids)
    rng = np.random.default_rng(SEED)
    base = {k: icc(mm[k].to_numpy(), grp) for k in ("zm", "zl")}
    bs = {k: [] for k in base}
    by = {u: np.where(gids == u)[0] for u in ug}
    for _ in range(DRAWS):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([by[u] for u in pick])
        gg = np.concatenate([np.full(len(by[u]), f"{i}|") for i, u in enumerate(pick)]).astype(object) + grp[idx]
        for k in base:
            bs[k].append(icc(mm[k].to_numpy()[idx], gg))
    iccdf = pl.DataFrame([dict(series=k, icc=base[k], lo=float(np.quantile(bs[k], 0.025)), hi=float(np.quantile(bs[k], 0.975))) for k in base])
    iccdf.write_parquet(OUTD / "weeks_review_icc.parquet")
    pc = per.drop_nulls(["surprise_model", "nmae_model"])
    corr = float(np.corrcoef(pc["nmae_model"].to_numpy(), pc["surprise_model"].abs().to_numpy())[0, 1])
    corr_l = float(np.corrcoef(pc["nmae_line"].to_numpy(), pc["surprise_book"].abs().to_numpy())[0, 1])
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3, tbl_width_chars=220):
        print("1. weeks model better:"); print(wk)
        print(cons.pivot(on="week", index="market", values="gain"))
        print("2. probability scores:"); print(prob.select("market", "n", "brier_model", "brier_book", "d_brier", "d_brier_lo", "d_brier_hi", "logloss_model", "logloss_book", "d_ll", "d_ll_lo", "d_ll_hi", "mean_p_model", "mean_p_book", "over_rate"))
        print("3. bars:"); print(bars.pivot(on="stated", index="market", values="hit"))
        print(bars.filter(pl.col("stated") == 0.8)); print("wrong lean but 80% bar reached:", wrong_share, "of", save.height)
        print("right side:"); print(side)
        print("4. thirds:"); print(thirds)
        print("5. icc:"); print(iccdf); print("corr(model normalized miss, |total surprise|)", corr, " line:", corr_l, "games", pc.height)
    return dict(per=per)


if __name__ == "__main__":
    main()

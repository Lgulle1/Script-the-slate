"""4b.5: the game simulation through the walk-forward backtest, against the Phase 3 model.

  python run_sim_backtest.py [--time-slate SEASON WEEK]

Simulates every 2020-2024 game with 2,000 draws (sim.simulate.BACKTEST, flagged backtest_mode in every output; 2025 never loaded),
every input walk-forward (sim/inputs.py). For every market the simulated outcomes become a mean, a median and threshold
probabilities (P(X > rung) on config.LADDERS; moneyline = P(home wins)), and these are compared with the Phase 3 combined
volume x efficiency model on IDENTICAL rows:

  mean_*    MAE of the simulated mean vs the Phase 3 prediction   (moneyline: Brier of P(home win))
  median_*  MAE of the simulated median vs the Phase 3 prediction (information only)
  brier_*   mean Brier over the ladder rungs: simulated probabilities vs the Phase 3 model's walk-forward predictive
            probabilities (eval/grade.predictive_scores: its own earlier residuals around its prediction -- exactly how Phase 3 is graded)

Gain = Phase 3 loss - simulation loss (positive = the simulation is better); 95% interval from the 3.3 season-week cluster
bootstrap (config.BOOTSTRAP_RESAMPLES resamples, fixed seed); per-season gains and mean bias in the same row. No average across markets.
sim_weight_probabilities / sim_weight_mean are fixed in advance (not tuned): 1.0 if gain > 0 and the simulation wins at least 2
seasons separately on that metric, else 0.0 -- the simulation then stays at zero weight for that market and the market is flagged.

Also written: the 20% calibration table of the simulated and the Phase 3 probabilities (sim_backtest_calibration_*.parquet) and the simulated
vs actual margin / total / team-plays distributions (sim_backtest_distributions_*.parquet, docs/sim_backtest_distributions_*.png).
Running twice gives identical files.
"""
import argparse
import json
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

import config
from eval import baselines as bl
from eval import compare, grade
from eval import eligibility as elig
from models import game_state as gs
from sim import simulate as sm
from sim.inputs import build_sim_data

VERSION = config.ELIGIBLE_PLAYER_RULE["version"]
RESULTS_PATH = config.result_path("sim_backtest_results")
CALIBRATION_PATH = config.result_path("sim_backtest_calibration")
DISTRIBUTIONS_PATH = config.result_path("sim_backtest_distributions")
PLOT_PATH = config.ROOT / "docs" / f"sim_backtest_distributions_{VERSION}.png"
PHASE3_PREDICTIONS = config.PROCESSED_DIR / f"volume_efficiency_predictions_{VERSION}.parquet"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
SAMPLE_PER_GAME = 200
MIN_SEASONS_FOR_WEIGHT = 2
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


# ====================================================================== simulate everything
def simulate_all(sd, mode=sm.BACKTEST):
    """Summaries for every eligible (player, market) and game market, plus per-game distribution material."""
    by_game = {}
    for r in sd.eligible.iter_rows(named=True):
        by_game.setdefault(r["game_id"], {})[r["player_id"]] = set(elig.markets_of(r["elig"]))
    rows, dist, skipped = [], [], []
    meta = {r["game_id"]: r for r in sd.games.iter_rows(named=True)}
    plays = {(r["game_id"], r["team"]): r["plays"] for r in sd.tgs.select("game_id", "team", "plays").iter_rows(named=True)}
    for gid in sd.games.sort("season", "week", "game_id")["game_id"].to_list():
        g = sd.game_input(gid)
        if g is None:
            skipped.append(gid)
            continue
        gsim = sm.simulate_game(g, mode)
        m = meta[gid]
        for r in sm.summarize_game(gsim, by_game.get(gid, {})):
            rows.append(dict(season=m["season"], week=m["week"], game_id=gid, **r))
        hp, ap = plays[(gid, m["home_team"])], plays[(gid, m["away_team"])]
        draws = {"margin": (gsim.margin, m["margin"]), "total": (gsim.total, m["total"]), "home_points": (gsim.home.points, m["home_pf"]),
                 "away_points": (gsim.away.points, m["away_pf"]), "home_plays": (gsim.home.plays.astype(float), hp),
                 "away_plays": (gsim.away.plays.astype(float), ap)}
        for var, (d, actual) in draws.items():
            dist.append(dict(season=m["season"], week=m["week"], game_id=gid, variable=var, actual=float(actual),
                             pit=float((d < actual).mean() + 0.5 * (d == actual).mean()), sample=d[:SAMPLE_PER_GAME].astype(float).tolist(),
                             q25=float(np.quantile(d, 0.25)), q75=float(np.quantile(d, 0.75)), q05=float(np.quantile(d, 0.05)), q95=float(np.quantile(d, 0.95))))
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(dist, infer_schema_length=None), skipped


# ====================================================================== comparison with Phase 3
def _gain(d, cluster, seasons):
    lo, hi = compare.cluster_bootstrap(d, cluster)
    per = {int(s): float(d[seasons == s].mean()) for s in sorted(set(seasons.tolist()))}
    won = sum(1 for v in per.values() if v > 0)
    gain = float(d.mean())
    return dict(gain=gain, lo=lo, hi=hi, per=per, won=won, verdict=compare.verdict(gain, lo, hi, won))


def phase3_distribution(ph3: pl.DataFrame, market: str) -> pl.DataFrame:
    """Phase 3 predictions with their walk-forward predictive probabilities (the way 3.3 graded them)."""
    sub = ph3.filter(pl.col("market") == market)
    if market == "moneyline":
        return grade.moneyline_scores(sub)
    return grade.predictive_scores(sub, list(config.LADDERS[market]))


def compare_markets(sim: pl.DataFrame, ph3: pl.DataFrame):
    key = ["season", "week", "entity"]
    results, calib = [], []
    for market in MARKET_ORDER:
        s = sim.filter(pl.col("market") == market)
        p3 = phase3_distribution(ph3, market)
        rungs = ["win"] if market == "moneyline" else list(config.LADDERS[market])
        pc = [f"p_over_{r}" for r in rungs]
        a = p3.select(*key, "prediction", "actual", **{f"ph3_{c}": c for c in pc})
        b = s.select(*key, "sim_mean", "sim_median", **{f"sim_{c}": c for c in pc})
        t = a.join(b, on=key, how="inner").drop_nulls(["prediction", "sim_mean", *[f"ph3_{c}" for c in pc], *[f"sim_{c}" for c in pc]]).sort(key)
        row = dict(market=market, kind="game" if market in bl.GAME_MARKETS else "player", n=t.height, n_phase3=int(ph3.filter(
            (pl.col("market") == market) & pl.col("prediction").is_not_null()).height), n_sim=s.height, simulation_mode=sm.BACKTEST.name,
                   backtest_mode=True)
        if not t.height:
            results.append(row)
            continue
        y, p0 = t["actual"].to_numpy(), t["prediction"].to_numpy()
        seasons, cluster = t["season"].to_numpy(), t["season"].to_numpy() * 100 + t["week"].to_numpy()
        if market == "moneyline":
            ph3_p = np.clip(t["ph3_p_over_win"].to_numpy(), grade.EPS, 1 - grade.EPS)
            sim_p = np.clip(t["sim_p_over_win"].to_numpy(), grade.EPS, 1 - grade.EPS)
            b3, bs = (ph3_p - y) ** 2, (sim_p - y) ** 2
            probs = {"win": (ph3_p, sim_p, y)}
            metric = "brier"
            row.update(metric=metric)
        else:
            row.update(metric="mae")
            sm_mean, sm_med = t["sim_mean"].to_numpy(), t["sim_median"].to_numpy()
            e0, e1, e2 = np.abs(y - p0), np.abs(y - sm_mean), np.abs(y - sm_med)
            g = _gain(e0 - e1, cluster, seasons)
            gm = _gain(e0 - e2, cluster, seasons)
            row.update(mae_phase3=float(e0.mean()), mae_sim_mean=float(e1.mean()), mae_sim_median=float(e2.mean()),
                       mean_gain=g["gain"], mean_gain_pct=100 * g["gain"] / float(e0.mean()), mean_ci_lo=g["lo"], mean_ci_hi=g["hi"],
                       mean_seasons_won=g["won"], mean_verdict=g["verdict"], median_gain=gm["gain"], median_ci_lo=gm["lo"], median_ci_hi=gm["hi"],
                       bias_phase3=float((p0 - y).mean()), bias_sim_mean=float((sm_mean - y).mean()), bias_sim_median=float((sm_med - y).mean()),
                       sim_weight_mean=1.0 if (g["gain"] > 0 and g["won"] >= MIN_SEASONS_FOR_WEIGHT) else 0.0)
            row.update({f"mean_gain_{k}": v for k, v in g["per"].items()})
            ph3_cols = np.column_stack([t[f"ph3_p_over_{r}"].to_numpy() for r in rungs])
            sim_cols = np.column_stack([t[f"sim_p_over_{r}"].to_numpy() for r in rungs])
            outcome = np.column_stack([(y > r).astype(float) for r in rungs])
            b3, bs = ((ph3_cols - outcome) ** 2).mean(axis=1), ((sim_cols - outcome) ** 2).mean(axis=1)
            probs = {r: (ph3_cols[:, i], sim_cols[:, i], outcome[:, i]) for i, r in enumerate(rungs)}
        gb = _gain(b3 - bs, cluster, seasons)
        row.update(brier_phase3=float(b3.mean()), brier_sim=float(bs.mean()), brier_gain=gb["gain"], brier_gain_pct=100 * gb["gain"] / float(b3.mean()),
                   brier_ci_lo=gb["lo"], brier_ci_hi=gb["hi"], brier_seasons_won=gb["won"], brier_verdict=gb["verdict"],
                   sim_weight_probabilities=1.0 if (gb["gain"] > 0 and gb["won"] >= MIN_SEASONS_FOR_WEIGHT) else 0.0)
        row.update({f"brier_gain_{k}": v for k, v in gb["per"].items()})
        results.append(row)
        for src, idx in (("phase3", 0), ("simulation", 1)):
            allp = np.concatenate([v[idx] for v in probs.values()])
            allo = np.concatenate([v[2] for v in probs.values()])
            for c in grade.calibration_table(allp, allo).iter_rows(named=True):
                calib.append(dict(market=market, source=src, band=c["band"], n=c["n"], mean_predicted=c["mean_predicted"], actual_frequency=c["actual_frequency"]))
    return pl.DataFrame(results, infer_schema_length=None), pl.DataFrame(calib, infer_schema_length=None)


# ====================================================================== simulated vs actual distributions
def distribution_table(dist: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for var in ("margin", "total", "home_points", "away_points", "home_plays", "away_plays"):
        d = dist.filter(pl.col("variable") == var)
        actual = d["actual"].to_numpy()
        pooled = np.array([x for s in d["sample"].to_list() for x in s], dtype=float)
        pit = d["pit"].to_numpy()
        for src, v in (("actual", actual), ("simulated", pooled)):
            rows.append(dict(variable=var, source=src, n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                             **{f"q{int(q * 100):02d}": float(np.quantile(v, q)) for q in QUANTILES},
                             coverage_50=float(((actual >= d["q25"].to_numpy()) & (actual <= d["q75"].to_numpy())).mean()) if src == "actual" else None,
                             coverage_90=float(((actual >= d["q05"].to_numpy()) & (actual <= d["q95"].to_numpy())).mean()) if src == "actual" else None,
                             pit_ks=float(grade.pit_flatness(pit)["ks"]) if src == "actual" else None,
                             pit_max_dev=float(grade.pit_flatness(pit)["max_dev"]) if src == "actual" else None))
    return pl.DataFrame(rows)


def plot_distributions(dist: pl.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("margin", "Home margin (points)"), ("total", "Total points"), ("home_plays", "Home team plays"), ("away_plays", "Away team plays")]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for j, (var, label) in enumerate(panels):
        d = dist.filter(pl.col("variable") == var)
        actual = d["actual"].to_numpy()
        pooled = np.array([x for s in d["sample"].to_list() for x in s], dtype=float)
        lo, hi = np.quantile(np.concatenate([actual, pooled]), [0.002, 0.998])
        bins = np.linspace(lo, hi, 45)
        ax[0, j].hist(pooled, bins=bins, density=True, alpha=0.55, label="simulated (pooled draws)", color="#4C78A8")
        ax[0, j].hist(actual, bins=bins, density=True, alpha=0.55, label="actual games", color="#F58518")
        ax[0, j].set_title(label)
        ax[0, j].legend(fontsize=8)
        ax[1, j].hist(d["pit"].to_numpy(), bins=10, range=(0, 1), density=True, color="#54A24B", alpha=0.8)
        ax[1, j].axhline(1.0, color="k", lw=0.8, ls="--")
        ax[1, j].set_title(f"PIT of the actual in each game's simulated distribution ({var})")
    fig.suptitle(f"Simulation (2,000 draws/game, backtest mode) vs actual, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ====================================================================== run
def run(save=True, sim_data=None):
    sd = sim_data or build_sim_data()
    sim, dist, skipped = simulate_all(sd)
    ph3 = pl.read_parquet(PHASE3_PREDICTIONS).filter(pl.col("method") == "vol_x_eff")
    results, calib = compare_markets(sim, ph3)
    dtab = distribution_table(dist)
    if save:
        meta = {b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(), b"simulation_mode": sm.BACKTEST.name.encode(),
                b"run_id": sm.run_id(sm.BACKTEST).encode(), b"draws_per_game": str(sm.BACKTEST.n_sim).encode(),
                b"bootstrap": json.dumps(dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
                b"weight_rule": f"1.0 if gain > 0 and seasons_won >= {MIN_SEASONS_FOR_WEIGHT} else 0.0".encode()}
        pq.write_table(results.to_arrow().replace_schema_metadata(meta), RESULTS_PATH)
        pq.write_table(calib.to_arrow().replace_schema_metadata(meta), CALIBRATION_PATH)
        pq.write_table(dtab.to_arrow().replace_schema_metadata(meta), DISTRIBUTIONS_PATH)
        plot_distributions(dist, PLOT_PATH)
    return results, calib, dtab, skipped, sim


def time_slate(sd, season: int, week: int, store_dir=None):
    """Wall-clock of one week's slate at 20,000 draws per game (simulate, summarise and, if store_dir, store each game once)."""
    games = sd.games.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("game_id")["game_id"].to_list()
    by_game = {}
    for r in sd.eligible.filter(pl.col("game_id").is_in(games)).iter_rows(named=True):
        by_game.setdefault(r["game_id"], {})[r["player_id"]] = set(elig.markets_of(r["elig"]))
    t0 = time.time()
    n = 0
    for gid in games:
        g = sd.game_input(gid)
        if g is None:
            continue
        gsim = sm.simulate_game(g, sm.FULL)
        sm.summarize_game(gsim, by_game.get(gid, {}))
        if store_dir:
            sm.store_game(gsim, store_dir)
        n += 1
    return n, time.time() - t0


def print_summary(results, dtab, skipped):
    print(f"\nSimulation vs Phase 3, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward, {sm.BACKTEST.n_sim:,} draws/game "
          f"(BACKTEST MODE; holdout {config.HOLDOUT_SEASON} untouched). Gain = Phase 3 loss - simulation loss (positive = simulation better).")
    r = lambda c, k=4: pl.col(c).round(k)
    with pl.Config(tbl_rows=20, tbl_cols=24, tbl_width_chars=250, tbl_hide_dataframe_shape=True, fmt_str_lengths=14):
        print("\nMean prediction (MAE; moneyline Brier of P(home win) is under 'brier'):")
        print(results.select("market", "n", "mae_phase3", "mae_sim_mean", "mean_gain", pl.format("[{}, {}]", r("mean_ci_lo"), r("mean_ci_hi")).alias("ci"),
                             pl.format("{}/5", "mean_seasons_won").alias("won"), "mean_verdict", "sim_weight_mean").with_columns(
            [pl.col(c).round(4) for c in ("mae_phase3", "mae_sim_mean", "mean_gain")]))
        print("\nThreshold probabilities (mean Brier over the ladder rungs):")
        print(results.select("market", "n", "brier_phase3", "brier_sim", "brier_gain", pl.format("[{}, {}]", r("brier_ci_lo", 5), r("brier_ci_hi", 5)).alias("ci"),
                             pl.format("{}/5", "brier_seasons_won").alias("won"), "brier_verdict", "sim_weight_probabilities").with_columns(
            [pl.col(c).round(5) for c in ("brier_phase3", "brier_sim", "brier_gain")]))
        print("\nBias (prediction - actual):")
        print(results.select("market", "bias_phase3", "bias_sim_mean", "bias_sim_median").with_columns([pl.col(c).round(3) for c in ("bias_phase3", "bias_sim_mean", "bias_sim_median")]))
        print("\nGame outcomes: simulated (pooled draws) vs actual:")
        print(dtab.with_columns([pl.col(c).round(2) for c in dtab.columns if c not in ("variable", "source", "n")]))
    print(f"\ngames without simulation inputs: {len(skipped)}")
    print("No overall average is reported on purpose: each market stands on its own.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-slate", nargs=2, type=int, metavar=("SEASON", "WEEK"))
    a = ap.parse_args()
    t0 = time.time()
    sd = build_sim_data()
    print(f"inputs built in {time.time() - t0:.0f}s")
    if a.time_slate:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            n, secs = time_slate(sd, *a.time_slate, store_dir=d)
            print(f"{n} games x {sm.FULL.n_sim:,} draws (simulated, summarised, stored): {secs:.1f}s")
    t0 = time.time()
    results, calib, dtab, skipped, _ = run(sim_data=sd)
    print(f"simulation + comparison: {time.time() - t0:.0f}s")
    print_summary(results, dtab, skipped)
    print(f"\nsaved {RESULTS_PATH.name} ({results.height} rows), {CALIBRATION_PATH.name}, {DISTRIBUTIONS_PATH.name}, {PLOT_PATH.relative_to(config.ROOT)}")

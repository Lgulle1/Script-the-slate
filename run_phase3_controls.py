"""Phase 3 controls (3.4): how much of the Phase 3 gain is real?  Does not block Phase 4.

  python run_phase3_controls.py [--recompute]

1. Feature control. The same LightGBM (same params, same walk-forward harness, same combine step) but each quantity
   sees ONLY its own recency-weighted usage feature -- the player's recency-weighted mean of the stat being predicted
   (pass_att: rec_attempts, rush_att: rec_carries, targets: rec_targets, qb_rush_att: rec_rush_att_ex_kneel; for the
   efficiency ratios the recency-weighted pooled ratio itself: rec_comp_pct, rec_yds_per_cmp, rec_ypc, rec_catch_pct,
   rec_yds_per_rec, rec_qb_ypc). The two team models (plays, points per play) get the team analogue: team_off_plays /
   team_off_pts_per_play. Reported against the best baseline and against the full Phase 3 model.
2. Objective control. A recency-weighted MEDIAN baseline for every market (same recency weights as the `recency`
   baseline; weighted median = the smallest value whose cumulative weight reaches half the total). The Phase 3 models use
   an L1 objective (medians) while the baselines predict means, so part of the gain may come from matching the loss
   rather than from better features. Reported: the full model's gain against this baseline, and the median baseline's gain
   against the best mean baseline. Each market's mean bias (prediction - actual; none for moneyline) for all four
   predictors is reported too.

Everything is judged on IDENTICAL rows (all five baselines, the full model, the feature control and the median baseline
all have a prediction), with the 3.3 losses (MAE; Brier for moneyline), the same season-week cluster bootstrap
(config.BOOTSTRAP_RESAMPLES resamples, config.BOOTSTRAP_SEED) and the same CLEARS / EDGE / NO rule. Gain = reference loss -
subject loss, positive = the subject is better. The full-model and baseline predictions come from
data/processed/volume_efficiency_predictions_<pool version>.parquet (written by run_volume_efficiency_backtest.py, itself
byte-reproducible); --recompute reruns them instead. One row per market is written to
phase3_controls_results_<pool version>.parquet (versioned by pool like the other result files). 2025 is never loaded.
"""
import argparse
import json

import lightgbm
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from eval import compare
from features import volume_features as vf
from features.weights import games_elapsed, recency_weight
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint

RESULTS_PATH = config.result_path("phase3_controls_results")
PREDICTIONS_PATH = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
FULL, CONTROL, MEDIAN = "vol_x_eff", "feature_control", "recency_median"

# quantity -> the single feature the feature control keeps
USAGE_FEATURE = {"pass_att": "rec_attempts", "rush_att": "rec_carries", "targets": "rec_targets", "qb_rush_att": "rec_rush_att_ex_kneel",
                 "comp_pct": "rec_comp_pct", "yds_per_cmp": "rec_yds_per_cmp", "ypc": "rec_ypc", "catch_pct": "rec_catch_pct",
                 "yds_per_rec": "rec_yds_per_rec", "qb_ypc": "rec_qb_ypc"}
TEAM_FEATURE = {"volume": "team_off_plays", "efficiency": "team_off_pts_per_play"}


# ------------------------------------------------------------------ 1. feature control
def _keep(df: pl.DataFrame, feature: str) -> pl.DataFrame:
    keys = [c for c in volume._KEYS if c in df.columns]
    return df.select(*keys, feature)


def reduced_tables(tables: vf.FeatureTables, kind: str) -> vf.FeatureTables:
    """The feature tables cut down to each quantity's single recency-weighted usage feature."""
    return vf.FeatureTables({q: _keep(df, USAGE_FEATURE[q]) for q, df in tables.players.items()},
                            _keep(tables.teams, TEAM_FEATURE[kind]))


def run_feature_control(data) -> pl.DataFrame:
    vt, et = vf.load_feature_tables(data), efficiency.load_efficiency_tables(data)
    pair = combine.combined_predictors(volume.volume_predictors(reduced_tables(vt, "volume")),
                                       efficiency.efficiency_predictors(reduced_tables(et, "efficiency")))
    return bt.walk_forward(CONTROL, data, *pair)


# ------------------------------------------------------------------ 2. objective control
def weighted_median(values, weights):
    """Smallest value whose cumulative weight reaches half the total weight (None if there is no weight)."""
    pairs = sorted((v, w) for v, w in zip(values, weights) if v is not None and w > 0)
    tot = sum(w for _, w in pairs)
    if not tot:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2 - 1e-12:
            return v


def median_player(history, targets, cutoff):
    """Recency-weighted median of the player's whole prior history, per market (weights as in bl.player_recency)."""
    out = []
    for t in targets:
        h = bl._player_hist(history.player_log, t, cutoff)
        if not h.height:
            out.append({m: None for m in bl.PLAYER_MARKETS})
            continue
        w = [recency_weight(bl._games_back(s, g, team, wk, t))
             for s, g, team, wk in zip(h["season"].to_list(), h["team_game_num"].to_list(), h["team"].to_list(), h["week"].to_list())]
        out.append({m: weighted_median(h[c].to_list(), w) for m, c in bl.PLAYER_MARKETS.items()})
    return out


def _team_median(tlog, team, cutoff, season, game_num):
    h = bl._team_hist(tlog, team, cutoff)
    if not h.height:
        return None
    w = [recency_weight(games_elapsed(s, g, season, game_num)) for s, g in zip(h["season"].to_list(), h["team_game_num"].to_list())]
    return weighted_median(h["pf"].to_list(), w)


def median_game(history, targets, cutoff):
    """Home / away points = recency-weighted median of each team's own points scored; spread, total and moneyline follow
    exactly as in bl.predict_game."""
    import math
    out = []
    for t in targets:
        home = _team_median(history.team_log, t.home_team, cutoff, t.season, t.home_game_num)
        away = _team_median(history.team_log, t.away_team, cutoff, t.season, t.away_game_num)
        if home is None or away is None:
            out.append({"spread": None, "moneyline": None, "total": None})
            continue
        margin = home - away
        out.append({"spread": margin, "total": home + away,
                    "moneyline": 0.5 * (1 + math.erf(margin / config.GAME_MARGIN_SD / math.sqrt(2)))})
    return out


def run_median_baseline(data) -> pl.DataFrame:
    return bt.walk_forward(MEDIAN, data, median_player, median_game)


# ------------------------------------------------------------------ comparison
def _gain(market, ref, subj, y, cluster, seasons):
    """Paired gain = loss(ref) - loss(subj): mean, cluster-bootstrap CI, per-season gains and seasons won."""
    d = compare._loss(market, ref, y) - compare._loss(market, subj, y)
    lo, hi = compare.cluster_bootstrap(d, cluster)
    per = {int(s): float(d[seasons == s].mean()) for s in sorted(set(seasons.tolist()))}
    won = sum(1 for g in per.values() if g > 0)
    return dict(gain=float(d.mean()), lo=lo, hi=hi, per_season=per, seasons_won=won, verdict=compare.verdict(float(d.mean()), lo, hi, won))


def control_rows(base: pl.DataFrame, full: pl.DataFrame, control: pl.DataFrame, median: pl.DataFrame) -> pl.DataFrame:
    """One row per market (see the module docstring)."""
    rows = []
    for market in MARKET_ORDER:
        wide = base.filter(pl.col("market") == market).pivot(on="method", index=[*compare.KEY, "actual"], values="prediction")
        for name, df in ((FULL, full), (CONTROL, control), (MEDIAN, median)):
            wide = wide.join(df.filter(pl.col("market") == market)
                             .select(*compare.KEY, **{name: pl.col("prediction")}), on=compare.KEY, how="left")
        t = wide.drop_nulls([*compare.BASELINE_METHODS, FULL, CONTROL, MEDIAN]).sort(compare.KEY)
        y = t["actual"].to_numpy()
        seasons = t["season"].to_numpy()
        cluster = seasons * 100 + t["week"].to_numpy()
        base_losses = {b: compare._loss(market, t[b].to_numpy(), y).mean() for b in compare.BASELINE_METHODS}
        best = min(base_losses, key=base_losses.get)
        pred = {k: t[k].to_numpy() for k in (best, FULL, CONTROL, MEDIAN)}
        g = {"full_vs_best": _gain(market, pred[best], pred[FULL], y, cluster, seasons),
             "control_vs_best": _gain(market, pred[best], pred[CONTROL], y, cluster, seasons),
             "control_vs_full": _gain(market, pred[FULL], pred[CONTROL], y, cluster, seasons),   # negative = the full model is better
             "full_vs_median": _gain(market, pred[MEDIAN], pred[FULL], y, cluster, seasons),
             "median_vs_best": _gain(market, pred[best], pred[MEDIAN], y, cluster, seasons)}
        row = dict(market=market, n=t.height, metric="brier" if market == "moneyline" else "mae", best_baseline=best,
                   best_loss=float(base_losses[best]), full_loss=float(compare._loss(market, pred[FULL], y).mean()),
                   control_loss=float(compare._loss(market, pred[CONTROL], y).mean()),
                   median_loss=float(compare._loss(market, pred[MEDIAN], y).mean()))
        for name, r in g.items():
            row.update({f"{name}_gain": r["gain"], f"{name}_ci_lo": r["lo"], f"{name}_ci_hi": r["hi"],
                        f"{name}_seasons_won": r["seasons_won"], f"{name}_verdict": r["verdict"]})
            for s, v in r["per_season"].items():
                row[f"{name}_gain_{s}"] = v
        row["share_of_gain_kept_by_control"] = (g["control_vs_best"]["gain"] / g["full_vs_best"]["gain"]
                                                if g["full_vs_best"]["gain"] > 0 else None)
        for name, key in (("full", FULL), ("control", CONTROL), ("median", MEDIAN), ("best", best)):
            row[f"bias_{name}"] = None if market == "moneyline" else float((pred[key] - y).mean())
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def run(recompute=False, save=True):
    data = bt.load_backtest_data()
    if recompute or not PREDICTIONS_PATH.exists():
        import run_volume_efficiency_backtest as rv
        allp = pl.concat([rv.run_baselines(data), rv.run_models(data)])
    else:
        allp = pl.read_parquet(PREDICTIONS_PATH)
    base = allp.filter(pl.col("method").is_in(list(compare.BASELINE_METHODS)))
    full = allp.filter(pl.col("method") == FULL)
    control, median = run_feature_control(data), run_median_baseline(data)
    table = control_rows(base, full, control, median)
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        arrow = table.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(),
            b"lightgbm": lightgbm.__version__.encode(), b"bootstrap": json.dumps(
                dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"feature_control": json.dumps(dict(players=USAGE_FEATURE, teams=TEAM_FEATURE)).encode()})
        pq.write_table(arrow, RESULTS_PATH)
        pl.concat([control, median]).sort("method", "market", "season", "week", "entity").write_parquet(
            config.PROCESSED_DIR / f"phase3_controls_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet")
    return table, fp


def print_summary(t: pl.DataFrame, fp: str):
    print(f"\nPhase 3 controls, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward  fingerprint {fp}")
    print("Gain = reference loss - subject loss (positive = subject better); MAE, moneyline = Brier; CI = 95% season-week bootstrap.")
    print("Identical rows for every column in a market (so n can sit a little below the Phase 3 table's n).\n")
    r = lambda c, k=4: pl.col(c).round(k)
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, tbl_hide_dataframe_shape=True, fmt_str_lengths=14):
        print("A. Feature control (one usage feature) vs best baseline and vs the full model")
        print(t.select("market", "n", "best_baseline", full=r("full_vs_best_gain"), control=r("control_vs_best_gain"),
                       control_ci=pl.format("[{}, {}]", r("control_vs_best_ci_lo"), r("control_vs_best_ci_hi")),
                       control_verdict="control_vs_best_verdict", kept=pl.col("share_of_gain_kept_by_control").round(2),
                       control_vs_full=r("control_vs_full_gain"),
                       cvf_ci=pl.format("[{}, {}]", r("control_vs_full_ci_lo"), r("control_vs_full_ci_hi"))))
        print("\nB. Objective control: full model vs recency-weighted median baseline")
        print(t.select("market", median_vs_best=r("median_vs_best_gain"), full_vs_median=r("full_vs_median_gain"),
                       ci=pl.format("[{}, {}]", r("full_vs_median_ci_lo"), r("full_vs_median_ci_hi")),
                       seasons_won="full_vs_median_seasons_won", verdict="full_vs_median_verdict"))
        print("\nC. Mean bias (prediction - actual)")
        print(t.select("market", *[r(f"bias_{n}") for n in ("best", "median", "full", "control")]))
    print("\nNo overall average is reported on purpose: each market stands on its own, and no market is dropped here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rerun the baselines and the full model instead of reading the saved predictions")
    a = ap.parse_args()
    table, fp = run(a.recompute)
    print_summary(table, fp)
    print(f"\nsaved {RESULTS_PATH.name} ({table.height} rows)")

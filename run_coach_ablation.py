"""4d.1: does the coaching prior improve the Phase 3 models?  (coach_ablation_results_<pool version>.parquet)

  python run_coach_ablation.py [--tag TAG] [--results-dir DIR] [--pred-dir DIR]

The gate was fixed in docs/overnight_plan.md ("4d.1 gate") before this was built. Runs the combined volume x efficiency model through the walk-forward
harness over 2020-2024 (2025 never loaded) WITH the coaching columns of features/coach_prior.py and WITHOUT them (the committed Phase 3 run), same
hyperparameters and cutoffs. Per market, judged on identical rows: the gain (positive = the prior helps) with the 3.3 season-week bootstrap interval
on three row sets -- W1+W2 (weeks 1-6, or the first six games under a new head coach), W0 (everything else) and all. layer_weight is 1.0 only if
the market CLEARS on W1+W2 (gain > 0, interval excluding zero, >= MIN_SEASONS_WON seasons) AND is not significantly worse on W0 (the interval's
upper bound >= 0); otherwise 0.0, flagged, the market kept. A null control (the coaching columns shuffled among the teams of the same week) is
reported, not gated. The number of team-games with a prior and of change events in each row set is in the report.
"""
import argparse
import json
from pathlib import Path

import lightgbm
import numpy as np
import polars as pl
import pyarrow.parquet as pq

import config
from eval import backtest as bt
from eval import baselines as bl
from eval import compare
from features import coach_prior as cp
from features import volume_features as vf
from models import combine, efficiency, volume
from run_baseline_backtest import fingerprint
from run_comps_ablation import _paired, gain_stats

PREDICTIONS_PATH = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
MARKET_ORDER = list(bl.PLAYER_MARKETS) + list(bl.GAME_MARKETS)
WITHOUT, WITH = "vol_x_eff", "vol_x_eff_coach"
NULL_SEED = 20261012
ROWSETS = ("W1+W2", "W0", "all")


def results_name(tag: str | None = None) -> str:
    return f"coach_ablation_results_{config.ELIGIBLE_PLAYER_RULE['version']}" + (f"_{tag}" if tag else "")


def predictions_name(tag: str | None = None) -> str:
    return f"coach_ablation_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}" + (f"_{tag}" if tag else "") + ".parquet"


def null_features(feats: pl.DataFrame, seed: int = NULL_SEED) -> pl.DataFrame:
    """The coaching columns shuffled among the team-games of the same season and week (a fixed permutation): the same values, no information."""
    f = feats.sort("season", "week", "game_id", "team")
    grp = f.select(pl.struct("season", "week").rank("dense")).to_series().to_numpy()
    rng = np.random.default_rng(seed)
    perm = np.arange(f.height)
    for gid in np.unique(grp):
        idx = np.flatnonzero(grp == gid)
        perm[idx] = idx[rng.permutation(len(idx))]
    return f.select("game_id", "team", "season", "week").hstack(f.select(cp.COACH_COLUMNS)[perm])


def assign_windows(pred: pl.DataFrame, data, win: pl.DataFrame) -> pl.DataFrame:
    """Each prediction row's window: player rows by the player's team that week; game rows are W1+W2 when either team's game is."""
    w = win.select("game_id", "team", "season", "week", "window")
    pl_rows = pred.filter(pl.col("kind") != "game")
    game_rows = pred.filter(pl.col("kind") == "game")
    team_of = data.player_log.select("player_id", "season", "week", "team").unique(["player_id", "season", "week"])
    a = (pl_rows.join(team_of, left_on=["entity", "season", "week"], right_on=["player_id", "season", "week"], how="left")
         .join(w.select("team", "season", "week", "window"), on=["team", "season", "week"], how="left", maintain_order="left"))
    b = (game_rows.join(w.rename({"window": "w"}).group_by("game_id").agg(window=pl.when((pl.col("w") == "W1+W2").any()).then(pl.lit("W1+W2")).otherwise(pl.lit("W0"))),
                        left_on="entity", right_on="game_id", how="left", maintain_order="left"))
    return pl.concat([a.drop("team"), b], how="diagonal_relaxed")


def layer_decision(stats: dict) -> tuple:
    """(layer_weight, reason): 1.0 only if CLEARS on W1+W2 and not significantly worse on W0."""
    w12, w0 = stats["W1+W2"], stats["W0"]
    if w12["verdict"] != "CLEARS":
        return 0.0, f"not CLEARS on W1+W2 ({w12['verdict']})"
    if w0.get("ci_hi") is not None and w0["ci_hi"] < 0:
        return 0.0, "significantly worse on W0"
    return 1.0, "CLEARS on W1+W2, not worse on W0"


def run_with(data, feats: pl.DataFrame, name: str) -> pl.DataFrame:
    vt = vf.load_feature_tables(data, coach=feats)
    et = efficiency.load_efficiency_tables(data, coach=feats)
    pair = combine.combined_predictors(volume.volume_predictors(vt), efficiency.efficiency_predictors(et))
    return bt.walk_forward(name, data, *pair)


def ablation_rows(without: pl.DataFrame, with_: pl.DataFrame, data, win: pl.DataFrame, null: pl.DataFrame | None = None) -> pl.DataFrame:
    rows = []
    for market in MARKET_ORDER:
        paired = _paired(market, without, with_)
        wmap = assign_windows(without.filter(pl.col("market") == market).select("kind", "season", "week", "entity").unique(), data, win)
        t = paired.join(wmap.select("season", "week", "entity", "window"), on=["season", "week", "entity"], how="left")
        stats = {"all": gain_stats(market, t), "W1+W2": gain_stats(market, t.filter(pl.col("window") == "W1+W2")),
                 "W0": gain_stats(market, t.filter(pl.col("window") == "W0"))}
        row = dict(market=market, metric="brier" if market == "moneyline" else "mae")
        for rs, key in (("all", "all"), ("W1+W2", "w12"), ("W0", "w0")):
            s = stats[rs]
            row.update({f"n_{key}": s["n"], f"gain_{key}": s["gain"], f"ci_lo_{key}": s["ci_lo"], f"ci_hi_{key}": s["ci_hi"],
                        f"seasons_won_{key}": s["seasons_won"], f"verdict_{key}": s["verdict"]})
        row["layer_weight"], row["decision"] = layer_decision(stats)
        if null is not None:
            nt = _paired(market, without, null).join(wmap.select("season", "week", "entity", "window"), on=["season", "week", "entity"], how="left")
            g = gain_stats(market, nt.filter(pl.col("window") == "W1+W2"))
            row.update(gain_null_w12=g["gain"], ci_lo_null_w12=g["ci_lo"], ci_hi_null_w12=g["ci_hi"], verdict_null_w12=g["verdict"])
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col(pl.Null).cast(pl.Float64))


def run(tag=None, results_dir=None, pred_dir=None, save=True):
    data = bt.load_backtest_data()
    feats, stops = cp.build()
    feats = feats.filter(pl.col("season").is_in(config.BACKTEST_SEASONS))
    assert feats["season"].is_in(config.BACKTEST_SEASONS).all(), "coaching features outside the backtest seasons"
    win = cp.windows(feats)
    without = pl.read_parquet(PREDICTIONS_PATH).filter(pl.col("method") == WITHOUT)
    with_ = run_with(data, feats, WITH)
    null = run_with(data, null_features(feats), f"{WITH}_null")
    table = ablation_rows(without, with_, data, win, null)
    counts = dict(team_games=feats.height, with_prior=int((feats["coach_state"] == 1).sum()), new_without_prior=int((feats["coach_state"] == 2).sum()),
                  w12_games=int((win["window"] == "W1+W2").sum()), post_change_games=int(win["post_change"].sum()))
    fp = fingerprint(data)
    if save:
        config.ensure_data_dirs()
        arrow = table.to_arrow().replace_schema_metadata({
            b"data_fingerprint": fp.encode(), b"seasons": json.dumps(config.BACKTEST_SEASONS).encode(), b"lightgbm": lightgbm.__version__.encode(),
            b"bootstrap": json.dumps(dict(resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, unit="season-week cluster")).encode(),
            b"coach_columns": json.dumps(cp.COACH_COLUMNS).encode(), b"counts": json.dumps(counts).encode(), b"null_seed": str(NULL_SEED).encode(),
            b"layer_weight_rule": b"1.0 only if CLEARS on W1+W2 and the W0 interval's upper bound >= 0, else 0.0; " + config.LAYER_WEIGHT_RULE.encode()})
        pq.write_table(arrow, (results_dir or config.ROOT) / f"{results_name(tag)}.parquet")
        with_.sort("market", "season", "week", "entity").write_parquet((pred_dir or config.PROCESSED_DIR) / predictions_name(tag))
        stops.write_parquet((results_dir or config.ROOT) / "coach_stops.parquet")
    return table, fp, counts


def print_summary(t: pl.DataFrame, fp: str, counts: dict):
    print(f"\n4d.1 coaching prior ablation, {config.BACKTEST_SEASONS[0]}-{config.BACKTEST_SEASONS[-1]} walk-forward  fingerprint {fp}")
    print(f"team-games {counts['team_games']}, with a prior {counts['with_prior']}, new coach without a prior {counts['new_without_prior']}, "
          f"W1+W2 games {counts['w12_games']} (post-change {counts['post_change_games']})\n")
    r = lambda c, k=4: pl.col(c).round(k)
    with pl.Config(tbl_rows=20, tbl_cols=30, tbl_width_chars=250, tbl_hide_dataframe_shape=True, fmt_str_lengths=40):
        print(t.select("market", n_w12="n_w12", gain_w12=r("gain_w12"), ci_w12=pl.format("[{}, {}]", r("ci_lo_w12"), r("ci_hi_w12")), v_w12="verdict_w12",
                       gain_w0=r("gain_w0"), ci_w0=pl.format("[{}, {}]", r("ci_lo_w0"), r("ci_hi_w0")), gain_null=r("gain_null_w12"),
                       weight="layer_weight", why="decision"))
    off = t.filter(pl.col("layer_weight") == 0.0)["market"].to_list()
    print(f"\nlayer_weight 0 (flagged, markets kept): {off}")
    print("A flat result under option 3 is inconclusive about play calling (plan); the prior exists only for head coaches with an earlier stop.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--pred-dir", type=Path, default=None)
    a = ap.parse_args()
    table, fp, counts = run(a.tag, a.results_dir, a.pred_dir)
    print_summary(table, fp, counts)
    print(f"\nsaved {results_name(a.tag)}.parquet ({table.height} rows)")

"""Grade the 2026 weeks 1-5 reconstruction against the real results and the transparent 4-game trailing baseline (no lines involved). Descriptive only (docs/overnight_plan.md, "2026 weeks 1-5 reconstruction").

  python scripts/reconstruct_2026_eval.py
"""
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from live import store as ST

STORE = config.ROOT / "records" / "retro_2026" / "predictions"
OUT = config.ROOT / "data" / "processed" / "reconstruct_2026_graded.parquet"
spec = importlib.util.spec_from_file_location("retro5", config.ROOT / "scripts" / "retro_check_2026_week5.py")
retro5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retro5)
IS_GAME = pl.col("market").is_in(["spread", "total", "moneyline"])


def graded() -> pl.DataFrame:
    preds = ST.PredictionStore(STORE).load()
    preds = preds.sort("version").group_by(ST.KEY, maintain_order=True).last()
    parts = []
    for wk in sorted({int(g.split("_")[1]) for g in preds["game_id"].unique()}):
        p = preds.filter(pl.col("game_id").str.starts_with(f"2026_{wk:02d}_"))
        act = retro5.actuals(2026, wk)
        j = p.join(act, on=["game_id", "player_id", "market"], how="left", nulls_equal=True)
        j = j.filter(pl.col("game_id").is_in(act["game_id"].unique().to_list()))
        j = j.with_columns(did_not_play=pl.col("actual").is_null() & ~IS_GAME, actual=pl.when(pl.col("actual").is_null() & ~IS_GAME).then(0.0).otherwise(pl.col("actual")))
        j = retro5.baseline_trailing(j, 2026, p["kickoff_time"].min().date())
        parts.append(j.with_columns(week=pl.lit(wk)))
    return pl.concat(parts, how="diagonal_relaxed").with_columns(err=(pl.col("mean") - pl.col("actual")).abs(), err_trail4=(pl.col("trail4") - pl.col("actual")).abs())


def main():
    j = graded()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    j.write_parquet(OUT)
    played = j.filter(~pl.col("did_not_play") & pl.col("actual").is_not_null() & ~IS_GAME)
    cov = played.filter(pl.col("q10").is_not_null()).group_by("market").agg(n=pl.len(), in_10_90=((pl.col("actual") >= pl.col("q10")) & (pl.col("actual") <= pl.col("q90"))).mean())
    t = played.group_by("market").agg(n=pl.len(), mae=pl.col("err").mean(), mae_trail4=pl.col("err_trail4").mean(), bias=(pl.col("mean") - pl.col("actual")).mean()).join(cov.drop("n"), on="market", how="left").with_columns(
        vs_trail4_pct=(1 - pl.col("mae") / pl.col("mae_trail4")) * 100).sort("market")
    byweek = played.group_by("week").agg(n=pl.len(), skill_pct=(1 - pl.col("err").sum() / pl.col("err_trail4").sum()) * 100).sort("week")
    games = j.filter(IS_GAME & pl.col("actual").is_not_null())
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=3, tbl_hide_dataframe_shape=True):
        print(t); print(byweek)
        print(games.group_by("market").agg(n=pl.len(), mae=pl.col("err").mean()).sort("market"))
        ml = games.filter(pl.col("market") == "moneyline")
        print("game winner picked right:", float(((pl.Series(ml["mean"]) > 0.5) == (pl.Series(ml["actual"]) > 0.5)).mean()), "of", ml.height)
    return j


if __name__ == "__main__":
    main()

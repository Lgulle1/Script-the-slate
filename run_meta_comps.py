"""Comparables through the stage-2 stacker (docs/overnight_plan.md, "Comparables through the stage-2 stacker"): stage 2 with vs without the S1-S5 shift block, plus a permuted-block control.

Writes data/processed/meta_comps_results.parquet. Development seasons only; no line."""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from eval import compare
from eval import meta as M
import run_meta as RM

SHIFTS = config.PROCESSED_DIR / "comp_shifts_recency_weighted.parquet"
BASE = ["model", "baseline", "last3", "season_avg", "recency", "role_avg", "blend_70_30", "inj"]
MARKETS = ["targets", "rec", "rec_yds", "rush_att", "rush_yds"]
COMP = [f"{k}_S{i}" for i in range(1, 6) for k in ("shift_vol", "shift_eff", "n_eff", "nomatch", "best_sim")]


def with_comps(rows: pl.DataFrame, market: str) -> pl.DataFrame:
    c = pl.read_parquet(SHIFTS).filter((pl.col("market") == market) & pl.col("player_id").is_not_null())
    c = c.select("season", "week", "player_id", *[pl.col(x) if not x.startswith("nomatch") else pl.col(x).cast(pl.Float64) for x in COMP]).unique(["season", "week", "player_id"])
    return rows.join(c, left_on=["season", "week", "entity"], right_on=["season", "week", "player_id"], how="left")


def permute_block(d: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    parts = []
    for _, g in d.group_by("season", "week", maintain_order=True):
        idx = rng.permutation(g.height)
        parts.append(g.with_columns([pl.Series(c, g[c].to_numpy()[idx]) for c in COMP]))
    return pl.concat(parts)


def main() -> None:
    out = []
    for m in MARKETS:
        r = with_comps(RM.rows_for(m).with_columns(market=pl.lit(m)), m)
        base = M.stage2(r, m, features=BASE)
        withc = M.stage2(r, m, features=BASE + COMP)
        ctrl = M.stage2(permute_block(r), m, features=BASE + COMP)
        j = base.select("entity", "season", "week", "actual", "cluster", base_pred="pred").join(withc.select("entity", "season", "week", comp_pred="pred"), on=["entity", "season", "week"]).join(
            ctrl.select("entity", "season", "week", ctrl_pred="pred"), on=["entity", "season", "week"])
        y, seas, cl = j["actual"].to_numpy(), j["season"].to_numpy(), j["cluster"].to_numpy()
        res = dict(market=m, n=j.height, coverage=float(r.select(pl.col("shift_vol_S1").is_not_null().mean())[0, 0]))
        for tag, col in (("comps", "comp_pred"), ("control", "ctrl_pred")):
            g = RM.gain(np.abs(j["base_pred"].to_numpy() - y), np.abs(j[col].to_numpy() - y), cl, seas)
            res.update({f"{tag}_{k}": v for k, v in g.items()})
        res["decision"] = "CLEARS" if (res["comps_verdict"] == "CLEARS" and res["control_verdict"] != "CLEARS") else "parked"
        out.append(res)
    t = pl.DataFrame(out)
    t.write_parquet(config.PROCESSED_DIR / "meta_comps_results.parquet")
    pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(240)
    print(t.select("market", "n", pl.col("coverage").round(2), pl.col("comps_gain_pct").round(3), "comps_ci_lo", "comps_ci_hi", "comps_seasons_won", "comps_verdict",
                   pl.col("control_gain_pct").round(3), "control_verdict", "decision"))


if __name__ == "__main__":
    main()

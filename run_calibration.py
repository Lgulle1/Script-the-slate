"""Phase 6.3: build the calibrator from the committed 2020-2024 walk-forward predictions and report the fold check (plan: "Prediction schedule, calibration and grading").

Writes data/processed/calibration_eval.parquet (per market and fold), calibration_reliability.parquet, calibration/<market>_reliability.png and the live artifact
calibrator_v1.{json,parquet}. Development seasons only; no line is read; 2025 is never opened. A second run is byte-identical.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import config
from eval import calibrate as C

PRED = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
OUT = config.PROCESSED_DIR
BINS = np.linspace(0, 1, 11)


def load() -> pl.DataFrame:
    d = pl.read_parquet(PRED).filter((pl.col("method") == "vol_x_eff") & (pl.col("season") <= C.LAST) & pl.col("market").is_in(list(C.RUNGS)))
    return d.drop_nulls(["prediction", "actual"]).select("market", "season", "week", "entity", "prediction", "actual")


def reliability(p: np.ndarray, y: np.ndarray, stage: str, market: str) -> list:
    rows = []
    b = np.clip(np.digitize(p, BINS) - 1, 0, 9)
    for i in range(10):
        m = b == i
        if m.sum():
            rows.append(dict(market=market, stage=stage, bin=i, n=int(m.sum()), mean_pred=round(float(p[m].mean()), 6), observed=round(float(y[m].mean()), 6)))
    return rows


def main() -> None:
    rows = load()
    cfg = C.Config()
    ev_rows, rel_rows, notes = [], [], []
    (OUT / "calibration").mkdir(exist_ok=True)
    for m in cfg.rungs:
        wf = C.walk_forward(rows, m, cfg)
        if not wf.height:
            continue
        ev = C.evaluate(wf, cfg.rungs[m])
        if not ev["folds"]:
            notes.append(m)                                   # too few rows for the fold check: stays uncalibrated (c = 1, no isotonic), flagged
            continue
        for i, f in enumerate(ev["folds"]):
            ev_rows.append(dict(market=m, fold=f, n=wf.filter(pl.col("season") == f).height, c=ev["c"][i], cov_err_before=ev["cov_before"][i], cov_err_after=ev["cov_after"][i],
                                brier_before=ev["brier_before"][i] if cfg.rungs[m] else None, brier_after=ev["brier_after"][i] if cfg.rungs[m] else None,
                                use_width=ev["use_width"], use_isotonic=ev["use_isotonic"]))
        for f in ev["folds"]:
            te, tr = wf.filter(pl.col("season") == f), wf.filter(pl.col("season") < f)
            cv = {f"cov_{int(l * 100)}": C.coverage(te, 1.0)[l] for l in C.LEVELS}
            ev_rows[[r["market"] == m and r["fold"] == f for r in ev_rows].index(True)].update(cv)
        if cfg.rungs[m]:
            pb, pa, yy = [], [], []
            for f in ev["folds"]:
                tr, te = wf.filter(pl.col("season") < f), wf.filter(pl.col("season") == f)
                ptr, ytr = C._pairs(tr, cfg.rungs[m])
                pte, yte = C._pairs(te, cfg.rungs[m])
                iso = C.fit_isotonic(ptr, ytr)
                pb.append(pte); pa.append(iso(pte)); yy.append(yte)
            pb, pa, yy = np.concatenate(pb), np.concatenate(pa), np.concatenate(yy)
            rel_rows += reliability(pb, yy, "before", m) + reliability(pa, yy, "after", m)
            fig, ax = plt.subplots(figsize=(4, 4))
            for st, p in (("before", pb), ("after isotonic", pa)):
                r = [x for x in reliability(p, yy, st, m)]
                ax.plot([x["mean_pred"] for x in r], [x["observed"] for x in r], marker="o", label=st)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_title(f"{m}: ladder reliability, folds {ev['folds'][0]}-{ev['folds'][-1]}")
            ax.set_xlabel("predicted probability"); ax.set_ylabel("observed share"); ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / "calibration" / f"{m}_reliability.png", metadata={"Software": None}, dpi=100)
            plt.close(fig)
    pl.DataFrame(ev_rows, infer_schema_length=None).sort("market", "fold").write_parquet(OUT / "calibration_eval.parquet")
    pl.DataFrame(rel_rows).sort("market", "stage", "bin").write_parquet(OUT / "calibration_reliability.parquet")
    cal = C.build(rows, cfg)
    cal.save(OUT / "calibrator_v1")
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
    e = pl.read_parquet(OUT / "calibration_eval.parquet")
    print(e.group_by("market").agg(folds=pl.len(), cov_err_before=pl.col("cov_err_before").mean().round(3), cov_err_after=pl.col("cov_err_after").mean().round(3), c=pl.col("c").mean().round(2),
                                   brier_before=pl.col("brier_before").mean().round(4), brier_after=pl.col("brier_after").mean().round(4), use_width=pl.col("use_width").first(), use_isotonic=pl.col("use_isotonic").first()).sort("market"))
    print(e.select("market", "fold", "cov_50", "cov_80", "cov_90").sort("market", "fold"))
    print("no fold check (too few rows; width fix and isotonic not applied):", notes)
    print("calibrator hash", cal.hash)


if __name__ == "__main__":
    main()

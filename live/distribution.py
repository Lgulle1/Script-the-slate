"""Attach the calibrated predictive distribution (median, q05..q95, ladder) to a run's predictions (plan: "Prediction schedule, calibration and grading").

The mean is never touched. A market the calibrator has no distribution for (below the burn-in, no pool) keeps null quantiles and a null ladder, never a made-up one.
"""
from __future__ import annotations

import json

import polars as pl

from eval import calibrate as C

DIST_COLUMNS = ["median", *C.QCOLS, "ladder"]


def attach(preds: pl.DataFrame, cal: C.Calibrator | None) -> pl.DataFrame:
    """`preds` (market, mean, ...) with median, q05..q95 and a JSON ladder from `cal`; row order and means unchanged."""
    cols = {c: [] for c in DIST_COLUMNS}
    for m, mean in zip(preds["market"].to_list(), preds["mean"].to_list()):
        d = cal.distribution(m, mean) if (cal is not None and mean is not None and m in cal.markets) else None
        for c in DIST_COLUMNS:
            if d is None:
                cols[c].append(None)
            elif c == "ladder":
                cols[c].append(json.dumps(d["ladder"], sort_keys=True) if d["ladder"] else None)
            else:
                cols[c].append(d[c])
    return preds.with_columns([pl.Series(c, v, dtype=pl.String if c == "ladder" else pl.Float64) for c, v in cols.items()])

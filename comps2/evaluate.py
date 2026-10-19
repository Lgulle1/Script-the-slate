"""Signal ladder on the practice seasons (2021-2023): does each search's shift line up with how players actually did against their expectation, and does
similarity beat the same search without it (its control)?

For each search and market group: correlation of shift with rel (clipped), on rows where the search found something, and the MAE change from
expected * (1 + b * shift) with b fit on the EARLIER practice seasons only (2021 fit on 2020's rows is not possible -> b fit leave-one-season-out inside 2021-2023 is
NOT used; b is fit on season s-1 and applied to s, so 2021 is skipped for MAE).
"""
from __future__ import annotations

import numpy as np
import polars as pl

GROUPS = {"rush": ("rush_att", "rush_yds"), "rec": ("targets", "rec", "rec_yds"), "pass": ("pass_att", "pass_cmp", "pass_yds"), "qb_rush": ("qb_rush_att", "qb_rush_yds")}


def ladder(c: pl.DataFrame, comps, seasons=(2021, 2022, 2023)) -> pl.DataFrame:
    rows = []
    c = c.filter(pl.col("season").is_in(seasons))
    for g, ms in GROUPS.items():
        x = c.filter(pl.col("market").is_in(ms))
        for comp in comps:
            y = x.drop_nulls(f"{comp}_shift").filter(pl.col(f"{comp}_shift").is_not_nan())
            if y.height < 50:
                continue
            s, r = y[f"{comp}_shift"].to_numpy(), y["relc"].to_numpy()
            per = [float(np.corrcoef(z[f"{comp}_shift"].to_numpy(), z["relc"].to_numpy())[0, 1]) for (_,), z in y.group_by("season")]
            over = (y["actual"].to_numpy() > y["expected"].to_numpy()).astype(float)
            q1, q4 = np.quantile(s, [0.2, 0.8])
            rows.append(dict(group=g, comp=comp, n=y.height, corr=float(np.corrcoef(s, r)[0, 1]), corr_min_season=min(per), slope=float(np.polyfit(s, r, 1)[0]),
                             mae_gain=_mae_gain(y, comp), over_lift=float(over[s >= q4].mean() - over[s <= q1].mean())))
    return pl.DataFrame(rows)


def _mae_gain(y, comp):
    """b fit on season s-1 (least squares of rel on shift), applied to s; average MAE change in % of the base MAE."""
    out, base = [], []
    ss = sorted(y["season"].unique().to_list())
    for s in ss[1:]:
        tr, te = y.filter(pl.col("season") == s - 1), y.filter(pl.col("season") == s)
        a = tr[f"{comp}_shift"].to_numpy()
        b = float((a * tr["relc"].to_numpy()).sum() / max((a * a).sum(), 1e-12))
        ex, ac, sh = te["expected"].to_numpy(), te["actual"].to_numpy(), te[f"{comp}_shift"].to_numpy()
        out.append(np.abs(ex * (1 + b * sh) - ac).sum())
        base.append(np.abs(ex - ac).sum())
    return float(1 - sum(out) / sum(base)) if base else np.nan

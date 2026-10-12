"""The scheduled job (every ~6 hours, UTC, Wednesday through Monday): record the forecasts of the games kicking off soon, and in the weekly slots
(Wednesday 12:00 and Saturday 18:00 UTC) the as-of injury / depth-chart / roster / schedule tables. Runs from a fresh container: it needs no raw database.

  python -m ingest.scheduled_snapshot       prints one line: forecast rows recorded and the as-of directory (if any)
"""
from __future__ import annotations

from datetime import datetime, timezone

import config
from ingest import record_forecasts as rf
from ingest import save_asof as sa


def run(now: datetime | None = None, forecasts=None, asof=None) -> dict:
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    forecasts = forecasts or (lambda at: rf.main(at))
    asof = asof or (lambda at: sa.save(sa.fresh_frames(), sa.ASOF_DIR, at))
    out = {"forecast_rows": forecasts(now), "asof": None}
    if sa.is_asof_slot(now):
        out["asof"] = asof(now)
    return out


if __name__ == "__main__":
    r = run()
    print(f"forecast rows recorded: {r['forecast_rows']}; as-of snapshot: " + (str(r['asof'].relative_to(config.ROOT)) if r["asof"] else "none (not a weekly slot)"))

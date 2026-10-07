# Script the Slate

NFL game prediction pipeline: ingest, features, models, simulation, evaluation,
and a Streamlit app.

## The one rule

**Predictions never see a sportsbook line except through the explicit
market-aware track, which is logged separately from the pure track.**

- **Pure track:** no market data of any kind (spreads, totals, moneylines) is
  used as an input.
- **Market-aware track:** the only place a sportsbook line may enter a
  prediction. Its predictions and results are logged separately and are never
  mixed with the pure track's.

## Layout

`ingest/` `features/` `models/` `sim/` `eval/` `app/` `tests/`, plus `data/`
(`raw/`, `snapshots/`, `processed/`), which is git-ignored. Constants live in
`config.py`. Seasons 2020-2024 are used for backtesting; 2025 is a locked
holdout.

## Setup

Python 3.11+:

```bash
pip install -e .
pytest
```

## Data rules

- **Coaching identity comes from the `coaches` table, never from nflverse.** The
  schedules table's `home_coach` / `away_coach` columns lag mid-season hires and
  sometimes never update, so nothing may join or key on them.
- **Player joins go through the master `players` table** (`ingest/ids.py`).
- **Raw pulls are append-only**; downstream code keeps the latest `pulled_at`.

# 4e: observed vs forecast weather in the backtest (options for your choice; nothing built)

**The problem.**
- The weather table holds Meteostat station observations at kickoff for 2020-2025 games (1,757 rows `meteostat_observed`).
- Forecast rows (`meteostat_forecast`, with hours_before_kickoff) exist only for upcoming 2026 games (15 so far).
- The nflverse schedules' `temp` / `wind` columns are also game-time values (2016-2024, about 60-70% filled).
- A backtest on any of these uses weather nobody knew when the prediction was made, which flatters the layer. Live, the model will only have a
  forecast.

## Options

**A. Degrade the observed values to forecast quality (the gate on 2020-2024).**
- Before the model sees them, observed values are coarsened or perturbed by about a typical day-ahead forecast error. The layer must CLEAR
  on the degraded values under the one weight rule; exact observed values are reported alongside for reference only.
- Two ways, fixed in advance:
  - **A1 bins:** wind <10 / 10-15 / 15-20 / 20+ mph; temperature <=32 / 33-50 / 51-70 / >70 F; precipitation yes/no. A day-ahead
    forecast usually lands in the right bin.
  - **A2 noise:** wind +/- N(0, 4 mph), temperature +/- N(0, 4 F), precipitation flipped with probability 0.2, fixed seed. These sizes are
    my proposal, not measured; option C would measure them.
- **For:** no new data, covers the whole dev window, and fails a layer that only works with perfect weather.
- **Against:** the error model is assumed, not measured.

**B. Archived forecasts (Open-Meteo Historical Forecast API).**
- Model forecasts archived from 2021-03 (GFS) or 2022-11 (ICON) (https://open-meteo.com/en/docs/historical-forecast-api).
- But the archive is "a stitched time series" built from "the initial hours of each update"
  (https://openmeteo.substack.com/p/introducing-the-historical-forecast): close to a nowcast, not what was forecast a day ahead.
- True full runs (Single Runs API) start 2024-03 for ECMWF.
- Terms (https://open-meteo.com/en/terms): data "under the terms of the CC-BY 4.0 licence"; the free API "for non-commercial purposes";
  "Less than 10'000 API calls per day"; no AI / ML clause.
- **For:** real model output.
- **Against:** wrong lead time for 2021-2023, only 2024 has true runs, and it is a new source to approve.

**C. Record real forecasts going forward.**
- The pipeline already stores Meteostat forecasts for upcoming games with hours_before_kickoff. Keeping a fixed snapshot (for example 24
  hours before kickoff) for every 2026 game gives a true forecast-time record, and later the measured forecast error that A2 needs.
- **For:** exactly what the live model sees.
- **Against:** no backtest value now; it takes a season to accumulate.

**D. A now, C going forward (my recommendation).**
- The V1 gate is option A on 2020-2024: the layer keeps weight 1 in a market only if it CLEARS on the degraded weather. Pick A1 or A2 now.
  I lean A1 (bins): it assumes less about the error's size than A2.
- C runs from now on. After a season of forecast-time records, the layer is re-tested on real forecasts, and A2's noise sizes are replaced
  by the measured errors.
- Indoor games stay neutral, and a missing weather row is a missing feature, never zero (build plan 4e.1).

Whichever you choose goes into the plan before 4e is built.

# Script the Slate: Build Plan

Oct 7, 2026 · @liam

## How to use this plan

This takes you from an empty folder to a tested, operating free build. Each phase has four parts: what it is, a build prompt to hand to Cursor, the manual steps you do yourself (accounts, API calls, running commands, checking output), and a gate you must pass before starting the next phase. Do not skip a gate because the code ran without errors. A gate means the numbers came back and look right, not that nothing crashed.

Working style: paste each build prompt into Cursor as its own task, let it write the code, then run the manual steps yourself in a terminal. When something fails, paste the error back to me rather than to Cursor first; I can usually tell you whether it is a real problem or an expected gap (a blocked API, a missing column) before Cursor goes and invents a workaround. Cursor writes code to a spec; it does not decide what the spec means. Measured results (like the RB test's 0.62-yard gain with a confidence interval touching zero) are read market by market and decide what gets tested or diagnosed next. They never decide that a market or a layer is abandoned.

A few things stay constant across every phase. Everything is append-only: raw pulls are never overwritten, only added to, with a `pulled_at` timestamp on every row. Nothing is called proven from one season; every gate requires more than one season of walk-forward results. And the model never sees a sportsbook line unless a phase explicitly says the market-aware track does, which is logged and graded separately from the pure track.

The data is split three ways, and the split never changes. 2020-2024 is development: every model, every parameter, every threshold and every gate decision uses only this. 2025 is a one-time historical benchmark: it is run once at the end of Phase 6 to report performance and is never used to choose or tune anything. 2026 onward is the live prospective holdout. Older seasons (2016-2019) may fill history pools and training windows but are never scored.

## Phase 0: project skeleton and environment

This sets up the folder structure in 0.1, a Python environment with every library the free build needs, and a place to run it. Nothing football-specific happens yet.

**Build prompts for Cursor (one task each):**

**0.1 -- Scaffold and dependencies**

```text
Create a new Python 3.11+ project called script-the-slate. Set up this folder
structure: ingest/, features/, models/, sim/, eval/, app/, tests/,
data/raw/, data/snapshots/, data/processed/. Add a pyproject.toml (or
requirements.txt) with: nflreadpy, polars, duckdb, numpy, scikit-learn,
lightgbm, shap, streamlit, requests, pytest. Add a .gitignore that excludes
data/ entirely. Do not write any football-specific code yet.
```

**0.2 -- Config and stadium table**

```text
Add config.py that holds constants in one place:

- DUCKDB_PATH (players and coaches) and RAW_DUCKDB_PATH (all pulls): the two DuckDB file paths.
- DATA_START_SEASON = 2016. Every pull starts here.
- BACKTEST_SEASONS = 2020 through 2024. All development, tuning and gate decisions use only these.
- HOLDOUT_SEASON = 2025. A one-time historical benchmark. Never used to choose or tune anything, and any code that requests it raises an error.
- PROSPECTIVE_START_SEASON = 2026. The live holdout.
- MAX_UNMATCHED_FRACTION = 0.01 (used by ids.py). FEATURE_HISTORY_START = 2020 (the first season models and baselines read). ELIGIBLE_PLAYER_RULE: a player is eligible if he is QB1, RB1, RB2, FB1, WR1-WR3 or TE1 on the pre-game depth chart, or averaged at least 5 carries or 2 targets over his previous 4 games. The player pools below mean the eligible players of that group.
- GAME_MARGIN_SD = 13.5 (the normal-curve spread used to turn an expected margin into a home-win probability in the baselines) and ROLE_WINDOW_DAYS = 365 (the trailing window for role and opponent-allowed averages).
- MARKETS: the eleven markets, each with its key, the player pool it covers, and the CONTINUITY_PENALTIES key it uses:
  pass_att    QB1 pass attempts                        penalty key pass_yds
  pass_cmp    QB1 completions                          penalty key pass_yds
  pass_yds    QB1 passing yards                        penalty key pass_yds
  rush_att    carries by every RB-group player (RB1, RB2, other RBs and fullbacks)   penalty key rush_att
  rush_yds    rushing yards, same pool                 penalty key rush_yds
  targets     targets by WR, TE and RB (RB includes fullbacks)   penalty key rec
  rec         receptions, same pool                    penalty key rec
  rec_yds     receiving yards, same pool               penalty key rec_yds
  total       game total points                        penalty key game_total
  spread      home margin                              penalty key game_total
  moneyline   home win probability                     penalty key game_total
  The touchdown markets (pass_tds, rec_tds, anytime_td), first_half_total and team_total are added after these eleven work end to end; their rows stay in CONTINUITY_PENALTIES for that day.
- stadium_coordinates: a dict with one entry per NFL team, filled from the table below (team -> latitude, longitude, fixed_roof: outdoors / dome / retractable). A retractable roof's actual game-day state comes from the schedules table's own roof column, not this static table. This table is only for knowing where to point the weather pull.
```

| Team | Lat | Lon | Fixed roof |
| --- | --- | --- | --- |
| Arizona Cardinals | 33.5276 | -112.2626 | Retractable |
| Atlanta Falcons | 33.7554 | -84.4008 | Retractable |
| Baltimore Ravens | 39.2780 | -76.6227 | Outdoors |
| Buffalo Bills | 42.7738 | -78.7870 | Outdoors |
| Carolina Panthers | 35.2258 | -80.8528 | Outdoors |
| Chicago Bears | 41.8623 | -87.6167 | Outdoors |
| Cincinnati Bengals | 39.0954 | -84.5160 | Outdoors |
| Cleveland Browns | 41.5061 | -81.6995 | Outdoors |
| Dallas Cowboys | 32.7473 | -97.0945 | Retractable |
| Denver Broncos | 39.7439 | -105.0201 | Outdoors |
| Detroit Lions | 42.3400 | -83.0456 | Dome |
| Green Bay Packers | 44.5013 | -88.0622 | Outdoors |
| Houston Texans | 29.6847 | -95.4107 | Retractable |
| Indianapolis Colts | 39.7601 | -86.1639 | Retractable |
| Jacksonville Jaguars | 30.3239 | -81.6373 | Outdoors |
| Kansas City Chiefs | 39.0489 | -94.4839 | Outdoors |
| Las Vegas Raiders | 36.0909 | -115.1833 | Dome |
| Los Angeles Chargers | 33.9535 | -118.3392 | Dome |
| Los Angeles Rams | 33.9535 | -118.3392 | Dome |
| Miami Dolphins | 25.9580 | -80.2389 | Outdoors |
| Minnesota Vikings | 44.9736 | -93.2575 | Dome |
| New England Patriots | 42.0909 | -71.2643 | Outdoors |
| New Orleans Saints | 29.9511 | -90.0812 | Dome |
| New York Giants | 40.8128 | -74.0742 | Outdoors |
| New York Jets | 40.8128 | -74.0742 | Outdoors |
| Philadelphia Eagles | 39.9008 | -75.1675 | Outdoors |
| Pittsburgh Steelers | 40.4468 | -80.0158 | Outdoors |
| San Francisco 49ers | 37.4030 | -121.9700 | Outdoors |
| Seattle Seahawks | 47.5952 | -122.3316 | Outdoors |
| Tampa Bay Buccaneers | 27.9759 | -82.5033 | Outdoors |
| Tennessee Titans | 36.1665 | -86.7713 | Outdoors |
| Washington Commanders | 38.9078 | -76.8644 | Outdoors |

These coordinates are from memory, not a freshly verified source, so sanity-check a few against Google Maps before relying on them, and add any neutral-site or international games by hand as they're scheduled -- they won't be in this table.

**0.3 -- README and smoke test**

```text
Add a README naming the project Script the Slate and stating its one rule: predictions never see a
sportsbook line except through the explicit market-aware track, which is
logged separately from the pure track. Write a single test in tests/ that
imports every dependency and confirms the data/ folders exist.
```

**Manual steps:**

1. Create the GitHub (or local git) repo, clone it, and hand Cursor that folder.
2. After Cursor runs, create a virtual environment and install: `python -m venv .venv`, activate it, then `pip install -e .` (or `pip install -r requirements.txt`).
3. Run `pytest` and confirm the one test passes.
4. Manually fill in the 32 stadium coordinates in `config.py` (this is a one-time lookup you or I can do together; Cursor should not guess coordinates).

**Gate:** `pytest` passes, and `python -c "import nflreadpy, polars, duckdb, lightgbm, shap, streamlit"` runs with no errors. Nothing to measure yet; this phase just has to exist.

## Phase 1: data layer, coaching table, weekly snapshots

This pulls every free table, stamps it with when it was pulled, and never overwrites a row. It also loads the coaching table we already seeded from the web, and sets up the weather call. This is the single biggest gate: if the data layer is wrong, every later layer inherits the mistake silently.

**Build prompts for Cursor (one task each):**

**1.1 -- Master player id table**

```text
In ingest/pull_nflverse.py, write a function that pulls load_players() once
and saves it as the master id table in DuckDB, append-only with a pulled_at timestamp, plus a view players_current (latest pull per player) that every join uses. Every other script that joins
player data must go through this table (it matches far more reliably than
joining through weekly rosters alone), never around it.
```

**1.2 -- Pull the core play and player tables**

```text
In pull_nflverse.py, add functions to pull, for every season from DATA_START_SEASON (2016) through the current season: load_schedules, load_pbp, load_player_stats, load_snap_counts, load_rosters_weekly. Each gets a pulled_at UTC timestamp column and is appended (never overwritten) to its own DuckDB table. Re-running a pull must not create duplicate rows for the same pull timestamp.
```

**1.3 -- Pull the weekly-snapshot tables**

```text
In pull_nflverse.py, add functions to pull load_injuries and load_depth_charts
the same append-only, timestamped way as 1.2. These are the tables we will
re-pull on a Wed/Fri/Sun cadence once live, so the pulled_at timestamp matters
most here -- do not deduplicate or overwrite past pulls.
```

**1.4 -- Pull the advanced-stats tables**

```text
In pull_nflverse.py, add functions to pull load_ftn_charting (2022+),
load_nextgen_stats for rushing, receiving, and passing, load_pfr_advstats for
rush, rec, and pass (2018+), load_ff_opportunity, and load_officials. Same
append-only, timestamped pattern.
```

**1.5 -- Pull participation (training labels only)**

```text
In pull_nflverse.py, add a function to pull load_participation for
2016-2025 only (it will raise an error outside that range -- catch it and
skip rather than crash). Tag this table clearly as training-labels-only in
its schema or a comment, since it is not available for the current season.
```

**1.6 -- Weather pull with fallback**

```text
Write pull_weather.py using Meteostat (no API key and no account).

1. For every game in the schedules table, get the stadium coordinates from config.stadium_coordinates and the kickoff time. Pull hourly observations from Meteostat per stadium per calendar year (one pull per stadium-year, not one per game), then for each game pick the observation closest to kickoff.
2. Store temperature (F), wind (mph) and precipitation amount where Meteostat provides it, each row tagged source = meteostat_observed (the kickoff was already past at pull time), meteostat_forecast (the kickoff was still ahead; Meteostat returns model forecast data for roughly the next week, so also store pulled_at and hours_before_kickoff) or missing. Baselines and backtests use only observed rows. Meteostat does not provide precipitation probability, so keep a precip_prob_pct column and leave it null.
3. If no observation exists (for example a game more than about a week away), write the row as missing. Never raise an error for a missing row.
4. Store an indoor flag on each row: true for a fixed dome, or for a retractable roof that the schedules roof column says was closed. State this in the module docstring. The features layer treats indoor games as neutral weather, not as missing weather.
5. Tests: every played 2020-2025 game has a weather row; games more than about a week away are written as missing; one failed stadium-year pull does not stop the others.
```

**1.7 -- Coaching table loader**

```text
Write load_coaches.py that reads data/coaches_seed.csv (columns: team, role, name, start_year_with_team, prior_stops, source_url, confidence) into a coaches table in DuckDB, with an extra boolean column for whether each row was hand-corrected after import. This seed holds the current (2026) staff only. Historical staff for 2016 through 2025 is built separately in 4d.0, and until that exists the head-coach and offensive-coordinator continuity factors in features/weights.py fire only when a player changes teams.
```

**1.8 -- Canonical id join helper**

```text
Write ids.py with a function add_canonical_gsis_id(df) that takes a dataframe with any of {gsis_id, pfr_id, espn_id, pfr_player_id} and returns it joined to a single canonical gsis_id from the master player table (1.1). If more than MAX_UNMATCHED_FRACTION (config, 0.01) of rows in a batch fail to match, raise IdMatchError with the unmatched count and a sample of the unmatched rows. Never return a silent null. Test with a batch under the threshold (passes) and a batch over it (raises).
```

**1.9 -- Data freshness audit**

```text
Write data_audit.py: for every table pulled so far, report row count per
season, the max week present, and percent-filled for a hand-picked list of
important columns per table. Write the output to data/data_audit.json. This
is meant to be re-run every week going forward, not just once.
```

**Manual steps:**

1. Run `python ingest/pull_nflverse.py` once for seasons 2016 through the current season. This will take a few minutes and download a lot of data; let it finish.
2. Save the coaching CSV I already compiled (`coaching_staff_2026_seed.csv`, sent earlier in this chat) as `data/coaches_seed.csv`, then run `load_coaches.py`. Before trusting it, resolve the four open conflicts by hand: the Green Bay, Falcons and 49ers play callers, and the Bengals offensive coordinator. A quick web search settles each one.
3. Nothing to set up for weather. The weather pull uses Meteostat, which needs no key, no account and no contact address.
4. Run `pull_weather.py` for all games from 2020 through the current season and confirm: every played game has a weather row, games more than about a week away are written as missing, domed stadiums carry the indoor flag, and a deliberately broken stadium-year does not stop the rest of the run.
5. Run `data_audit.py` and read the output. Compare it against the Oct 6 2026 audit numbers (row counts per season and fill rates per column) -- yours should be close. If row counts are wildly different, something changed upstream and needs investigating before you go further.

**Gate:** every table loads for 2016 through the current season, `data_audit.json` shows fill rates in the same range as the Oct 6 2026 audit, the coaching table covers all 32 teams with the four conflicts resolved, and the weather pull returns a row for every played game from 2020 on.

Note on API keys. Nothing in this plan needs one. Meteostat is called without an account, and nflreadpy pulls public release files, not an authenticated API. If you ever add a paid line source like The Odds API, that is the first point in this whole build where a real secret shows up. At that point, put the key in a .env file (never committed to git) and load it with python-dotenv rather than hardcoding it in a script. The .env pattern is already set up so it is ready if that day comes.

## Phase 2: baselines and the grading harness

This builds the five baselines (last-3, season average, recency-weighted average, a 70/30 player-opponent blend, role average) for all eight player markets plus the three game markets, and a harness that scores any model walk-forward and reproduces the same numbers every time. Nothing in this phase is clever. Its only job is to be trustworthy enough that later phases can be judged against it without doubt.

**Build prompts for Cursor (one task each):**

**2.1 -- Weighting functions**

```text
In features/weights.py, implement three pure functions. No model training happens in this file.

1. recency_weight: weight = 0.5 ** (effective_games_ago / 6), half-life 6 team games. An offseason gap between two seasons counts as 8 games.
2. continuity_weight(market, past_game, target_game): the product of the per-market penalties in CONTINUITY_PENALTIES for each factor that CHANGED between the past game and the game being predicted (1.0 = no change). Detect the factors from the weekly depth charts (features/lineups.py) and the rosters:
   QB: the QB1 differs. RB_group: the top 2 RBs differ. WR_TE_group: WR1 through WR3 and TE1 differ. OL: the top 5 OL differ. role: the player's own depth-chart slot (for example RB1 vs RB2) differs. HC and OC: from the coaches table; until coaches_history exists (4d.0), HC and OC fire only when the player changed teams, and the output says so.
   'Differs' means the set of names changed at all. Write these definitions in a block comment at the top of the file.
3. quality_weight: observed 1.0, derived 0.9, estimated 0.65. It is not applied in Phase 2 or 3 (every input there is observed, so it would be 1.0); it is first applied in 4c.

Each market uses the CONTINUITY_PENALTIES row named by its penalty key in config.MARKETS.

Store CONTINUITY_PENALTIES in config.py with these 11 rows and 7 factors (multiplier applied when that factor changed). The keys are PENALTY keys, not market keys; config.MARKETS maps each of the eleven backtest markets to one of them:

CONTINUITY_PENALTIES = {
    "pass_yds":         {"QB":0.25,"HC":0.85,"OC":0.65,"OL":0.90,"role":0.50,"RB_group":0.95,"WR_TE_group":0.80},
    "pass_tds":         {"QB":0.25,"HC":0.85,"OC":0.65,"OL":0.90,"role":0.50,"RB_group":0.95,"WR_TE_group":0.75},
    "rush_yds":         {"QB":0.90,"HC":0.85,"OC":0.80,"OL":0.65,"role":0.35,"RB_group":0.65,"WR_TE_group":0.95},
    "rush_att":         {"QB":0.90,"HC":0.85,"OC":0.80,"OL":0.65,"role":0.35,"RB_group":0.65,"WR_TE_group":0.95},
    "rec_yds":          {"QB":0.55,"HC":0.85,"OC":0.70,"OL":0.95,"role":0.35,"RB_group":0.90,"WR_TE_group":0.65},
    "rec":              {"QB":0.55,"HC":0.85,"OC":0.70,"OL":0.95,"role":0.35,"RB_group":0.90,"WR_TE_group":0.65},
    "rec_tds":          {"QB":0.50,"HC":0.85,"OC":0.70,"OL":0.95,"role":0.30,"RB_group":0.88,"WR_TE_group":0.60},
    "anytime_td":       {"QB":0.60,"HC":0.85,"OC":0.70,"OL":0.90,"role":0.30,"RB_group":0.70,"WR_TE_group":0.70},
    "game_total":       {"QB":0.55,"HC":0.80,"OC":0.70,"OL":0.90,"role":0.90,"RB_group":0.95,"WR_TE_group":0.90},
    "first_half_total": {"QB":0.55,"HC":0.80,"OC":0.70,"OL":0.90,"role":0.90,"RB_group":0.95,"WR_TE_group":0.90},
    "team_total":       {"QB":0.50,"HC":0.80,"OC":0.68,"OL":0.88,"role":0.90,"RB_group":0.92,"WR_TE_group":0.88},
}

Tests: no change in any factor gives 1.0; a QB change on pass_yds gives 0.25; the offseason gap counts as 8 games; every market in config.MARKETS resolves to an existing penalty key.
```

**2.2 -- Baseline predictors**

```text
In eval/baselines.py, implement five baseline predictors for every market in config.MARKETS, using the player pools defined there (the RB group means every eligible RB-group player under ELIGIBLE_PLAYER_RULE: RB1, RB2, fullbacks and any other RB who qualifies):

1. last_3: the average of the last 3 games.
2. season_avg: the season-to-date average.
3. recency_weighted: the average weighted by features/weights.py recency_weight.
4. blend_70_30: 70% the player's own recency-weighted average plus 30% the opponent's allowed average for that role over the trailing ROLE_WINDOW_DAYS (365).
5. role_avg: the average for his position and depth-chart slot group over the trailing ROLE_WINDOW_DAYS.

Game markets use team-level versions of the same five on points scored and allowed: total = expected points for both teams, spread = expected home margin, moneyline = the normal CDF of the expected margin divided by GAME_MARGIN_SD.

Each baseline must use only data available before the game being predicted. Write them as pure functions that take a cutoff date, not code that peeks at the full season.

Tests: changing any result from the target week onward leaves every prediction for that week unchanged; a player with no history returns missing, not zero.
```

**2.3 -- Walk-forward harness**

```text
In eval/backtest.py, write a walk-forward harness. For each week W in BACKTEST_SEASONS (2020-2024), fit or compute using only weeks before W, predict week W, and store the prediction and the actual result. Any request for HOLDOUT_SEASON (2025) or later raises an error.

A predictor is any function that takes the cutoff and returns a dict of market -> predictions. The harness scores only the markets the predictor returns, and accepts extra per-game actuals (for example team plays) when a market needs them. A new predictor must plug in without changing the harness.

Tests: running twice gives identical results; requesting 2025 raises an error.
```

**2.4 -- Grading functions**

```text
In eval/grade.py, implement: mean absolute error, Brier score at each
threshold-ladder rung, log loss, a calibration table (bucket predictions by
stated probability in 20% bands, compare to actual frequency), the
whole-curve check (where the actual result falls in the predicted
distribution -- give each baseline a simple spread estimated from its own
historical residuals so it has a distribution, not just a point estimate),
and skill-versus-baseline (1 minus the model's Brier divided by the
baseline's).
```

**2.5 -- Run and save the baseline backtest**

```text
Write run_baseline_backtest.py that runs all five baselines (2.2) through the harness (2.3) for all eleven markets in config.MARKETS across 2020-2024, grades them (2.4), and writes the results to baseline_results.parquet plus a human-readable summary table printed to the console. For each market, name the best baseline on identical rows and store that choice, because later phases compare against it. Running the script twice must produce identical files.
```

**Manual steps:**

1. Run `run_baseline_backtest.py` and read the summary table. For RB rushing yards, the recency-weighted baseline should land somewhere in the same range as the Oct 6 test (28.37 yards MAE on 2024-2025 data; your 2020-2024 number will differ some because it's a different, larger sample, but it shouldn't be wildly off).
2. Spot-check by hand: pick one player-week, work out what the recency-weighted average should be with pen and paper, and confirm the code matches. Do this once per market type (a rushing market and a passing market) -- it catches off-by-one week errors that are easy to introduce and easy to miss.
3. Confirm the whole-curve check comes back roughly flat for at least one baseline. A baseline that already passes this sanity check means the grading code itself is working, which matters more right now than whether the baseline is good.

**Gate:** a reproducible table of baseline error and calibration for all eleven markets, saved as an artifact file, with 2025 completely untouched by this phase. If you re-run the backtest script twice, it must produce identical numbers.

## Phase 3: volume x efficiency model per market -- the baseline every later layer must beat

This is the RB test extended to every market. It measures how much signal recent usage plus a gradient-boosted regression captures on free data before comparables, injuries and simulation are added. The result becomes the bar those layers have to beat in Phase 4, market by market. It does not decide whether the comparables engine gets built. That engine is the point of the project and is tested on its own in 4c.

**Build prompts for Cursor (one task each):**

**3.1 -- Volume models**

```text
In models/volume.py, build one LightGBM regression per volume quantity:
- pass_att: QB1 pass attempts.
- rush_att: carries by every eligible RB-group player (ELIGIBLE_PLAYER_RULE; RB1, RB2 and fullbacks all stay in the rush pool).
- targets: targets by WR, TE and RB.
- team_plays: total offensive plays, the volume side of the game markets.

Features, each computed only from games before the row's own week cutoff (put the code in features/volume_features.py):
- Own history: recency-weighted and recency-times-continuity-weighted trailing stats via features/weights.py, plus his share of team usage.
- Team rates: rush attempts, dropbacks, plays, pass rate, points.
- Opponent allowed: the same five rates, plus what the opponent allowed that role over the trailing ROLE_WINDOW_DAYS.
- Context: depth-chart slot, home or away, game number in the season.
- No sportsbook line of any kind.

Hyperparameters (a starting point, not a fixed spec): 150 trees, 8 leaves, minimum 30 rows per leaf, L1 objective, learning rate 0.05, fixed random seed. Train through the walk-forward harness (2020-2024), retraining each week on only earlier weeks. Never touch 2025. No comparables, injuries or simulation yet.

Tests: altering any stat, score or lineup from week W onward leaves every earlier feature unchanged; the recency feature matches the baseline recency prediction to 9 digits on a sample of rows; two runs give identical results; requesting 2025 raises an error; team pass rates are fractions between 0 and 1 (DuckDB returns sums as decimals, so cast to float).
```

**3.2 -- Efficiency models**

```text
In models/efficiency.py, build one LightGBM regression per efficiency quantity, with the same feature philosophy (own history, team rates, opponent allowed, context, no lines), the same hyperparameter starting point and the same walk-forward training as 3.1, and the same exclusion of 2025 and of comparables, injuries and simulation:
- completion rate (completions per attempt), for pass_cmp.
- yards per completion, for pass_yds.
- yards per carry, for rush_yds.
- catch rate (receptions per target), for rec.
- yards per reception, for rec_yds.
- points per play per team, for total, spread and moneyline.

In models/combine.py, combine volume and efficiency into each market's final prediction:
pass_att = volume; pass_cmp = pass_att x completion rate; pass_yds = pass_att x completion rate x yards per completion; rush_att = volume; rush_yds = rush_att x yards per carry; targets = volume; rec = targets x catch rate; rec_yds = targets x catch rate x yards per reception; team points = team_plays x points per play; total = both teams' points added; spread = home minus away; moneyline = the normal CDF of the spread divided by GAME_MARGIN_SD.
```

**3.3 -- Run and compare against baselines**

```text
Write run_volume_efficiency_backtest.py. It runs the five baselines and the combined volume-times-efficiency model (3.1, 3.2, combine.py) through the phase 2 harness for all eleven markets in config.MARKETS across 2020-2024.

For each market:
1. Pick the best baseline on identical rows.
2. Gain = best-baseline error minus model error (MAE for the player and total markets, Brier for moneyline, so positive means better).
3. 95% confidence interval on the gain from a bootstrap that resamples whole season-weeks with replacement (players in the same week are not independent), 10,000 resamples, fixed seed. Also report the plain row-level interval for reference.
4. Per-season gains and the number of seasons won.
5. Mean bias (predicted minus actual).
6. A verdict: CLEARS if the gain is positive in more than one season separately and the interval excludes zero; EDGE if the gain is positive but the interval touches or crosses zero; NO otherwise.

Write the per-market results and the summary to volume_efficiency_results.parquet. Running it twice must give a byte-identical file. Do not average across markets.
```

**3.4 -- Controls (does not block Phase 4)**

```text
Write run_phase3_controls.py. It measures how much of the Phase 3 gain is real and does not block Phase 4.

1. Feature control: the same LightGBM using only the player's recency-weighted usage feature. Report its gain against the best baseline and against the full Phase 3 model.
2. Objective control: add a recency-weighted MEDIAN baseline per market and report the full model's gain against it. The Phase 3 models use an L1 objective, which predicts medians, while the baselines predict means, so part of the gain may come from matching the loss function rather than from better features. Also report each market's mean bias.

One row per market with gains, 95% intervals from the same season-week bootstrap as 3.3 (10,000 resamples, fixed seed), per-season gains and the bias. Save to phase3_controls_results.parquet.
```

**Manual steps:**

1. Run the backtest and look at the summary table market by market. For each market, you want: the model beats the best baseline, in more than one season separately (not just pooled), with a confidence interval that excludes zero.
2. Do not average across markets to get a comfortable overall number. A model that wins big on QB attempts and loses on RB rushing yards is not "net positive" -- it means something is wrong with the RB rushing yards model specifically, or that market is just harder (small sample, high noise, more injury variance).
3. For any market where the interval touches or crosses zero (the RB rushing yards test landed right at this edge: 0.62 yards, CI \[-0.02, 1.27\]), flag it. The market stays in the build. Diagnose it: too few examples, a feature that is leaking or useless, a baseline stronger than expected, or the median-versus-mean mismatch measured in 3.4. Re-test it after every Phase 4 layer, because a layer may be exactly what lifts it.

**Gate:** the volume and efficiency models run for all eleven markets, walk-forward on 2020-2024, the results reproduce identically on a second run, and the per-market table is saved with its 95% intervals. That table is the bar every Phase 4 layer is measured against, market by market. Phase 4 starts regardless of how many markets clear their baseline here. A market that does not clear it is something to diagnose (features, objective, baseline), not a reason to stop and not a reason to drop the market. Comparables, injuries and simulation are exactly what this base model does not have.

Phase 3 results recorded Oct 7 2026 (2020-2024 walk-forward; gain = best-baseline error minus model error, positive is better; 95% interval from the season-week bootstrap; file volume\_efficiency\_results.parquet, reproduced byte-identical on a second run). EDGE and NO markets stay in the build and are re-tested after every Phase 4 layer.

| Market | Best baseline | Gain | 95% CI | Seasons won | Verdict |
| --- | --- | --- | --- | --- | --- |
| Pass attempts | blend 70/30 | +0.22 | \[+0.08, +0.37\] | 4/5 | CLEARS |
| Pass completions | blend 70/30 | +0.10 | \[+0.012, +0.198\] | 4/5 | CLEARS (marginal) |
| Rush attempts | recency | +0.10 | \[+0.065, +0.144\] | 4/5 | CLEARS |
| Rush yards | recency | +0.53 | \[+0.28, +0.77\] | 4/5 | CLEARS |
| Targets | recency | +0.027 | \[+0.018, +0.037\] | 5/5 | CLEARS |
| Receptions | recency | +0.020 | \[+0.013, +0.026\] | 5/5 | CLEARS |
| Receiving yards | recency | +0.60 | \[+0.48, +0.72\] | 5/5 | CLEARS |
| Pass yards | blend 70/30 | +0.94 | \[-0.27, +2.22\] | 3/5 | EDGE |
| Total | blend 70/30 | +0.15 | \[-0.09, +0.40\] | 4/5 | EDGE |
| Spread | recency | -0.04 | \[-0.26, +0.19\] | 3/5 | NO |
| Moneyline (Brier) | recency | -0.003 | \[-0.009, +0.004\] | 1/5 | NO |

## Phase 3.5: bridge the built code to this plan

Phases 0 through 3 were built before this plan was tightened. This phase brings the code that already exists into line with the plan as it now reads, and builds the few tables Phase 4 reads that do not exist yet. Nothing here changes a model, feature, baseline or result unless the audit finds code that disagrees with the plan, and then you decide before anything result-changing is touched.

Build prompts for Cursor (one task each, in order):

**3.5.1 -- Audit the built code against the plan (report only)**

```text
Write scripts/audit_vs_plan.py and run it. It compares what is actually built against the spec below, prints a table, and writes docs/audit_vs_plan.md. In this task do NOT change any model, feature, baseline, config value or result. Report only. For each item print: the item, what the plan says, what the code does (file and line), and MATCH or DIFF.

1. config.py: DB_PATH; DATA_START_SEASON = 2016; BACKTEST_SEASONS = 2020 through 2024; HOLDOUT_SEASON = 2025 (any request for it raises an error); PROSPECTIVE_START_SEASON = 2026; MAX_UNMATCHED_FRACTION = 0.01; GAME_MARGIN_SD = 13.5; ROLE_WINDOW_DAYS = 365; MARKETS with exactly these eleven keys: pass_att, pass_cmp, pass_yds, rush_att, rush_yds, targets, rec, rec_yds, total, spread, moneyline, each carrying its player pool and its continuity-penalty key (pass_att, pass_cmp, pass_yds -> pass_yds; rush_att -> rush_att; rush_yds -> rush_yds; targets, rec -> rec; rec_yds -> rec_yds; total, spread, moneyline -> game_total); CONTINUITY_PENALTIES with the eleven penalty rows.
2. Pulls: every table pulled from 2016 through the current season, append-only with pulled_at; participation 2016-2025 only and tagged training-labels-only; FTN from 2022; PFR from 2018.
3. Weather: Meteostat; one pull per stadium-year; closest hourly observation to kickoff; source tag meteostat or missing; precip_prob_pct always null; an indoor flag column on each row; unplayed games written as missing.
4. ids.py: add_canonical_gsis_id raises IdMatchError above MAX_UNMATCHED_FRACTION.
5. features/weights.py: recency = 0.5 ** (games_ago / 6) with the offseason counted as 8 games; continuity factors detected as QB1 differs, top 2 RBs differ, WR1 to WR3 plus TE1 differ, top 5 OL differ, own depth-chart slot differs, HC and OC only on a team change until coaches_history exists; quality 1.0 / 0.9 / 0.65; every market resolves to a penalty key through config.MARKETS.
6. eval/baselines.py: five baselines for all eleven markets; the RB group includes RB1, RB2, other RBs and fullbacks; the targets pool is WR, TE and RB; role and opponent-allowed averages use ROLE_WINDOW_DAYS; moneyline is the normal CDF of the expected margin divided by GAME_MARGIN_SD; all are cutoff-based pure functions.
7. eval/backtest.py, eval/grade.py, run_baseline_backtest.py: harness never touches 2025; grading functions as in 2.4; the best baseline per market on identical rows is stored.
8. models/volume.py, models/efficiency.py, models/combine.py: the quantities, features and hyperparameters in 3.1 and 3.2, and the combine formulas in 3.2.
9. eval/compare.py and run_volume_efficiency_backtest.py: season-week bootstrap with 10,000 resamples and a fixed seed; row-level interval also reported; per-season gains; mean bias; verdicts CLEARS, EDGE and NO defined as in 3.3; a second run gives a byte-identical parquet.
10. Tests: future-blindness, holdout lock, determinism, recency-matches-baseline; the full suite passes.

Then STOP and show me the table.
```

**3.5.2 -- Apply the fixes (decisions made after the audit)**

```text
Apply these fixes from the audit. None of them may change a number. After all of them, re-run run_baseline_backtest.py and run_volume_efficiency_backtest.py. baseline_results.parquet and volume_efficiency_results.parquet must be byte-identical to the committed versions (volume_efficiency_results.parquet is commit e7e08c5). If either differs, stop and show me a per-market difference.

1. config.py: add DATA_START_SEASON = 2016 and PROSPECTIVE_START_SEASON = 2026. Move MAX_UNMATCHED_FRACTION into config.py and import it in ingest/ids.py (value unchanged). Keep DUCKDB_PATH and RAW_DUCKDB_PATH as they are; two databases is fine. Do not add DB_PATH.
2. Holdout lock: any training or evaluation loader asked for HOLDOUT_SEASON (2025) or later through max_season raises an error, the same as the harness. The raw pull functions are exempt, since pulls must still fetch 2025 and 2026 data. Add a test.
3. config.MARKETS: build the single dict (eleven keys, each with its player pool, its continuity-penalty key and its baseline family) from the pieces now spread across eval/baselines.py, eval/backtest.eligible_markets() and BASELINE_TO_PENALTY_MARKET. Make those places read from config.MARKETS and keep BASELINE_TO_PENALTY_MARKET as an alias. Add a test that the resolved mapping equals the old one for every market.
4. players table: stop using CREATE OR REPLACE. Append each pull with pulled_at and add a view players_current (latest pull per player) that every master-id join uses. Add a test that the 99.8% join rate and the canonical id outputs are unchanged.
5. Weather: add an indoor column (fixed dome, or a retractable roof that the schedules roof column says was closed). Split the source tag into meteostat_observed (the kickoff was already past at pull time), meteostat_forecast (the kickoff was still ahead) and missing. Forecast rows also store pulled_at and hours_before_kickoff. Baselines and backtests use only observed rows. Add a test.
6. Best baseline: write the chosen best baseline per market, on identical rows, to a new file best_baseline.parquet from run_baseline_backtest.py. baseline_results.parquet itself must not change.
7. Continuity detection: if one implementation can serve both weights.continuity_weight() and the feature code in volume_features.py and lineups.py with identical numbers, make weights.py the single implementation. If making them one changes any number, do NOT do it; stop and tell me what differs.
8. quality_weight() stays unused in Phase 2 and 3, since every input there is observed (1.0). It is first applied in 4c. Put a one-line note in the docstring.
9. Do not change the WR/TE group definition (WR1 through WR3 plus TE1). Record it in the block comment at the top of weights.py.
```

**3.5.2b -- Pull 2016 through 2019 without changing any result**

```text
FIRST run pull_players() so the master id table is current and covers players from 2016 on (the 3.5.2 check found 13.1% of rosters_weekly ids unmatched, probably the 2026 rookie class). It appends, and the players_current view picks it up. Print the join rate per season for 2016 through 2026 before and after. The 2020-2024 join rates must not drop. THEN pull every table that has data for 2016 through 2019 (pulls start at DATA_START_SEASON, FTN from 2022, PFR from 2018), append-only with pulled_at, into the same raw database. This changes data only. Models, baselines and features must keep reading from 2020 on exactly as they do now: add a named constant FEATURE_HISTORY_START = 2020 and make every loader use it. Then re-run both backtests and confirm both parquet files are byte-identical to the committed versions. Re-run data_audit.py and report row counts per season for the new years. Whether FEATURE_HISTORY_START moves to 2016 is a separate decision for later; do not change it.
```

**3.5.2c -- Replace the slot-based pools with the eligibility rule and add QB rushing as its own markets (result-changing, compare before accepting)**

```text
The graded pools for rushing and receiving currently use depth-chart slots 1-2 only, so RB3 and below and unlisted RBs are never scored. Replace them with one eligibility rule, kept in config.ELIGIBLE_PLAYER_RULE and used everywhere in the plan: a player is eligible for a game if he is RB1, RB2, FB1, WR1 through WR3, or TE1 on the pre-game depth chart, OR averaged at least 5 carries or 2 targets over his previous 4 games. Apply it to rush_att, rush_yds, targets, rec and rec_yds. Fullbacks who carry or catch passes must be scored.

QUARTERBACKS ARE NOT IN THE RB POOL. rush_att and rush_yds are RB-group only (RB1, RB2, FB1, and any other RB or fullback who meets the rule). A QB who runs a lot must not enter them through the carries clause.

QB RUSHING IS ITS OWN PAIR OF MARKETS. Add qb_rush_att and qb_rush_yds to config.MARKETS, which makes thirteen markets. Pool: the team's QB1 for a game, if he averaged at least QB_RUSH_MIN_ATT (config, starting at 4) rush attempts over his previous 4 games. Designed runs and scrambles count; kneel-downs do not. Use the rush_att and rush_yds continuity-penalty keys for these two markets and say so in a config comment as an assumption. Add the volume quantity qb_rush_att to models/volume.py and the efficiency quantity QB yards per carry to models/efficiency.py. In models/combine.py, qb_rush_yds = qb_rush_att x QB yards per carry. Run the same five baselines on the QB pool.

Do NOT overwrite anything. Rename the committed results to baseline_results_slotpool_v1.parquet and volume_efficiency_results_slotpool_v1.parquet. Run the baselines and the Phase 3 backtest on the new pools into new files (eligibility_v2). Print a before-and-after table for the original eleven markets (against the slotpool_v1 files) and show the two QB markets on their own: graded rows, best-baseline error, model error, gain, 95% interval, seasons won, verdict. Also report how many QBs and how many QB games are in the QB pool. Then STOP. I decide whether the new pools become the official Phase 3 table.
```

**3.5.3 -- Build the tables Phase 4 reads**

```text
Persist these four tables to DuckDB and parquet, from the walk-forward runs over 2020-2024 (2016-2019 only where a feature needs history, never scored), using only the Phase 3 models and features, with no comparables, injuries or simulation. Every row records the cutoff it was built with, and nothing in a row may use information from its own week or later.

1. walkforward_predictions: one row per player-game (or team-game) per quantity. Columns: season, week, game_id, player_id or team, role, quantity (pass_att, rush_att, targets, team_plays, qb_rush_att, completion rate, yards per completion, yards per carry, QB yards per carry, catch rate, yards per reception, points per play), expected (the comp-free prediction), actual, residual (actual minus expected), model_version, trained_through_week. Phase 4c reads this for its comp-free expectations and residual spreads.
2. market_predictions: one row per player-game per market. Columns: season, week, game_id, player_id or team, market, the combined model prediction, all five baseline predictions, which baseline was best, actual, model_version. Every Phase 4 ablation compares against these rows.
3. player_game_roles: one row per player-game. Columns: season, week, game_id, team, player_id, position, expected_role from the pre-game depth chart (QB1, RB1, RB2, FB, WR1, WR2, WR3, slot, TE1, TE2, other), played_role from snap counts, snap share, and his share of the team's carries, targets and dropbacks that game. Phase 4a and 4c read this.
4. team_game_rates: one row per team-game. Columns: plays, dropbacks, pass rate, rush attempts, points scored and allowed, sacks and scrambles per dropback, the share of plays spent in each score state (trailing 9+, trailing 1-8, tied, leading 1-8, leading 9+) and the dropback rate in each state. Phase 4b reads this.

Tests: each table has the expected row count per season; trained_through_week is always before week; the table content is identical on a second run; no 2025 rows exist; the share columns in player_game_roles sum to no more than 1 per team-game.
```

**3.5.4 -- Build the Phase 3 controls**

Run the 3.4 prompt now, so the controls table exists before the Phase 4 layers are measured against the Phase 3 model.

Manual steps: read the audit table from 3.5.1 line by line and decide each Class B item from 3.5.2 (paste the table back to me and we will go through them). Confirm the two parquet files came back byte-identical after the Class A fixes. Spot-check player\_game\_roles by hand for five players you know (an RB1, an RB2, a fullback, a WR1 and a slot receiver) on one real week, and confirm the roles and shares match what you remember of that game.

**Gate:** the audit shows MATCH on every item, or each remaining DIFF has a decision recorded; the baseline and Phase 3 result files are unchanged or every change is explained market by market; and the four tables from 3.5.3 plus the controls table exist for 2020-2024. Phase 4 starts after this passes. It is a correctness check on what already exists, not a test of whether the model works, so it never decides that a market or layer is dropped.

## Phase 4: add layers one at a time

Each of these is its own sub-phase with its own gate: injuries and role shifts, the game-state simulation, comparables, coaching priors, weather. Build and test each in isolation before combining them, because if you add three layers at once and accuracy improves, you won't know which one did it -- or whether one helped and another quietly hurt while the first one covered for it.

### 4a -- Injuries and role shifts

**4a.1 -- Status-to-probability model**

```text
In models/injuries.py, build the status-to-probability model.

1. Join the injuries table to snap counts. Group by position and by report-status plus practice-status combination (for example Questionable with DNP Wednesday, limited Thursday, limited Friday). Use the practice-status trajectory across the week, not just the final designation.
2. For each group measure two things from past games: P(play) = how often the player actually played, and snap_share_given_play = his snaps that game divided by his own trailing normal snap share (so a player who plays at 60% of his usual workload shows 0.6).
3. Shrink thin groups toward the position-level rate with n/(n+k) (start k = 10). Never let a group of fewer than 10 observations stand on its own.
4. Timing: the probability for week W uses only data from weeks before W for fitting, and the status snapshot that existed at prediction time (Friday report for the main run, Sunday snapshot for late inactives). Never use a status posted after the cutoff.
5. Output a function expected_snap_share(player, game, as_of) = P(play) * snap_share_given_play, plus the two parts separately.
6. Players on IR, PUP, or suspended (from the rosters table) get P(play) = 0 without going through the model.
7. Tests: the fitted rates match the audit numbers within 2 percentage points (Out plays 0%, Doubtful 0.5%, Questionable 66%, no designation 93.6%); a status posted after the cutoff never changes the output; future weeks never change the fit for earlier weeks.
```

**4a.2 -- Role redistribution**

```text
In models/injuries.py, add role redistribution: when a teammate will miss time, move his workload to the right players.

1. Roles: QB, RB1, RB2/FB, WR1, WR2, WR3, slot, TE1, TE2, and OL group. For a game where a player in role r is likely out (use P(play) from 4a.1, not a yes/no), find past games where that team played without that player or without anyone in role r. Measure how each remaining player's share of carries, targets, dropbacks and snaps moved versus his own trailing baseline.
2. Shrink the team-specific shift toward the league-wide shift for role r with n/(n+k) (start k = 4), so one past game does not drive it.
3. Apply the shift as an expectation across the two scenarios (player plays with snap_share_given_play, player is out), weighted by P(play). Output expected shares per remaining player for the game.
4. Replacement quality matters: when the replacement is a different quality player, carry his own trailing efficiency forward instead of the starter's.
5. Shares must stay consistent: for each team, expected carry shares sum to 1 and target shares sum to 1. If nobody is out, output equals the baseline exactly.
6. These expected shares feed the volume and efficiency models as features (4a.4) and the lineup-adjusted vectors in 4c.1.
7. Tests: the share-sum identity; no-injury equals baseline; a planted missing WR1 raises the other receivers' targets in the direction history says.
```

**4a.3 -- In-game injury exits**

```text
In models/injuries.py, add in-game injury exits as a separate draw.

1. Define an early-exit game and write the rule in the file header: a player who played less than half of his usual snaps AND is on the next week's injury report with a designation tied to that game. Use the same rule everywhere.
2. Build each player's workload baseline ONLY from games that are NOT early-exit games. Do not let the same tail event be counted in both the baseline and the exit draw.
3. Separately model P(early exit) by position, adjusted by the player's own history and shrunk toward the position rate with n/(n+k) (start k = 15), and the share of the game he completes when he does exit (an empirical distribution by position).
4. Output a function that, for a player and game, returns P(early exit) and a sampler for share completed, for the 4b simulation to draw from.
5. Tests: the baseline excludes exit games; baseline plus exit draw reproduces the historical mean workload within 3% in aggregate; no future data.
```

**4a.4 -- Wire injuries into the Phase 3 models and re-run**

```text
Add expected_snap_share, the redistributed shares from 4a.2 and P(early exit) as features to the volume and efficiency models in models/volume.py and models/efficiency.py. Same hyperparameters, same walk-forward, 2025 never touched. Write run_injury_ablation.py: for every market, run the model with and without the injury features through the phase 2 harness (2020-2024), report the gain, the 95% CI from the same season-week bootstrap as 3.3 (10,000 resamples, fixed seed), per-season gains, and the bias. One row per market, no averaging across markets. Save to injury_ablation_results.parquet. Running twice must give identical numbers.
```

Manual steps: print five real injury situations from last season as before-and-after cards (a team's RB1 out, a WR1 out, a QB switch, a Questionable player with a DNP on Wednesday, an offensive line starter out). Each shows P(play), the shares before and after redistribution, and who got the workload. Check them against what actually happened in those games.

Gate: with the injury features, the Phase 3 backtest improves or at minimum does not get worse, in more than one season, for the markets it should plausibly affect (a missing WR1 should move his team's other receivers). Judged on 2020-2024 only. A market with no lift keeps the layer at zero weight and is flagged, not dropped from the build.

### 4b -- Game-state simulation

**4b.1 -- Team ratings**

```text
In features/team_ratings.py, build opponent-adjusted team ratings.

1. Four ratings per team: offense pass, offense rush, defense pass, defense rush, each in EPA per play. Fit with ridge regression on all games before the target week: per-play EPA for that play type = league mean + offense rating of the team with the ball + defense rating of the other team + home-field term. Weight games with the recency half-life from features/weights.py.
2. Bootstrap (200 resamples of games) to get a standard deviation for each rating. Store the mean and the SD.
3. Week 1 and early-season start: begin each team at last season's final ratings multiplied by a carryover factor (start 0.6, tuned ONLY on 2020-2024), shrunk toward league average, then blend toward current-season data with n/(n+k) (start k = 6 games).
4. Leave an input hook for a coaching prior (a starting mean and strength per rating) that defaults to nothing. 4d fills it. Do not build coaching logic here.
5. Tests: ratings for week W use only games before W; a team's offense and defense ratings sum to roughly zero across the league; identical runs give identical ratings.
```

**4b.2 -- Expected margin with starting-QB adjustment**

```text
In models/game_state.py, build the expected margin and total.

1. Expected points for each side = league average points + that side's offense ratings (pass and rush, scaled by expected plays and run/pass mix) against the other side's defense ratings, plus home field, plus a rest-difference term. Fit the scaling terms on 2020-2024 walk-forward data.
2. STARTING-QB ADJUSTMENT: gap between this week's expected starter and the quarterback the team ratings were built on, using each QB's EPA per dropback shrunk toward the league mean with n/(n+k) (start k = 200 dropbacks), converted to points through a fitted coefficient (points per 0.1 EPA/dropback times expected dropbacks). If the starter is unknown, use a probability-weighted blend of the possible starters.
3. The margin and total come ONLY from this model. Never from a sportsbook line, closing line or schedules-table spread.
4. Output the mean and the standard deviation of the margin and of the total, with the SDs estimated from prior walk-forward errors (not a fixed constant).
5. Tests: swapping home and away flips the margin and keeps the total; no market line anywhere in the inputs; walk-forward only.
```

**4b.3 -- Dropback rate by state and play accounting**

```text
In models/game_state.py, build the dropback-rate-by-state model and the play accounting.

1. STATES (by score difference from the team's point of view): trailing 9+, trailing 1-8, tied, leading 1-8, leading 9+.
2. STATE SHARES: fit from history how a team's share of plays in each state depends on the pregame expected margin (multinomial logistic regression or binned averages), so the simulation can turn a drawn margin into state shares.
3. DROPBACK RATE per team per state: the team's observed dropback rate in that state, shrunk toward the league average for that state with n/(n+k) (start k = 100 plays), where n is how many plays the team has in that state.
4. ACCOUNTING, implemented exactly:
   dropbacks = plays * sum over states(state share * dropback rate in that state)
   pass attempts = dropbacks - sacks - scrambles
   rush attempts = plays - dropbacks + scrambles
   sacks and scrambles come from per-dropback rates, shrunk toward league average.
5. Write a unit test that confirms the accounting identity holds exactly on synthetic data, and that state shares sum to 1.
```

**4b.4 -- The 20,000-simulation Monte Carlo loop**

```text
In sim/simulate.py, build the game simulation.

For each simulated game, in this order:
1. Draw the margin from the 4b.2 mean and SD.
2. Convert the margin to state shares (4b.3).
3. Draw total plays from the team's expected plays and its walk-forward residual SD.
4. Draw dropback rate by state (4b.3), then split into sacks, scrambles, pass attempts and rush attempts using the exact accounting.
5. Draw ONE shared passing-efficiency shock and ONE shared rushing-efficiency shock per team, so players on the same team move together. Set the shock SDs so the simulated within-team correlation of player residuals matches the historical correlation.
6. For each player: draw his share of the team's carries, targets and dropbacks from the expected shares (from 4a.2), apply the 4a.3 early-exit draw, then draw efficiency as the Phase 3 efficiency model prediction plus the team shock plus a player-level residual sampled from the empirical walk-forward residuals for that market and role.
7. Run 20,000 simulated games per real game. Use a fixed random seed per game so reruns are identical.
8. Store results once per game: simulation_run_id, game_id, simulation_id, then every player's outcomes in that simulated game. Never store once per individual prediction, so every prop for that game reads off the same stored set.
9. Provide a mode with fewer simulations (2,000) for backtesting, clearly flagged in the output.
```

**4b.5 -- Run the simulation through the backtest**

```text
Write run_sim_backtest.py. Run the simulation (2,000 draws per game) through the phase 2 harness for 2020-2024, never 2025. For every market, turn the simulated outcomes into the market's mean and median prediction and into threshold probabilities. Compare against the Phase 3 model: MAE, Brier score for threshold probabilities, calibration table, and per-season gains with the 95% CI from the same season-week bootstrap as 3.3 (10,000 resamples, fixed seed). Report simulated vs actual margin, total and team-plays distributions (mean, SD, quantiles, and a plot). One row per market, no averaging across markets. Save to sim_backtest_results.parquet. Running twice must give identical numbers.
```

Manual steps: time one week's 20,000-draw slate on your laptop and confirm it finishes in minutes, not hours. Plot simulated vs actual margin, total and team-plays distributions and read them yourself before trusting anything downstream.

Gate: simulated game outcomes (final scores, team plays) match the historical distribution of real games reasonably well, and the game-level markets and the player markets are each re-tested against the Phase 3 model on 2020-2024. A market with no lift keeps the simulation at zero weight and is flagged, not dropped from the build.

### 4c -- Comparables engine

**4c.0 -- Component, market and search definitions (build this first)**

```text
Create models/comps_spec.py holding constants only, no logic. Every other 4c task imports from it. Do not add, drop or rename anything below without asking me.

1. UNITS. One standardized vector per unit, per team per game, per window.
   OFFENSE units: run_offense, pass_offense, rb_rotation, ol_protection, receiver_usage.
   DEFENSE units: run_defense, pass_rush, pass_coverage, coverage_mix.
   PLAYER ARCHETYPES (one vector per player per window): rb_archetype, receiver_archetype (WR, TE and pass-catching RB), qb_archetype.
   Never collapse these into one offense vector and one defense vector. The whole point is that one part of a team can resemble one historical team while another part resembles a completely different one.

2. FEATURES per unit. Build only from columns that exist in the loaded tables (play-by-play, FTN, PFR, snap counts, rosters). Start from this list, drop what is not available, add nothing silently:
   - run_offense: rush EPA/play, rush success rate, yards/carry, explosive run rate (10+ yards), rush rate over expected, run-location and run-gap shares, shotgun-run share.
   - pass_offense: EPA/dropback, CPOE, yards/attempt, explosive pass rate, average depth of target, sack rate, play-action rate (FTN), screen rate (FTN), motion rate (FTN), no-huddle rate.
   - rb_rotation: RB1 share of RB carries, RB2 share, RB1 snap share, RB target share, goal-line carry share.
   - ol_protection: sack rate allowed, pressure rate allowed (PFR/FTN), run-block EPA proxies available in play-by-play (rush EPA by run location).
   - receiver_usage: target share of the top 3 targets, WR/TE/RB target split, average depth of target by position, slot vs outside rate where available.
   - run_defense: rush EPA allowed, success rate allowed, yards/carry allowed, explosive runs allowed, box-count faced (FTN), run-location splits allowed.
   - pass_rush: sack rate, pressure rate (FTN/PFR), blitz rate (FTN), QB-hit rate.
   - pass_coverage: EPA/dropback allowed, CPOE allowed, yards/attempt allowed, explosive passes allowed, completion rate allowed by target position.
   - coverage_mix: man/zone rate, middle-closed rate. EXTENDED space only.
   - rb_archetype: carry share, yards/carry, explosive rate, rush EPA, target share, yards after contact (PFR), goal-line share, height and weight.
   - receiver_archetype: target share, average depth of target, air-yards share, yards after catch, catch rate over expected, slot rate, height and weight.
   - qb_archetype: EPA/dropback, CPOE, average depth of target, sack rate, scramble rate, rush attempts per game, play-action rate.
   After building the spec, PRINT the final column list for every unit with its source table and first season available, and stop so I can review it before 4c.1 starts.

3. SPACES. BASE = play-by-play features only, 2016 on. EXTENDED = BASE plus FTN (2022 on) plus PFR (2018 on). Two separate pools, never mixed in one nearest-neighbor search.

4. WINDOWS. last_3, last_6, season_to_date, recency_weighted, continuity_weighted. No window is assumed to win; 4c.6 selects one per unit and market.

5. MARKET -> UNITS (a search only uses the units listed for that market):
   - pass attempts, pass completions, pass yards: pass_offense, ol_protection, qb_archetype, pass_rush, pass_coverage, coverage_mix.
   - rush attempts, rush yards: run_offense, rb_rotation, rb_archetype, ol_protection, run_defense.
   - QB rush attempts, QB rush yards: run_offense, qb_archetype, ol_protection, run_defense.
   - targets, receptions, receiving yards: receiver_usage, receiver_archetype, pass_offense, pass_coverage, coverage_mix, pass_rush.
   - total, spread, moneyline: run_offense, pass_offense, run_defense, pass_rush, pass_coverage (team level only, no player archetypes).

6. THE FIVE SEARCHES. For a target (player or team, game, market):
   S1 own_history_vs_similar_defenses: past games of the target offense or player (same team, same player) against defenses whose lineup-adjusted defensive unit vectors resemble tonight's opponent.
   S2 opponent_history_vs_similar_offenses: past games of tonight's defense against offenses (and players) whose unit vectors resemble tonight's offense and the target player.
   S3 similar_players_vs_similar_defenses: past games of players whose archetype vector resembles the target player, on any team, against defenses resembling tonight's defense.
   S4 similar_offenses_vs_similar_defenses: past games in which an offense resembling tonight's (relevant units) faced a defense resembling tonight's (relevant units). Weight uses BOTH similarities multiplied.
   S5 similar_matchup_structure: past games matching tonight's interaction profile, using these interaction features: box count faced vs run rate, motion usage vs defense results against motion, offense explosive-play rate vs defense explosive-play rate allowed, run-direction share vs defense run-direction results (only where the sample is at least 30 plays), OL protection vs pass rush, play-action usage vs defense play-action results, screen usage vs defense screen results, blitz rate faced vs QB results against the blitz. EXTENDED space only. A missing interaction feature gets zero weight plus a completeness penalty. This search will often return no reliable comparable; that is expected, not a bug.

7. CONSTANTS (starting values; tunable ONLY on 2020-2024 walk-forward results): SIM_THRESHOLD = 0.70, MIN_NEFF = 2, QUALITY = {observed: 1.0, derived: 0.9, estimated: 0.65}, SHRINK_K = 5, TEAM_CAP_PER_SEARCH = 0.25, TEAM_CAP_AVG_ACROSS_SEARCHES = 0.15.
```

**4c.1 -- Multi-window fingerprint vectors**

```text
In models/comps.py, build the fingerprint vectors defined in models/comps_spec.py (4c.0). Do not decide what a component is; use the spec.

1. For every team-game, build one vector per UNIT and per WINDOW (last_3, last_6, season_to_date, recency_weighted using features/weights.py recency_weight, continuity_weighted using continuity_weight for the market). Do the same for player archetypes.
2. Standardize each feature as a z-score against the league for that season-week, using only data from before the target week. No future data anywhere.
3. Build BASE and EXTENDED spaces separately. A search uses EXTENDED only when every feature of that unit exists for both sides; otherwise BASE.
4. HEALTHY vs LINEUP-ADJUSTED. Every target vector (the player or team being predicted, and the opponent's units) is stored in two versions:
   a. healthy: built from the team's normal lineup.
   b. lineup-adjusted: rebuilt from the expected game-day lineup produced by the 4a injury layer. Each player's contribution to a unit is scaled by his expected snap share (4a.1 play probability, 4a.2 redistributed shares) and the replacements fill the rest.
   Every comparable search runs on the LINEUP-ADJUSTED target vector. Also run it on the healthy vector and store how the retrieved comparables changed (overlap of the top matches, change in the shift). Until 4a exists, adjusted = healthy and the log says so.
5. Historical pool vectors reflect who actually played that game (from snap counts), not the pre-game expectation.
6. Store every vector with its feature values, source table, observed/derived/estimated tag per feature, window, space, and the as-of week cutoff.
7. Tests: altering any stat, score or lineup from week W onward leaves every vector for earlier weeks unchanged; a unit with a missing source table returns missing, not zeros.
```

**4c.2 -- Quality-weighted distance and similarity**

```text
In models/comps.py, implement distance and similarity.

1. Per feature group g: d2_g = sum_f(w_f * q_f * (a_f - b_f)^2) / sum_f(w_f * q_f), over the features present on both sides. w_f is the feature weight (start equal). q_f is data quality: observed 1.0, derived 0.9, estimated 0.65. A feature missing on either side gets weight 0 and adds to a completeness penalty (reported, and used in 4c.5).
2. Collapse correlated features into groups BEFORE combining (run: rush EPA, success rate, yards/carry, explosive rate; pass: EPA/dropback, CPOE, yards/attempt, explosive rate; add any other strongly correlated groups you find and list them in the spec). The unit distance is the weighted mean of the group d2 values (start equal weights).
3. similarity = exp(-d2 / (2 * sigma^2)), using this same d2, NOT squared again. sigma is set per unit and per space as the median distance to each target's 20th nearest neighbor, computed walk-forward.
4. Distance is only computed between vectors of the same unit, window and space.
5. Tests: identical vectors give similarity 1.0; similarity falls as distance rises; changing a quality-0 or missing feature changes nothing; altering anything from week W onward leaves every similarity for earlier weeks unchanged.
```

**4c.3 -- The five searches and no-match rule**

```text
In models/comps.py, implement the five searches exactly as defined in models/comps_spec.py (S1 through S5). Do not reinterpret them. For a target (player or team, game, market), each search uses only the units the spec lists for that market, on the lineup-adjusted target vectors from 4c.1.

Each search returns, per matched historical observation: the match's similarity (per unit and combined), recency weight, continuity weight, data-quality weight, and final weight, as SEPARATE columns. It also returns n_eff, best similarity, and a no_match flag.

Pool rule: walk-forward only. The pool is every game before the target week, back to 2016 for BASE space and back to the first available season for EXTENDED space. Never include the target game or anything after it.

NO-MATCH RULE: if the best similarity is below SIM_THRESHOLD (starting 0.70) or n_eff is below MIN_NEFF (starting 2), the search returns no_match = True, shift = 0, and a widened-uncertainty flag. Never lower the threshold for one target to force a match.

Log, per season, how often each search returns no_match for each unit and market.

Tests: a target with a planted near-duplicate in the pool retrieves it first; a target with nothing above threshold returns no_match; S5 returns no_match whenever its required FTN features are missing.
```

**4c.4 -- Standardized residuals and shifts**

```text
In models/comps.py, turn each search's matches into standardized shifts.

1. EXPECTED VALUES are comp-free: use the walk-forward volume (3.1) and efficiency (3.2) models trained WITHOUT any comparable features. z = (actual - expected) / sigma, computed separately for volume and efficiency. sigma comes from prior walk-forward errors for that market and role, blended toward the player's own residual SD as his sample grows (start n/(n+10)).
2. WEIGHT of each historical observation h:
   one-sided searches (S1, S3): w = similarity * recency * continuity * data_quality.
   two-sided searches (S2, S4, S5): w = similarity_offense * similarity_defense * recency * continuity * data_quality.
   recency and continuity come from features/weights.py for that market. Keep all four factors as separate stored columns.
3. SAFEGUARDS against one historical team dominating: no single historical team-game may carry more than TEAM_CAP_PER_SEARCH of the total weight in one search, and no historical team may average more than TEAM_CAP_AVG_ACROSS_SEARCHES of the weight across the five searches. Redistribute excess weight pro rata and renormalize. Compute n_eff = (sum w)^2 / sum(w^2) AFTER capping.
4. SHIFT per search = sum(w * z) / sum(w), multiplied by n_eff / (n_eff + SHRINK_K), so a thin sample is pulled toward zero. One volume shift and one efficiency shift per search. A no_match search has shift 0.
5. DO NOT add the five shifts together and do not hand-weight them. Output one row per target and market with these columns: shift_vol_S1..S5, shift_eff_S1..S5, n_eff_S1..S5, nomatch_S1..S5, best_sim_S1..S5. They go to the volume and efficiency models as separate features, so the model learns which search matters for which market.
6. Store the full per-match table (similarity, recency, continuity, quality, final weight, z for volume and efficiency) so the Matchup screen can show it.
```

**4c.5 -- Input-share tracking**

```text
In models/comps.py, track input quality per search and feed it forward.

1. For every search and target, store the share of the final feature weight that was observed, derived and estimated as three SEPARATE columns (share_obs_S1..S5, share_der_S1..S5, share_est_S1..S5), plus the completeness penalty from 4c.2. Do not lump them.
2. Add these columns, together with the 4c.4 columns, as features to the volume (models/volume.py) and efficiency (models/efficiency.py) regressions. Same hyperparameters as 3.1 and 3.2, same walk-forward, 2025 never touched.
3. Expose the observed/derived/estimated shares to the confidence score's input-completeness component.
4. Mark every estimated value (for example a coverage shell inferred from tracking data) as estimated in storage. Never mix an observed value and an estimate in one untagged column.
```

**4c.6 -- Window selection and the comparables ablation**

```text
Write run_comps_ablation.py.

1. WINDOW SELECTION. For each unit and each market, choose among the five windows (last_3, last_6, season_to_date, recency_weighted, continuity_weighted) using ONLY 2020-2024 walk-forward error. No universal winning window is assumed. Save the chosen window per unit and market to a config file and print it.
2. ABLATION. For every market, run the Phase 3 model WITH and WITHOUT the comparable features through the phase 2 harness, 2020-2024 only. 2016-2019 data may fill the comparable pool but is never scored. 2025 is never used to decide anything.
3. For each market report: gain of with over without, 95% CI using the same season-week bootstrap as 3.3 (10,000 resamples, fixed seed), per-season gains, the no_match rate per search, and the average n_eff per search. One row per market. Do not average across markets.
4. ALSO report the gain by search: the model with only S1, only S2, and so on, so we can see which searches carry the lift.
5. Write results to comps_ablation_results.parquet. Running twice must give identical numbers.
```

Manual steps: print five real comparable searches as readable cards and read them like a human. Each card shows the target, then for every unit its top matches with similarity, recency, continuity, quality and final weight, the healthy-vs-lineup-adjusted change, which searches returned no reliable comparable, and the final volume and efficiency shifts. Ask yourself whether the closest games actually look similar. Include at least one case where a key defender is out (the adjusted vector should retrieve different comps than the healthy one) and one case that should return no reliable comparable.

Gate: validate with the ablation from 4c.6 on 2020-2024 walk-forward, never 2025. For each market, the comparable features stay in that market's model only if with beats without in more than one season separately, with an interval that excludes zero. If a market does not clear it, its comparable features are left out of that market's model for now and flagged. The market itself stays in the build and is re-tested after every later layer. Do not average across markets.

### 4d -- Coaching priors

**4d.0 -- Build the historical coaching table (prerequisite)**

```text
The coaches table from phase 1.7 only holds the 2026 staff, and 4d cannot work without each coach's prior stops. Create data/coaches_history.csv with one row per team-season-role for 2016 through 2026: season, team, role (head_coach, offensive_coordinator, play_caller), name. The play_caller role is whoever actually called the offensive plays that season, which can differ from the head coach or OC title.

Pre-fill it only from public sources whose terms allow collection. BEFORE collecting anything, list the sources you plan to use and what each one's terms say, and stop for my review. Mark any row you could not fill as UNKNOWN, never guess. Print the count of UNKNOWN rows per season. Then load it into DuckDB as coaches_history, with a test that every team-season has exactly one head_coach row and one play_caller row, or is flagged UNKNOWN.
```

**4d.1 -- Wire the coaching table into team ratings**

```text
In features/team_ratings.py, fill the coaching-prior hook from 4b.1 using coaches_history from 4d.0.

1. For a team whose play_caller or head coach is new this season (or new at any point in the data), build a prior from that coach's PRIOR stops, three tendencies only: pass rate over expected, pace (seconds per play in neutral game states), and fourth-down aggressiveness (go rate over expected). Compute each from play-by-play at his previous teams.
2. Weight his prior stops by recency with a half-life of 3 seasons, and by the length of the stop (a 1-game stop counts for almost nothing, using n/(n+k) with k = 17 games).
3. The prior feeds the 4b.1 starting values and the 4b.3 dropback-rate-by-state starting values for that team.
4. Blend toward current-season data as games accumulate with n/(n+k) (start k = 6 games). By week 8 the prior should have little weight.
5. A coach with no prior stop (first-time play caller) gets no prior, and the team starts from last season's rating alone. Mark him as no-prior in the output.
6. Tests: a team with no coaching change gets no prior; a prior never uses games from the coach's current team; walk-forward only.
```

Manual steps: check the prior against two or three known real coaching changes from the last few seasons (a team with a new play caller) and confirm it points the right way, for example a coach known for a pass-heavy offense raising the team's starting pass rate.

Gate: measurable improvement specifically in the first six weeks of a season and in the six weeks after a coaching change, judged on 2020-2024 only, since that is the only window this layer should matter in. Outside that window it should do nothing. No lift means the layer stays at zero weight and is flagged.

### 4e -- Weather

**4e.1 -- Wire weather into context adjustments**

```text
In models/game_state.py, wire the weather table from phase 1.6 into the context-adjustment step.

1. Inputs: wind speed, temperature, precipitation (use any precipitation column the weather table actually has; skip it if there is none, and note that precip_prob_pct is always null), dome vs open roof vs retractable roof, rest days, and short-week flag.
2. Domed and closed-roof games are set to a neutral indoor value, not treated as missing weather.
3. Learn from 2020-2024 history how each input moves two things: dropback rate and yards per attempt (relative to the comp-free expectation). Use a simple regression with the wind effect allowed to be nonlinear (flag at 15+ mph).
4. Apply the learned shifts inside the 4b simulation (dropback rate by state, passing efficiency) and as features in the Phase 3 volume and efficiency models.
5. A missing weather row flows through as a missing feature (zero weight plus a completeness penalty). It must never crash the pipeline.
6. Tests: a game with a missing weather row still runs; an indoor game gets no weather shift; no future data.
```

Manual steps: confirm the pipeline runs on a game with a missing weather row, and look at three outdoor games with real wind or cold to see the shifts point the right way (passing yards and dropback rate down in high wind).

Gate: lift only in outdoor games with real wind (15+ mph) or cold (32 degrees or below), judged on 2020-2024 only. If it is not measurable, leave it in at zero weight and say so plainly rather than forcing it.

Running rule for all of 4a-4e: after each layer, re-run the Phase 3 backtest (2020-2024 only, 2025 never used to decide anything) with that layer added and compare market by market against the Phase 3 model, with the 95% CI from the same season-week bootstrap as 3.3. If a layer does not add lift in more than one season for a given market, that layer is set to zero weight for that market and flagged. The market itself is never dropped from the build; it is re-tested after every later layer. A layer sitting in the model doing nothing is not neutral, it is a free parameter that can overfit later, so zero weight means removed from that market's feature set.

## Phase 5: live paper run each week

This is the first point where the engine predicts real, upcoming games instead of backtesting history. Nothing is bet or acted on here; this phase exists to prove the pipeline runs on a real weekly cadence and to start building a genuine prospective track record on 2026, which is the clean prospective holdout for every market.

**Build prompts for Cursor (one task each):**

**5.1 -- Weekly prediction run**

```text
Write run_weekly.py.

1. Pull fresh data for the upcoming week with the phase 1 scripts: injuries, depth charts, rosters, snap counts, schedules, and weather for each game. Each pull is timestamped.
2. ELIGIBLE PLAYERS: config.ELIGIBLE_PLAYER_RULE (see 0.2 and 3.5.2c). Predict EVERY eligible player in every market that applies to his position. Predict everyone, whether or not any line exists.
3. Generate predictions with the full model stack that currently exists (phases 3-4, each layer only if it has passed its gate and has non-zero weight). Store for each prediction: prediction_id, run_id, run_timestamp, kickoff_time, game_id, player_id, market, model_version, mean, median, quantiles (5, 10, 25, 50, 75, 90, 95), the threshold-probability ladder, the confidence components, every comparable shift and no_match flag, the injury status snapshot used, and a hash of the input data.
4. PREDICTIONS ARE STORED BEFORE KICKOFF, ALWAYS. The script must refuse to write a prediction for any game whose kickoff time has passed and must log that it refused.
5. A rerun on Sunday morning writes a NEW version (higher version number) for each prediction. It never overwrites or edits an earlier one. Grading uses the latest version stored before kickoff, and every version is kept.
6. Separately log the current schedules-table line (spread, moneyline, total) for anyone who wants a market comparison. This goes to its OWN table, never into the prediction record.
7. Tests: a game with a past kickoff writes nothing; a rerun creates version 2 and leaves version 1 untouched; the prediction table has no column that comes from a sportsbook line.
```

**5.2 -- Weekly grading run**

```text
Write grade_week.py.

1. After games finish, pull the actual results and grade the latest pre-kickoff version of every prediction for that week, appending to a running graded_predictions table. Grading the same week twice must not create duplicate rows (idempotent).
2. Per prediction record: actual value, error, absolute error, the probability the model gave to the actual outcome at each ladder threshold, Brier score for each threshold, log loss, and which quantile band the actual landed in.
3. A player who was eligible but did not play is graded with actual = 0 and a did_not_play flag. Report did-not-play rows separately in every summary so they do not hide inside the averages.
4. Keep the PURE track (model only) and the MARKET-AWARE track (anything that used or compared a line) in separate tables so they can never be mixed in reporting.
5. Write a weekly summary per market: number of predictions, MAE, Brier, calibration table, and the same numbers for the best Phase 2 baseline on the same rows.
6. Tests: grading twice gives identical output; a prediction version stored after kickoff is never graded.
```

**Manual steps:**

1. Run `run_weekly.py` every Friday (after the Friday injury report) and again Sunday morning for late inactives, per the injury-timing rule.
2. Run `grade_week.py` every Tuesday once the prior week's games are final.
3. Do this for several consecutive weeks before drawing any conclusions -- one week tells you almost nothing.

**Gate:** the weekly cycle runs without manual intervention beyond kicking off the two scripts, predictions are stored before kickoff every time (never after, which would be leakage), and you have at least three to four weeks of graded live predictions before moving to phase 6.

## Phase 6: meta-learning layer and calibration

This is the two-stage learning layer: a shrinkage blend that stays close to the baseline until the model proves itself, then a LightGBM stacking layer on top of the submodel outputs and comparable shifts. This only turns on for a market once it has several thousand graded player-games from the 2020-2024 walk-forward backtest -- it is not useful on a handful of early live weeks.

**Build prompts for Cursor (one task each):**

**6.1 -- Stage-1 shrinkage blend**

```text
In eval/meta.py, implement the stage-1 blend.

1. For each market, final = a * model + (1 - a) * baseline, where baseline is that market's best Phase 2 baseline and model is the best model currently in the stack.
2. Fit a per market on walk-forward predictions only (minimize MAE), using only games before the target week. Constrain a to between 0 and 1.
3. Shrink the fitted value toward 0 for thin samples: a_used = a_fit * n/(n + 500), where n is the number of graded player-games for that market so far (start k = 500, tuned ONLY on 2020-2024).
4. When the model has not proven itself, a stays low and the output stays close to the baseline. Report a per market per season.
5. Tests: a = 0 reproduces the baseline exactly, a = 1 reproduces the model exactly; walk-forward only.
```

**6.2 -- Stage-2 stacking model**

```text
In eval/meta.py, implement stage 2.

1. A LightGBM model per market trained on: the submodel outputs (volume prediction, efficiency prediction), every comparable shift and n_eff and no_match flag (kept as separate columns), the injury and weather and coaching layer outputs, and the confidence components. It predicts the market's actual outcome. Walk-forward only.
2. ACTIVATION THRESHOLD: stage 2 turns on for a market only when that market has at least 3,000 graded player-games in the 2020-2024 walk-forward backtest (starting value, in config). Below that, the market stays on stage 1 alone. Print the count per market and which markets cleared.
3. When active, compare stage 2 against stage 1 on 2020-2024: MAE, Brier, per-season gains and the 95% CI from the same season-week bootstrap as 3.3. Keep stage 2 for a market only if it beats stage 1 in more than one season with an interval that excludes zero.
4. Save SHAP values per market so the Player report screen and the manual review can read them.
5. Tests: no row uses information from after its own week; a market below the threshold never runs stage 2.
```

**6.3 -- Calibration: width fix and isotonic correction**

```text
In eval/calibrate.py, implement calibration in two steps.

1. WIDTH FIX: one scale factor per market applied to the predictive distribution's spread, chosen so the whole-curve check comes out even, meaning that nominal 50%, 80% and 90% central intervals contain the actual value about 50%, 80% and 90% of the time.
2. ISOTONIC CORRECTION on individual threshold probabilities, fit per market.
3. Choose BOTH the scale factor and whether isotonic correction helps using ONLY 2020-2024 walk-forward results. 2025 reports performance only and must never be used to pick the method. Live 2026 results must never retroactively change it.
4. Output reliability diagrams (predicted probability vs observed frequency) before and after correction for every market, saved as images.
5. Tests: applying calibration never uses data from after the prediction's own week; a market with too few graded rows is left uncalibrated and flagged.
```

**Manual steps:** review the stage-2 feature importances (SHAP) for each market it turns on for, and sanity check that the model is leaning on things that make football sense, not spurious correlations from a small sample.

**Gate:** calibration curves are visibly flatter after correction than before (plot it, don't just trust a number), and stage 2 only activates for markets that actually clear its graded-sample threshold.

## Phase 7: Streamlit app screens

This is the interface: Slate, Player report, Matchup, Lines log, Grading, Model health -- all six screens defined below.

**Build prompts for Cursor (one task each):**

**7.1 -- Slate screen**

```text
In app/streamlit_app.py, set the page title to Script the Slate, then build the Slate screen.

A week selector (default: the upcoming week). One row per game, sorted by kickoff time, with columns: kickoff, away, home, expected margin (4b.2) with its SD, expected total with its SD, expected plays for each team, and the number of stored predictions for that game. Clicking a game opens the Matchup screen for it. Read only from stored predictions and stored simulation results; the screen never recomputes a model.
```

**7.2 -- Player report screen**

```text
Add the Player report screen: the full record for one player in one game, chosen with a player and game selector.

Show, in this order: (1) the threshold ladder for each market (for example 60+, 70+, 80+, 90+, 100+ rushing yards, each with a probability); (2) the distribution chart for the selected market with the median and mean marked; (3) the confidence breakdown showing each component separately (sample size, similarity strength, input completeness, freshness, continuity, submodel agreement) and the combined score; (4) the SHAP factor contributions from stage 2 if it is active for that market, otherwise the stage 1 blend weight; (5) the prediction version and the time it was stored. Read only stored values.
```

**7.3 -- Matchup screen**

```text
Add the Matchup screen for one game.

For a selected player and market, show the five comparable searches (S1 through S5) as separate panels. Each panel lists its top matches with similarity, recency, continuity, quality and final weight as separate columns, the volume and efficiency shift, n_eff, and the share of observed, derived and estimated inputs. A search that returned no reliable comparable shows a clear No reliable comparable label, weight 0, and the reason (best similarity under threshold, or n_eff under the minimum). Also show the healthy vs lineup-adjusted comparison: which comps changed when injuries were applied. Read only stored results.
```

**7.4 -- Lines log screen**

```text
Add the Lines log screen: a form to type in a sportsbook line and odds for any player and market, tied to that player's stored prediction.

The form saves the typed line, odds, book name and timestamp to the separate lines table. The screen then computes and displays, for display only: break-even probability from the odds, the model probability at that line, the gap in percentage points, and a flag if the gap and the prediction's confidence both exceed thresholds set in config. This screen must NEVER write into or alter the stored prediction. Typing, refreshing and saving a line must leave the prediction record byte-identical; add a test that hashes the prediction table before and after.
```

**7.5 -- Grading screen**

```text
Add the Grading screen, reading only the graded_predictions table from phase 5.2.

For each market: a table of number of predictions, MAE, Brier score and the best Phase 2 baseline's numbers on the same rows; a calibration (reliability) chart; and a line chart of MAE and Brier by week. Include a filter for season and week, and show did-not-play rows as a separate line. Keep the pure track and market-aware track on separate tabs.
```

**7.6 -- Model health screen**

```text
Add the Model health screen, backed by a factor_performance table. If that table does not exist yet, first write eval/factor_performance.py to build it from the ablation result files (comps_ablation_results, injury_ablation_results, sim_backtest_results) and the live graded_predictions.

One row per signal group (usage baseline, injuries, game-state simulation, each comparable search S1 through S5, coaching priors, weather) and market, with: out-of-sample MAE improvement, Brier improvement, calibration improvement, per-season sign of the gain, and recommendation hit rate (only for rows where a line was typed into the Lines log). Flag a signal group in red when it stops earning its weight: no gain in the most recent season or the live weeks. Show the flag with the numbers behind it.
```

**Manual steps:** run it locally with `streamlit run app/streamlit_app.py` and click through every screen with a real week's data before trusting any of it.

**Gate:** all six screens load without errors against real stored predictions, and the Lines log screen visibly does not alter any stored prediction when you type a line in -- test this directly by typing a line, refreshing, and confirming the prediction number is unchanged.

## Ongoing: weekly operating rhythm once live

Once phases 0 through 7 are done, the engine runs on a weekly cycle, matching the schedule table below.

| When | What runs |
| --- | --- |
| Tuesday or Wednesday | Pull the new play-by-play, FTN, snap counts and player stats from last week. Grade all of last week's predictions. Run `data_audit.py` as a freshness check. |
| Wednesday / Friday / Sunday | Pull injuries, depth charts, rosters (the timestamped snapshots). Friday's pull feeds the main weekly prediction; Sunday's catches late inactives. |
| Friday | Run `run_weekly.py` to generate and store this week's predictions, before any game kicks off. |
| Sunday / Monday night | Games happen. Nothing runs here except, optionally, you typing lines into the Lines log for games you're personally curious about. |
| Preseason / midseason | Decide whether to ship a new version, per phase 6's gates -- backtested first, never shipped off a single week's result. |

No model change happens after a single game, win or lose. New versions ship on a schedule and are backtested before they replace the running one, and the old version keeps running in the background so you can compare them on the same live weeks. This is the same discipline the whole plan has been building toward: the build order exists so you find out cheaply, at each phase, whether the next layer is worth having -- and the weekly rhythm exists so that once it's running, you don't quietly undo that discipline out of excitement after one good week or panic after one bad one.

# Comparables engine, 4c.0 - 4c.1 (spec and fingerprint vectors)

Status: 4c.0 (spec) and 4c.1 (vectors) done and approved. 4c.2 (quality-weighted distance and similarity) not started.
Everything is walk-forward: a vector for a game uses only games before that week's first kickoff. 2025+ (the holdout) is never loaded.

## Files
| file | role |
|---|---|
| `models/comps_spec.py` | constants only: units, features (source, columns, space, first season, quality tag), markets -> units, windows, the five searches, tunable constants |
| `features/comps_ledger.py` | per team-game and player-game numerators / denominators for every spec feature (pbp, FTN, PFR, participation, snap counts, rosters, depth charts) |
| `models/comps.py` | windows, league z-scores, healthy / adjusted / actual versions, storage, `retrieval_change()` |
| `build_comps.py` | rebuilds everything from scratch; `python build_comps.py` |
| `data/processed/comp_vectors_{team,player}.parquet` | the vectors (large, regenerable, git-ignored) |
| `comp_features.parquet`, `comp_completeness.parquet`, `comp_lineup_change.parquet` | feature table (source, quality tag), source coverage per season, how far the lineup versions move the vectors |
| `tests/test_comps.py`, `tests/test_comps_spec.py` | no-leakage, missing-not-zero, windows, z-scores, lineup versions, determinism |

## Vectors
One row per entity-game, unit, window, space and version; `z`, `raw` and `n` are parallel lists in the feature order of `comp_features.parquet`
(key: unit, space, position), with `as_of` = the week's first kickoff. A missing feature is null, never 0.
* Windows: `last_3`, `last_6`, `season_to_date`, `recency_weighted` (`features/weights.py` recency weight, same clock as `games_elapsed`), and
  `continuity_weighted:<penalty row>` for the six penalty rows the markets use. Continuity weighting multiplies the recency weight.
* Value: pooled weighted rate `sum(w n) / sum(w d)`. z-score against the league for the same season-week and window, clipped at +-4.
  League mean / SD use windows whose denominator is at least half the week's median positive denominator (min 8 entities).
* Player rates are shrunk toward the league by `SHRINK_K` = 5 opportunities, except body measures and per-game features (`NO_SHRINK`).
* Spaces: BASE (2016+: pbp, snap counts, rosters) and EXTENDED (BASE + FTN 2022+ + PFR 2018+ + participation 2018+). A unit whose EXTENDED list equals
  its BASE list is stored once, as BASE. A search uses EXTENDED only when every EXTENDED feature of the unit is present on both sides.

## Lineup versions (team units)
* `healthy`: the team's own pooled value (normal lineup).
* `adjusted` (targets, 2020-24): healthy + sum over players of (expected share - baseline share) x (his shrunk window value - the team's), with the shares from
  the 4a layer (`exp_*` vs `b_*`). Searches read this one (`models.comps.target_version`).
* `actual` (historical pool games): the same correction with each player's normal usage share (trailing share of the team's plays over the team's games,
  absences = 0) restricted to the players who played (snap counts or a touch) and renormalised. Touches in the game are not used. `pool_version`.
* Units with a correction: run_offense, pass_offense (QB-driven features), receiver_usage, rb_rotation. Not lineup-adjusted (adjusted = healthy):
  ol_protection, the four defense units, and the scheme features (play action, screen, motion, no-huddle, rush rate over expected); the 4a layer
  models offensive skill players only. `models.comps.LINEUP_NOT_ADJUSTED` lists them.
* `retrieval_change(healthy_hits, adjusted_hits)` (top-k overlap, shared weight mass, outcome shift) is built and tested; 4c.2-4c.5 call it per search.

## Approved decisions
4c.0 (approved):
1. `coverage_mix` reads the `participation` table (an approved exception to the BASE / EXTENDED source list, scoped to coverage_mix only).
2. BASE includes snap counts and rosters (same 2016+ coverage as pbp). "Play-by-play only" meant full 2016+ coverage.
3. Explosive pass = 20+ yards; goal line = `yardline_100 <= 5`.

Quality tags (`config.QUALITY_WEIGHTS`: observed 1.0, derived 0.9, estimated 0.65): `coverage_mix` features are `estimated` and stay `estimated`.
The tag describes the kind of data, not how much of it exists: man/zone and coverage-shell labels are inferred from player tracking, not charted facts like
yards gained, so they are discounted by 0.65 wherever they are used whatever their coverage. Completeness is logged separately (`comp_completeness.parquet`)
and is handled by the missing-feature / completeness-penalty machinery. Measured on pass attempts, participation classifies man/zone on ~100% in 2018-24
and a coverage shell on 89-96%; nothing exists for 2016-17. (The "~38%" quoted at 4c.0 was a share of all plays, where the table leaves run plays blank.)

4c.1 (approved as implemented): continuity weighting multiplies the recency weight; player rates shrink toward the league by `SHRINK_K` = 5; league SD
needs half the median sample; z-scores clipped at +-4; usage-share corrections are over team games.

## Data fixes made while building
* pbp carries today's franchise codes (LV, LAC) for old Oakland and San Diego games; the other tables keep the code of the day. `comps_ledger.canon` folds
  OAK -> LV, SD -> LAC, STL -> LA (80 team-games had no plays before).
* A scramble has no passer id (the QB is the rusher): QB dropbacks are credited via `coalesce(passer, rusher)`. Player and team dropbacks agree exactly.
* Sort ties for players who changed teams made two builds differ; the usage-share ledger now has a total order.

## Known limits
* EXTENDED vectors are complete for only ~59.5% of pass_offense / pass_rush / run_defense games in 2020-24 (FTN starts in 2022); BASE is used for the rest.
* The pool (`actual`) and target (`adjusted`) corrections use different share methods (proportional redistribution among players who played vs the 4a.2
  redistribution), because 4a.2 expectations exist only for 2020-24 games.
* Window values of low-volume players are noisy even after shrinkage; sample sizes are stored (`n`) for the 4c.2 completeness penalty.

## Reproducibility
Two from-scratch builds (4a injury layer included) were byte-identical on all five output files. Tests: 227 passing.

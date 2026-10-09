# No-match calibration memo (analysis only; nothing was changed)

**What you need to decide:** whether the comparables keep SIM_THRESHOLD = 0.70 and MIN_NEFF = 2 (build plan 4c.0.7) or start from other
values. The plan allows these constants to be tuned only on 2020-2024 walk-forward results, never per target, and never to flip a verdict. I have
not tuned anything. 4c.4-4c.6 run on the plan's values until you decide.

Source: `comp_search_log.parquet` and `comp_nomatch_sensitivity.parquet` (committed). 33,485 backtest targets (2020-2024), five searches,
recency_weighted window, after the review fixes. Two runs were byte-identical. Every number below is walk-forward: a target in week K is compared
only with games before K.

## 1. What happens at the plan's values
No-match rate (share of targets whose search returns no reliable comparable), 2020-2024 pooled:

| family | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| game (spread, total, moneyline) | 1.000 | 1.000 | n/a | 1.000 | 0.997 |
| pass | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| qb_rush | 0.962 | 1.000 | 0.993 | 1.000 | 0.997 |
| receiving | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 |
| rush | 0.970 | 1.000 | 0.998 | 1.000 | 0.997 |

S5 is no_match in all of 2020-2021 because FTN charting starts in 2022. The plan expects that.

So at these values, every shift is 0 for practically every target. The 4c.6 ablation would compare the model with a column of zeros against the
model without it. It cannot clear the gate, so the comparable features would be left out of every market and flagged (markets kept).

## 2. Why: the similarity scale, not a bug
- sigma (per unit, space, window) is the median distance from a target to its 20th-nearest neighbour in the whole pool (other teams / players,
  earlier weeks). A past game at that typical 20th-neighbour distance therefore has similarity exp(-1/2) = 0.61 for that unit. The 0.70 threshold
  sits above what the 20th-best game in the whole league reaches.
- Each unit alone does find close games. In S3 and S4, which search the whole pool, the median target's best match per unit is 0.89-0.99.
- But a search combines units. A side's similarity is the mean over its units, and the two-sided searches multiply the offense side by the
  defense side. One past game must be close on every unit at once, and then on both sides. The median best combined similarity is S1 0.40, S2 0.19,
  S3 0.33, S4 0.26 and S5 0.58. The share of targets whose best combined similarity reaches 0.70 is S1 9.7%, S2 0.1%, S3 2.2%, S4 0.2% and S5
  11.6% (2022+ only).
- S1 (the target's own past games) and S2 (past games against tonight's defense) search small subsets. Their nearest game is usually much farther
  than the 20th-nearest in the whole pool, which sets sigma.
- MIN_NEFF = 2 binds next. For rushing S1, 3,564 of 6,210 targets fail on the threshold. Another 2,385 have a game or two above it, but n_eff is
  below 2. Only 188 match.

## 3. What the same similarities would give at other values
No-match rate, mean of the five season rates, per family. Numbers are recounted from the similarities already computed, so no constant was changed.

MIN_NEFF = 2 (the plan's value):

| family | S1 @0.5 | S1 @0.6 | S1 @0.7 | S2 @0.5 | S3 @0.5 | S3 @0.6 | S4 @0.4 | S4 @0.5 | S5 @0.4 | S5 @0.5 | S5 @0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| game | 0.993 | 1.000 | 1.000 | 1.000 | n/a | n/a | 0.979 | 0.999 | 0.576 | 0.769 | 0.949 |
| pass | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.623 | 0.805 | 0.965 |
| qb_rush | 0.763 | 0.890 | 0.966 | 1.000 | 0.875 | 0.962 | 0.923 | 0.990 | 0.608 | 0.793 | 0.963 |
| receiving | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.588 | 0.776 | 0.953 |
| rush | 0.774 | 0.893 | 0.970 | 1.000 | 0.896 | 0.981 | 0.980 | 0.997 | 0.590 | 0.778 | 0.954 |

MIN_NEFF = 1:

| family | S1 @0.4 | S1 @0.5 | S1 @0.6 | S1 @0.7 | S2 @0.4 | S3 @0.4 | S3 @0.5 | S4 @0.4 | S4 @0.5 | S5 @0.5 | S5 @0.6 | S5 @0.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| game | 0.336 | 0.708 | 0.921 | 0.983 | 0.894 | n/a | n/a | 0.827 | 0.953 | 0.568 | 0.764 | 0.935 |
| pass | 0.576 | 0.827 | 0.946 | 0.985 | 0.937 | 0.630 | 0.869 | 0.910 | 0.979 | 0.570 | 0.762 | 0.930 |
| qb_rush | 0.119 | 0.213 | 0.376 | 0.586 | 0.908 | 0.265 | 0.485 | 0.569 | 0.799 | 0.567 | 0.775 | 0.940 |
| receiving | 0.626 | 0.852 | 0.952 | 0.986 | 0.974 | 0.830 | 0.955 | 0.958 | 0.993 | 0.570 | 0.766 | 0.935 |
| rush | 0.138 | 0.235 | 0.379 | 0.586 | 0.952 | 0.201 | 0.447 | 0.721 | 0.908 | 0.572 | 0.769 | 0.935 |

The S5 numbers include 2020-2021, which never match (no FTN). Within 2022-2024, S5's rates are lower than shown.

The full table (every season, market, search, threshold 0.4/0.5/0.6/0.7 and MIN_NEFF 1/2/3) is `comp_nomatch_sensitivity.parquet`.

## 4. Options (your call; I have not chosen)
A. **Keep 0.70 / 2.** Clean and per plan. The comparables then add nothing measurable now; 4c.6 leaves them out of every market, flagged.
B. **Pick new global values on 2020-2024 walk-forward results.** Fix the rule before looking at any gain, for example "the lowest threshold at which
   the with-comps model's walk-forward error stops improving, one value for all markets", and run it once. The plan allows tuning on 2020-2024
   walk-forward results only.
C. **Change how similarity is scaled**, for example sigma from the 5th nearest neighbour instead of the 20th, or a per-side threshold for the two-sided
   searches (offense >= 0.70 and defense >= 0.70, instead of their product >= 0.70). Each of these changes the plan's definitions (4c.2.3, 4c.4.2),
   so it is yours to make.
D. Any mix of the above.

Whatever you choose, two things limit the shifts regardless, and both are already listed in docs/overnight_plan.md. First, the comp-free
expectations behind z exist only for scored 2020-2024 rows (decision 2), so a 2020 target's matches carry no z yet. Second, n_eff counts past
team-games (decision 5).

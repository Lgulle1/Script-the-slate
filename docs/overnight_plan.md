# Overnight plan (written while you sleep; updated as each step finishes)

You asked me to keep going one phase at a time and not to start a phase until the previous one is bug-proof. The spec is
`docs/build_plan.md` (committed from your upload). Every step is committed and pushed as it finishes. Standing rules: pure track only; 2025
never loaded; raw pulls append-only; results byte-identical on a second run before they are committed; older result versions never overwritten;
no market dropped; no threshold tuned to flip a verdict; no PRs.

## Order of work
1. **4c.3 finished and bug-proofed** (in progress): conformance with the plan, two from-scratch builds byte-identical, the search log run twice,
   an independent code review, every confirmed bug fixed test-first, the full suite.
2. **4c.4 standardized residuals and shifts**, then its own verification.
3. **4c.5 input-share tracking** (feature wiring into the volume and efficiency models, like 4a.4), then its own verification.
4. **4c.6 window selection and the comparables ablation** (`run_comps_ablation.py`), run twice, byte-identical, CLEARS gate applied per market.
5. **4d.0 stops for you**: the plan says to list the coaching-history sources and their terms and STOP for your review before collecting anything.
   If there is time I will prepare that list (no collection).
6. **4e.1 weather**: independent of 4d. I will only start it if 4c is finished and verified, and I'll say so here.

## What I changed in 4c.3 to match the plan (it was built before you sent the plan)
- S2 and S5 are two-sided: weight = offense similarity x defense similarity (plan 4c.4). In S2 the defense is tonight's opponent itself, compared
  with what it was in the past game. S5 compares the offense side and the defense side of the interaction profile separately.
- Continuity comes from `features/weights.py` for the market in every search, not only in S1. A past game of another team flags every factor,
  so it carries the product of the whole penalty row.
- S5 sigma is estimated over the matchups the S5 pool admits (every FTN feature present). Before, it also counted 2016-2021 rows.
- A game without 4a expectations (2 of 2,688) falls back to the healthy target vector instead of counting the units as missing.

## Things I need you to decide (I have not decided them)
1. **The no-match calibration.** At SIM_THRESHOLD = 0.70 almost every search returns no_match (numbers in the 4c.3 log). The plan lets the
   constants be tuned only on 2020-2024 walk-forward results, never per target. I am not tuning anything; 4c.4-4c.6 run on the starting values,
   and I'll add a sensitivity table so you can see what each option would do.
2. **Pool expectations for 2016-2019.** 4c.4's z needs a comp-free expected value for every matched observation. `walkforward_predictions`
   (3.5.3) covers only 2020-2024 scored rows. Extending it means moving FEATURE_HISTORY_START to 2016, which 3.5.2b reserves for a later decision.
   Until then, matches without an expectation are retrieved but carry no z, so the shift uses only matches that have one.
3. **S3 "one-sided".** Plan 4c.4 lists S3 as one-sided, but S3's definition names two resemblances (the player's archetype and the defense
   faced). I use archetype similarity x defense similarity as S3's single similarity; say if you meant otherwise.
4. **Side similarity over several units** (for example the five offense units of rush_att) is the mean of the unit similarities. The plan does
   not fix this.

## Log
- 4c.3: S2/S5 two-sided, continuity in every search, S5 sigma fix, fallback to healthy. 50 tests pass. Vectors rebuilt twice with the
  played-population fix: all six outputs byte-identical.

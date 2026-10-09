# Overnight plan (written while you sleep; updated as each step finishes)

You asked me to keep going one phase at a time and not to start a phase until the previous one is bug-proof. The spec is
`docs/build_plan.md` (committed from your upload). Every step is committed and pushed as it finishes. Standing rules: pure track only; 2025
never loaded; raw pulls append-only; results byte-identical on a second run before they are committed; older result versions never overwritten;
no market dropped; no threshold tuned to flip a verdict; no PRs.

## Order of work
1. **4c.3 finished and bug-proofed** (done): conformance with the plan, two from-scratch builds byte-identical, the search log run twice,
   an independent code review, every confirmed bug fixed test-first, the full suite.
2. **4c.4 standardized residuals and shifts** (verifying): its own independent review (two cap bugs fixed test-first), the full backtest run twice
   plus once more on the faster code, all three to be byte-identical.
3. **4c.5 input-share tracking** (feature wiring into the volume and efficiency models, like 4a.4), then its own verification. How I read it:
   - **Shares.** For each search and target: share_obs / share_der / share_est. This is the share of the feature weight that went into the matches'
     similarities and came from observed / derived / estimated features. A feature's weight is its effective weight in the unit distance:
     the group weight times w_f x q_f over the group's features present on both sides. A match takes the mean over its search's units. The
     search averages its matches by their final (capped) weight. completeness_S is the matches' completeness penalty, averaged the same way.
     All are null when the search has no match.
   - **Features.** The volume model of a quantity takes the comparable columns of its own market (pass_att, rush_att, targets, qb_rush_att; team
     plays take the game markets'). Each efficiency model takes those of its market (pass_cmp, pass_yds, rush_yds, rec, rec_yds, qb_rush_yds; points
     per play takes the game markets'). Same hyperparameters, same walk-forward harness, 2025 never touched. A row without comparable features
     (not a scored target) has them missing, never zero.
   - **Confidence score.** It does not exist yet (Phase 5). The shares are stored per target and search under fixed column names, ready for its
     input-completeness component.
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

## What the independent code review found in 4c.1-4c.3, and the fixes
An independent review (a separate agent, read-only) went through 4c.1-4c.3. Every confirmed finding was fixed test-first: a test that fails on the
old code, then the fix. I checked each new test by putting its bug back: it fails.
1. **Leak: a target who did not play had no vector.** Archetype vectors existed only for player-games in which the player played, so a scored target
   (2020-2024) who then sat out had no vector, and "has a vector" told the search he played. 515 targets had no offensive snap and 69 more played at
   another position. Fix: every scored target gets a vector built from his earlier games, whether or not he played; the row is flagged `is_target`,
   and only rows flagged `in_pool` (he played) are ever observations of later searches.
2. **Leak: the league mean and SD of week W used the players who played in week W.** The archetype z-scores of week W were standardised over the
   players who turned out to play that week. Fix: the week's mean / SD and the shrinkage mean come from that week's PREGAME depth chart (every team-week
   2016-2024 has one). Test: removing a player's week-W game changes no other player's week-W vector.
3. **S5 player markets counted one past game several times.** S5 compares team-game matchups, so every receiver of a past game had the same S5
   similarity and n_eff counted the game once per receiver. Fix: n_eff counts historical team-games (the cap of 4c.4 is per team-game too). Team
   targets are unchanged (one observation per team-game). This is the one fix that changes a plan formula's reading; see decision 5 below.
4. **RB2 carry share was missing, not 0, in bell-cow games** (one back took every carry). Fix: 0 over the RB carries.
5. Smaller: the EXTENDED space was chosen even when its sigma did not exist yet (now BASE, as for team units); the median best similarity in the
   no-match log sorted missing values above every value; the matches table had no tie-break on the player id; S3 and S4 excluded the target's own
   games and tonight's defense, which the plan does not say (S3 says "on any team"), so the exclusions are gone.
6. Not run before: the plan's healthy-vs-adjusted comparison (4c.1.4). It now runs for every target and search in 4c.4 and is stored: overlap of the
   top 10 matches, shared weight, whether no_match flips, change in the volume and efficiency shifts.

## What the independent review of 4c.4 found, and the fixes
A second independent review covered 4c.4. It found no leakage: z, sigma, weights and caps of a week-K target use only earlier weeks, checked by a
brute-force recomputation. It did find two real problems in the team caps. Both were fixed test-first, and each test fails on the old code.
1. **A search that ended no_match still entered the across-search team cap, and the cap could zero a team out.** Example: S2 has three past games,
   but only one has an expectation, so S2 is no_match. Its team still counted toward the 15% average, and S4's shift dropped from 0.4 to 1e-7.
   Fix:
   - Only searches that match on their own (n_eff over the matches with an expectation >= MIN_NEFF after the per-game cap) enter the
     across-search cap, and the check repeats after it.
   - An over-cap team is scaled by one common factor, and only where other teams can take its weight, just enough to reach 15%.
   - When its share that no other team can take already breaks the cap, the team is left as it is and logged (`team_cap_frozen`), instead of being
     pushed out of every other search.
2. **With fewer than 4 past games carrying weight, the 25% per-game cap forced an equal split.** For weights 0.98 / 0.01 / 0.01 that claimed an
   n_eff of 3, and at exactly two games rounding decided MIN_NEFF. Fix: when the cap cannot hold, the weights are left as they are and n_eff
   decides. n_eff is compared with MIN_NEFF with a 1e-9 tolerance; no committed search-log row changes.
3. A thin efficiency side is now visible: new columns `nomatch_eff_S1..S5` and `n_eff_eff_S1..S5`. `n_eff_S*` reports the actual n_eff after the
   caps (0 without matches).
4. The runner now asserts that no target or match comes from 2025 before it prints "holdout untouched".

## What the independent review of 4c.5 found, and the fixes
No leakage. A brute-force check of the share math over 1,119 random pairs (missing, quality-0 and weight-0 features, group weights) matched to
2e-16. Phase 3 results are untouched when no comparable features are passed. Fixed test-first:
1. S2 and S4 never match at 0.70, so their share columns were null in every row. They would have reached LightGBM as `object` columns and crashed
   the first with-comps fit. Every comparable column is now a float.
2. Requested comparable columns could silently be missing. A file written before 4c.5, or a single search passed as a string, would have run the
   "with" model without them. Both are now errors.
3. The plan asks for the shares "for every search and target". They were empty without a match, which at 0.70 means about 99% of rows. A search
   without a match now reports them from its 10 closest past games, weighted by final weight; `nomatch_S*` says which applies.
   S5 without its FTN features is now computed, then forced to no_match, so its completeness penalty is reported. The committed search-log rows are
   unchanged (checked on 7,500 rows).
4. A match's shares now mirror how its similarity is formed: the mean over the similarity's factors (offense side and defense side for S2 / S4),
   each the mean of its units.

## Things I need you to decide (I have not decided them)
0. **The lineup-adjusted target vectors: see `docs/lineup_adjustment_memo.md`.** A comparable card exposed a double count. A player out for weeks is
   already missing from the healthy window, and the 4a baseline subtracts him again. Measured on 2020-2024, every variant I tried describes the coming
   game worse than the unadjusted (healthy) vector does. Calibration slopes are 0.02-0.27. Defense units have no lineup adjustment at all. The memo
   lists the options. Nothing was changed; the searches still run on the vectors as built.
1. **The no-match calibration: see `docs/nomatch_memo.md`.** At SIM_THRESHOLD = 0.70 and MIN_NEFF = 2, practically every search returns no_match
   (S1 rushing 97%, everything else 99.3-100%). That follows from how similarity is scaled, not from a bug. The memo explains why and gives the
   no-match rate the same similarities would give at thresholds 0.4-0.7 and MIN_NEFF 1-3. I am not tuning anything: 4c.4-4c.6 run on the plan's
   values until you decide.
2. **Pool expectations for 2016-2019.** 4c.4's z needs a comp-free expected value for every matched observation. `walkforward_predictions`
   (3.5.3) covers only 2020-2024 scored rows. Extending it means moving FEATURE_HISTORY_START to 2016, which 3.5.2b reserves for a later decision.
   Until then, matches without an expectation are retrieved but carry no z, so the shift uses only matches that have one.
3. **S3 "one-sided".** Plan 4c.4 lists S3 as one-sided, but S3's definition names two resemblances (the player's archetype and the defense
   faced). I use archetype similarity x defense similarity as S3's single similarity; say if you meant otherwise.
4. **Side similarity over several units** (for example the five offense units of rush_att) is the mean of the unit similarities. The plan does
   not fix this.
5. **n_eff over historical team-games.** The plan's n_eff = (sum w)^2 / sum(w^2) does not say what one observation is. For a player market several
   players of one past game can match (always in S5, where they share one matchup similarity). I sum the weights of each past team-game before the
   formula, so one game is one observation's worth of evidence; this drives both the no-match rule (MIN_NEFF = 2) and the shrinkage n/(n+5). Counting
   player-games instead is a one-line change (`cluster_neff`) if you prefer it.
6. **S1 and the across-search team cap.** Plan 4c.4.3 caps any historical team at a 15% average share across the five searches. S1 is the target's
   own past games, so its own team holds 100% of S1 by definition: with S1 counted, the cap can never hold (that team averages at least 20%), and the
   capping loop pushed that team's weight toward zero in S2-S5. I exempt S1 from the across-search average (it keeps the 25% per-team-game cap) and
   still divide by five. Say if you read the rule differently.
7. **Expectations exist only for scored rows.** The comp-free expectations (3.5.3) exist for the scored 2020-2024 player-games and team sides. A
   matched past game of a player who was not a scored target that day (for example a WR4) has no z and does not move the shift. This is the same
   limit as decision 2, for non-scored players.
8. **Efficiency z ignores how many carries / targets / attempts the past game had.** sigma is per quantity and role (plan 4c.4.1), so a 1-catch
   game's yards-per-reception z spreads about twice as wide as a 10-catch game's (sd 1.32 vs 0.70; largest |z| 13). Every z enters the shift with
   its match weight, while the Phase 3 efficiency model weights rows by that count, so one 1-catch game can drive a thin search. Two possible fixes:
   scale sigma by sqrt(typical count / count), or weight efficiency matches by the count. Both change the plan's formula, so I left it.
9. **"Historical team" in the across-search cap is the past game's offense** (the observation's team). Capping by the past defense would collide
   with S2 the way S1 collides with the offense cap (S2's past games all face tonight's defense).
10. **When a cap cannot hold** (fewer than 4 past games carry weight in a search, or a team's share that nobody else can take already breaks the
    15% average), the weights are left as they are and logged, never forced. See the 4c.4 review fixes above.

## Log
- 4c.3: S2/S5 two-sided, continuity in every search, S5 sigma fix, fallback to healthy. 50 tests pass. Vectors rebuilt twice with the
  played-population fix: all six outputs byte-identical.

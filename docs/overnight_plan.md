# Overnight plan (written while you sleep; updated as each step finishes)

You asked me to keep going one phase at a time and not to start a phase until the previous one is bug-proof. The spec is
`docs/build_plan.md` (committed from your upload). Every step is committed and pushed as it finishes. Standing rules: pure track only; 2025
never loaded; raw pulls append-only; results byte-identical on a second run before they are committed; older result versions never overwritten;
no market dropped; no threshold tuned to flip a verdict; no PRs.

## Order of work
1. **4c.3 finished and bug-proofed** (done): conformance with the plan, two from-scratch builds byte-identical, the search log run twice,
   an independent code review, every confirmed bug fixed test-first, the full suite.
2. **4c.4 standardized residuals and shifts** (done, see the log).
3. **4c.5 input-share tracking** (done, see the log). How I read it:
   - **Shares.** For each search and target: share_obs / share_der / share_est. This is the share of the feature weight that went into the matches'
     similarities and came from observed / derived / estimated features. A feature's weight is its effective weight in the unit distance:
     the group weight times w_f x q_f over the group's features present on both sides. A match takes the mean over its search's units. The
     search averages its matches by their final (capped) weight. completeness_S is the matches' completeness penalty, averaged the same way.
     A search without a match reports them from its 10 closest past games (see the 4c.5 review fixes).
   - **Features.** The volume model of a quantity takes the comparable columns of its own market (pass_att, rush_att, targets, qb_rush_att; team
     plays take the game markets'). Each efficiency model takes those of its market (pass_cmp, pass_yds, rush_yds, rec, rec_yds, qb_rush_yds; points
     per play takes the game markets'). Same hyperparameters, same walk-forward harness, 2025 never touched. A row without comparable features
     (not a scored target) has them missing, never zero.
   - **Confidence score.** It does not exist yet (Phase 5). The shares are stored per target and search under fixed column names, ready for its
     input-completeness component.
4. **4c.6 window selection and the comparables ablation** (`run_comps_ablation.py`), run twice, byte-identical, CLEARS gate applied per market.
   How I read it:
   - **Window per unit and market** (`run_comp_window_selection.py`, `models/comps_windows.py`). For each 2020-2024 target and unit, the 20
     past observations most similar on that unit alone predict the target's comp-free residual z by their similarity-weighted mean z. The window
     with the lowest error (volume z, plus efficiency z where the market has one) wins. Every target that any window scores counts for every
     window; a window that cannot score it is charged z^2, as a no-match shift of 0 would be. A market's continuity window is the variant of its
     own penalty row. No threshold is involved, so the choice does not depend on the no-match
     decision. The choice is written to `comp_windows.json`, and the searches then read each unit in its own window.
   - **Ablation**: the Phase 3 model with vs. without the comparable columns, and with one search's columns at a time. Per market it reports
     the 3.3 bootstrap interval, per-season gains, and the no-match rate and n_eff per search. Layer weight 1 only on CLEARS, otherwise 0 and
     flagged. The window (or "selected") is in the result file name, so no result overwrites another.
   - Order: the ablation on the recency_weighted window (running), the window selection (running), then the comparable table and the ablation
     on the selected windows.
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

## What the independent review of 4c.6 found, and the fixes
No leakage in the window selection: neighbours come strictly from earlier weeks, and the target's own z is used only as the truth. Fixed test-first:
1. **The window choice dropped spread, moneyline and total.** Team targets have player_id null, and the join that compared windows did not match
   nulls. The selected-window run would then have failed, since `comp_windows.json` had no game markets.
2. **Windows were compared only on targets every window could score.** That hid season_to_date's missing week-1 vectors. A window is now charged
   z^2 for every target it cannot score, and the summary reports how many those are.
3. Searches no longer fall back silently to the base window. An incomplete or unknown window choice is an error, and the wrapper survives copy.
4. The ablation report crashed while printing when a search never matched (its mean n_eff was null for every market). The results had already been
   saved byte-identical.
Also added, from the review's soundness notes:
- **A null control.** LightGBM samples 80% of the columns per tree, so adding any columns moves the predictions a little. The ablation now also
  runs the model with the comparable columns shuffled within market and week, and reports that "gain" next to the real one. It is reported only;
  the gate stays as fixed.
- **Which run decides the gate, fixed before seeing it:** the run on the windows chosen per unit and market. The recency_weighted run is a
  reference. The selected run records the window mapping it used.

## Your decisions of 2026-10-09 (written down before any code change and before any result they affect)
- **Lineup adjustment (decision 0): the searches run on the healthy target vectors.** The lineup-adjusted vectors double-count a player who is
  already missing from the window. Injuries reach the model separately through the 4a injury features. The adjusted-vector search is still run
  and stored as the comparison (4c.1.4), with the roles swapped: healthy is the search, adjusted the comparison.
- **Similarity scale for the no-match check (part of decision 1).** A search whose similarity multiplies two factors (S2, S4, S5: offense x defense;
  S3: archetype x defense faced) is checked on the geometric mean, sqrt(factor_1 x factor_2). This puts every search on the per-factor scale of S1,
  so one threshold means the same thing for every search. A match also needs its weaker factor at or above SIDE_FLOOR = 0.40, so a 0.95 / 0.30 pair
  cannot pass on the average. The match WEIGHT stays the product (plan 4c.4.2): a 0.63 / 0.63 pair passes the check but weighs less than a 0.90 / 0.90
  pair. best_sim_S* reports the checked (geometric-mean) similarity. You named S2, S4 and S5. I also apply it to S3, because S3's similarity
  is also a product of two factors (decision 3); say if you want S3 left on the product.
- **Efficiency sample size (decision 8): efficiency matches are weighted by their count.** An efficiency match's weight is multiplied by the
  denominator of its ratio in that past game: attempts (completion rate), completions (yards per completion), carries (yards per carry, QB yards
  per carry), targets (catch rate), receptions (yards per reception), plays (points per play). The Phase 3 efficiency model weights its rows the same way. The count enters before the
  caps, so the caps and n_eff of the efficiency side use the count-weighted weights: a search dominated by thin games has a smaller efficiency
  n_eff. The volume side is unchanged. The window selection's efficiency side uses the same count weights.
- **Decisions 2-7, 9 and 10: accepted as written.**
- **The threshold sweep (rule fixed before any error is computed).**
  - **Setup.** SIM_THRESHOLD in {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}, with MIN_NEFF = 2 and everything above. Window recency_weighted (the
    base window; the window selection is re-run after the threshold is chosen). 2020-2024 targets only; 2025 stays locked.
  - **The rows.** For every threshold, every target, search and quantity (volume, and efficiency where the market has one; markets that share a
    quantity count once), where the search matches on that side at that threshold.
  - **Real error.** (shift - z)^2: the search's shift against the target's own comp-free standardized residual z.
  - **Random-pairing error.** Each match is replaced by a past observation drawn at random from the same search's observation set as of the same
    target week, among those with an expectation on that side: the past games the search could have picked, ignoring similarity. The real
    weights are kept, so n_eff and the shrinkage are identical, and only the z's change. It is averaged over 10 draws with a fixed seed.
  - **Improvement.** The random-pairing error minus the real error, pooled over all searches, markets and sides, with the 95% season-week
    cluster bootstrap (the 3.3 settings: 10,000 resamples, fixed seed).
  - **Choice.** The lowest threshold whose pooled improvement is above zero with the interval's lower bound above zero. If no threshold passes,
    SIM_THRESHOLD stays 0.70, and that is the finding.
  - **Reporting.** Per search, market and side: the improvement and the match rate, for reading only. Two runs byte-identical.
  - **Then:** set SIM_THRESHOLD, re-run the window selection, rebuild the comparable table and run the ablation gate on the chosen windows.

## Per-unit searches (your decision of 2026-10-09, fixed here before anything is built or run)
**Why.** The searches as built average four to six units into one similarity per side, so a past game counts only if it resembles tonight on all of
them at once. At 0.40, S3 and S4 typically find one or two such games (n_eff 1.3-1.5, below MIN_NEFF 2), S1 has about 54 past games to choose from,
and S2 fails the 0.40 floor in two targets out of three. The spec itself says never to collapse the units ("one part of a team can resemble one
historical team while another part resembles a completely different one"). The searches are rebuilt per unit. The threshold sweep that is running
(all units at once) is kept as the baseline of the old design.

**The searches.** For every target and market, over the market's units only (comps_spec MARKET_UNITS):
- **S1** (the target's own past games): one search per DEFENSE unit. Similarity: tonight's opponent vs the defense faced back then, on that unit.
- **S2** (past games against tonight's defense): one search per OFFENSE unit, including the player's archetype. Similarity: tonight's offense (or
  player) vs the offense (or player) back then, on that unit.
- **S3** (similar players, any team; player markets): one search per pair (the player's archetype, a defense unit of the market), checked on the
  geometric mean of the two similarities with the 0.40 floor; weight the product.
- **S4** (all past games): one search per matched pair. The pair list is fixed now and will not grow after results: run_offense x run_defense,
  pass_offense x pass_coverage, ol_protection x pass_rush. A pair is used only where both of its units are in the market's units. That gives the
  game markets the run and pass pairs; pass the pass and OL pairs; rushing and QB rushing the run pair; receiving the pass pair. Each pair is checked on
  the geometric mean with the 0.40 floor, and weighted by the product.
- **S5**: unchanged (one matchup profile per side).
- Units no pair or search above names (rb_rotation, receiver_usage, coverage_mix and the archetypes outside S2 / S3) take part only where listed above.

**Every unit search** has its own similarity scale (the unit's sigma, as now), the threshold check, MIN_NEFF = 2 over historical team-games, weight =
similarity x recency x continuity x quality, the 25% per-team-game cap, and the count weights on the efficiency side. A unit search without a match
weighs 0 and is flagged. The 15% across-search team cap is taken over the target's unit searches (S1's exempt, as now), dividing by their number.

**Roll-up to the model (the 4c.4 columns are kept).** Per search, shift_vol_S is the n_eff-weighted mean of its matching unit searches' volume
shifts, and shift_eff_S the n_eff_eff-weighted mean of their efficiency shifts (n_eff counts team-games, so the count weights are in it).
No-match units weigh 0. nomatch_S is true when no unit search of S matches. n_eff_S is the largest unit n_eff, and best_sim_S the largest unit
best similarity. Shares and completeness use the same n_eff weights. The per-unit shifts, n_eff and flags are stored for reading; the model does not see them.

**Threshold.** It is re-chosen on the per-unit searches with the rule already fixed above, unchanged: the lowest of 0.40-0.70 whose pooled
improvement over random pairing (now per unit search, from the same unit search's observation set) has a season-week interval above 0, else
0.70. Then the window selection is re-run (it is already per unit), the table rebuilt, and the gate run.

**Unit diagnostic (reported, not used to drop units in this round).**
- **Cells.** One per (search, unit or pair, side, quantity), over all 2020-2024 targets where that unit search matches. Each cell gets two
  improvements: over a zero shift (the comp-free model alone, z^2 - (shift - z)^2) and over random pairing (as in the sweep).
- **Works.** A cell works when both improvements are above zero, each 95% season-week interval excludes zero after a Bonferroni correction for the
  number of cells m (interval level 1 - 0.05 / m, m counted before any result), and each improvement is positive in at least 2 seasons. EDGE: positive
  but failing a condition. NO: otherwise.
- **Report.** Per cell, the match rate, both improvements and intervals, the seasons won and the label.
- **Diagnostic only.** Dropping units on this test and then judging the gate on the same 2020-2024 data would be choosing on the test. Any unit is
  removed only later, by a rule written first.

**Gate (unchanged).** The Phase 3 model with vs without the comparable columns, 2020-2024 walk-forward, CLEARS per market (gain > 0, interval
excluding 0, at least 2 seasons won), layer weight 1 on CLEARS else 0 and flagged. The null control is reported. The per-unit diagnostic is added to
the gate report. 2025 stays locked.

## Threshold rule tightened (your decision of 2026-10-09)
Tightened after the all-units baseline result, and BEFORE anyone read the per-unit sweep: one copy had finished and had not been opened; the second
was still running. The baseline showed comps beating random pairing at every threshold while losing to a zero shift at every threshold, which
the rule as first written would have passed.
- **New rule.** The threshold is the lowest of 0.40-0.70 whose pooled comps beat BOTH random pairing (random error - real error) AND a zero shift
  (z^2 - real error), each pooled improvement above zero with its 95% season-week cluster bootstrap interval's lower bound above zero.
  Everything else in the sweep is unchanged.
- **If no threshold passes.** SIM_THRESHOLD stays 0.70 and the comparable layer gets weight 0 for V1. There is no re-tuning afterward. The
  team-level test below and the 2016 extension are the only further steps, in the order below.

## Team-level sanity test (your decision of 2026-10-09; fixed before it runs)
**Why.** The comps score each past game against its full comp-free expectation, which already removed the "this defense is tough" signal on
both sides. So the comps can only show lift on the finer style interaction. This test asks the plain question instead: do similar matchups
predict a team's output, measured against the offense's own normal output, without the base model?

**The test.**
- **Markets and data.** Team rushing yards and team pass attempts per team-game. Targets are every 2020-2024 team-game, walk-forward (only
  games before the target week); past games go back to 2016. 2025 stays locked.
- **(a) The offense's own recent average.** The recency-weighted mean (features/weights.py recency weights, the comps' half-life) of the
  team's own earlier games, across seasons.
- **(b) That average plus a matchup adjustment.** The adjustment is the similarity-weighted mean, over the matched past team-games, of (that
  game's outcome - that offense's own recent average before that game), shrunk by n_eff / (n_eff + 5). Matched past games come from the
  per-unit S4 pair of the market: run_offense x run_defense for rushing yards, pass_offense x pass_coverage for pass attempts. The pair is
  checked on the geometric mean with the 0.40 floor at SIM_THRESHOLD 0.50, fixed now, with MIN_NEFF 2, recency x continuity x quality
  weights, the 25% per-team-game cap and the healthy vectors on the recency_weighted window. Without a match, (b) = (a).
- **(b-random).** The same adjustment with the matched past games replaced by past team-games drawn at random from the same pool before the
  target week, keeping the real weights (so n_eff and the shrinkage are identical); 10 draws, fixed seed.
- **Loss and rows.** Absolute error. The primary rows are the team-games where (b) has a match; all team-games are also reported.
- **Pass, per market.** (b) beats (a), with the season-week bootstrap lower bound above zero, AND (b) beats (b-random) the same way. Each
  market is judged on its own.

**What each outcome means.**
- **(b) passes in a market:** similarity carries real signal, and the base model's opponent-allowed features already cover it. The next
  question is whether comps should feed or replace those features (a separate decision, written first).
- **(b) loses even here:** the matching itself is broken (unit features or the similarity scale) and is fixed before anything else.

**Order.** Per-unit sweep, then this test, then (if the signal is real) extending the comp-free predictions to 2016 (your approval). If nothing
shows after that, the comps ship at weight 0 for V1, and the work moves to 4d.

## Fingerprint diagnostic (your decision of 2026-10-09; fixed before it runs; time-boxed)
**Question.** Does each unit's fingerprint carry signal about its own unit's next game? If units that look alike do not behave alike, similarity
on them cannot work. This is not a way to rescue the layer; it finds out whether its inputs carry signal at all.

**The test.**
- **Units.** The nine team units: run_offense, pass_offense, rb_rotation, ol_protection and receiver_usage (offense), and run_defense, pass_rush,
  pass_coverage and coverage_mix (defense). The three player archetypes are not part of this round.
- **Outcome per unit.** Its headline feature, the first one comps_spec lists for the unit, computed on that single game from the comps ledger:
  - run_offense rush EPA per play; pass_offense EPA per dropback; rb_rotation RB1 share of RB carries; ol_protection sack rate allowed;
    receiver_usage top-3 target share;
  - run_defense rush EPA allowed; pass_rush sack rate; pass_coverage EPA per dropback allowed; coverage_mix man rate.
- **Targets and data.** Every 2020-2024 team-game with the unit's vector and outcome: the offense's row for offense units, the defense's row
  for defense units. Walk-forward: only earlier weeks are used. 2025 is locked.
- **(a) The unit's own trailing average.** The recency-weighted mean of the same outcome in the team's earlier games (the comps' half-life,
  across seasons).
- **(k) The fingerprint neighbours.** The 20 earlier team-games most similar to the target on that unit alone: recency_weighted window, the
  unit's sigma, BASE space (EXTENDED for coverage_mix), the healthy target vector against the pool vectors, as in the searches. The prediction
  is the similarity-weighted mean of the neighbours' own single-game outcomes.
- **(r) Random neighbours.** 20 earlier team-games drawn at random from the same pool with (k)'s weights; 10 draws, fixed seed. Reported only.
- **Loss.** Absolute error, in the outcome's own units.
- **Pass, per unit.** (k) beats (a): mean gain above zero with the season-week bootstrap lower bound above zero, at the Bonferroni level 1 - 0.05 / 9.

**What each outcome means.**
- **A unit passes:** its fingerprint carries signal, and the matching has something to work with on that unit.
- **Every unit fails:** the unit features are the problem. Rebuilding them is a large project, and it does not start without your decision.

**Stopping rule.** If the diagnostic fails AND the per-unit sweep has no passing threshold, the comps are parked at weight 0 for V1 (kept built
and tested) and the work moves to 4d.

## Things I need you to decide (I have not decided them)
0. **(Decided 2026-10-09: healthy vectors, see above.)** **The lineup-adjusted target vectors: see `docs/lineup_adjustment_memo.md`.** A comparable card exposed a double count. A player out for weeks is
   already missing from the healthy window, and the 4a baseline subtracts him again. Measured on 2020-2024, every variant I tried describes the coming
   game worse than the unadjusted (healthy) vector does. Calibration slopes are 0.02-0.27. Defense units have no lineup adjustment at all. The memo
   lists the options. Nothing was changed; the searches still run on the vectors as built.
1. **(Decided 2026-10-09: geometric-mean check + threshold sweep, see above.)** **The no-match calibration: see `docs/nomatch_memo.md`.** At SIM_THRESHOLD = 0.70 and MIN_NEFF = 2, practically every search returns no_match
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
8. **(Decided 2026-10-09: weight by count, see above.)** **Efficiency z ignores how many carries / targets / attempts the past game had.** sigma is per quantity and role (plan 4c.4.1), so a 1-catch
   game's yards-per-reception z spreads about twice as wide as a 10-catch game's (sd 1.32 vs 0.70; largest |z| 13). Every z enters the shift with
   its match weight, while the Phase 3 efficiency model weights rows by that count, so one 1-catch game can drive a thin search. Two possible fixes:
   scale sigma by sqrt(typical count / count), or weight efficiency matches by the count. Both change the plan's formula, so I left it.
9. **"Historical team" in the across-search cap is the past game's offense** (the observation's team). Capping by the past defense would collide
   with S2 the way S1 collides with the offense cap (S2's past games all face tonight's defense).
10. **When a cap cannot hold** (fewer than 4 past games carry weight in a search, or a team's share that nobody else can take already breaks the
    15% average), the weights are left as they are and logged, never forced. See the 4c.4 review fixes above.

## Log
- 2026-10-09, team-level sanity test (as fixed above). Two runs byte-identical (comp_team_test.parquet 6832298f70d52720, rows d7f281baea6cd198).
  FAILS in both markets: (b) is worse than the offense's own average, with intervals fully below zero.
  - Team rushing yards: matched team-games 78%, MAE (a) 39.11 vs (b) 39.75, -0.65 yards [-1.07, -0.23]; vs random +0.41 [-0.02, +0.83].
  - Team pass attempts: matched 67%, MAE 6.43 vs 6.53, -0.10 [-0.19, -0.02]; vs random +0.05 [-0.04, +0.13].
  - **Fixed reading:** the matching is broken and is fixed first.
  - **Exploratory, not a gate.** The comps adjustment has practically no correlation with the actual surprise (rushing 0.004, pass attempts
    0.03; the slope's sign flips across seasons), so it is not the right signal at the wrong size. A plain adjustment by what tonight's
    defense allowed recently (vs the league, full weight, no similarity) ALSO loses to (a): -0.76 [-1.43, -0.06] rushing yards and -0.19
    [-0.30, -0.08] pass attempts. A single team-game's surprise is large (rushing yards sd 50) next to any matchup effect, so a full-weight
    additive adjustment from a few noisy games hurts.
- 2026-10-09, threshold sweep on the searches as first built (all units of a side averaged; the baseline of the per-unit design). Two runs
  byte-identical (rows d29f5c3f3a1ad925); comp_threshold_sweep_allunits.parquet and _detail_allunits.parquet. Every threshold passes the
  rule (real comps beat random pairing): the lowest is 0.40, with an improvement of 0.0049 [0.0017, 0.0081] in squared z (about 0.5%). But at
  every threshold the real comps' error is ABOVE that of a zero shift (0.40: real 1.039, random 1.044, zero 1.030; 0.70: 1.055 / 1.059 / 1.048).
  The shifts make the prediction of the surprise worse than adding nothing; only S5's volume side beats zero (0.938 vs 0.944). This design is
  not built further; the per-unit sweep reports the zero-shift comparison next to the random one (the unit diagnostic requires both).
- 2026-10-09: you decided decisions 0, 1 (scale + sweep rule) and 8, and accepted the rest; written above before any code change. The
  selected-window build that was running on the old settings did not finish: both copies hit the 2-hour timeout (it splits the families into one
  search per market and was about twice as slow as expected). Since it is superseded, it was not re-run and has no results; its gate ablation is not run.
- 4c.6 preview, recency_weighted window, before the null control was added: no market clears. rec and rec_yds are EDGE (gains of 0.0007 receptions
  and 0.01 yards, intervals across 0). The other markets are slightly worse with the comparable columns: spread -0.11 points (0 of 5 seasons),
  rush_yds -0.07 yards, both intervals touching 0. Two runs gave byte-identical results. As expected with nearly every search no_match: the
  columns carry almost no information yet. Final runs with the null control and on the selected windows are in progress.
- 4c.4 closed. The full backtest was run byte-identical twice (shifts c652d63e56619d1b), after its independent review, the cap fixes, a hash-seed
  determinism fix and a 37% speed-up with identical output. At the plan's constants, of 93,632 target-markets S1 matches 302, S3 8, S5 256, and
  S2 and S4 none. The lineup-adjusted vectors change about 6% of S2 / S4's 10 closest games.
- 4c.5 closed. The comparable table now carries the input shares and completeness for every search and target, and was regenerated
  byte-identical twice (afe25688fc05fb91). Its 4c.4 columns are identical to before. On real data every volume and efficiency table (10 quantities
  and the two team tables) gets exactly one comparables row per scored row: no fan-out, all floats, deterministic. A LightGBM fit with the 55
  comparable columns runs. 4c.5 also had its independent review, with four fixes test-first.
- 4c.3: S2/S5 two-sided, continuity in every search, S5 sigma fix, fallback to healthy. 50 tests pass. Vectors rebuilt twice with the
  played-population fix: all six outputs byte-identical.

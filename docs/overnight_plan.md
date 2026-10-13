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
  unit's sigma, the space chosen as in the searches (EXTENDED when both sides are EXTENDED-complete and its sigma exists, else BASE; corrected
  before the run from "BASE, EXTENDED for coverage_mix"), the healthy target vector against the pool vectors, as in the searches. The prediction
  is the similarity-weighted mean of the neighbours' own single-game outcomes.
- **(r) Random neighbours.** 20 earlier team-games drawn at random from the same pool with (k)'s weights; 10 draws, fixed seed. Reported only.
- **Loss.** Absolute error, in the outcome's own units.
- **Pass, per unit.** (k) beats (a): mean gain above zero with the season-week bootstrap lower bound above zero, at the Bonferroni level 1 - 0.05 / 9.

**What each outcome means.**
- **A unit passes:** its fingerprint carries signal, and the matching has something to work with on that unit.
- **Every unit fails:** the unit features are the problem. Rebuilding them is a large project, and it does not start without your decision.

**Stopping rule.** If the diagnostic fails AND the per-unit sweep has no passing threshold, the comps are parked at weight 0 for V1 (kept built
and tested) and the work moves to 4d.

## One weight rule for every layer (your decision of 2026-10-09)
- **Rule.** A Phase 4 layer keeps weight 1 in a market only on CLEARS: gain > 0, the 95% season-week interval excluding zero, and a win in at
  least MIN_SEASONS_WON = 2 seasons. Otherwise its weight is 0 and the market is flagged, never dropped. It is config.LAYER_WEIGHT_RULE /
  eval.compare.layer_weight, used by every ablation runner.
- **Why.** The layers used different rules. 4a and 4b kept a layer on "gain > 0 and wins in more than one season", the plan's running rule,
  so EDGE markets kept weight 1. 4c used CLEARS, then stricter tests. Phase 6 will inherit whatever weights go in, so they follow one standard.
- **Multiple comparisons.** The comparables' threshold and unit diagnostics added a Bonferroni correction across their 115 unit-search cells.
  The per-market layer gates use one interval per market and no correction.
- **The registry.** layer_weights.csv (build_layer_weights.py) holds every layer's weight per market, the table Phase 6 starts from.
  - **4a, provisional.** pass_att, spread and moneyline (EDGE) go to weight 0, flagged "pending 4a rerun". The 4a ablation (335752f) predates
    the role-share fixes in models/injuries.py: a role holder new to the role takes the role's average share (657f1b8), role-aware
    baselines (609d30a, 4445d25), fixed row order (c0fdf6e). So it is rerun on the current code, twice and byte-identical, and the final 4a
    weights come from the rerun under this rule.
  - **4b.** Unchanged: spread CLEARS (1), everything else 0.
  - **4c.** Parked (0).
- **Order.** Phase 6 does not start until the rule, the provisional 4a weights and the 4a rerun are done.

## 4d decisions (yours, 2026-10-12)
- **Sources** (docs/coaching_sources_4d0.md).
  - Pro Football Reference: rejected.
  - nflverse head coaches (schedules, per game, 2016-2024): approved as the 4d baseline.
  - Wikipedia / Wikidata: not used for V1.
  - Sam Hoppen's play-caller file: NOT ingested. A permission request is drafted for you to send (docs/coaching_messages_4d0.md),
    together with a one-line question to nflverse about where its head-coach names come from.
  - Previews: ask before downloading even a sample of a source under review.
- **Route: option 3.**
  - coaches_history has head_coach from nflverse for 2016-2024. offensive_coordinator and play_caller are UNKNOWN for 2016-2025, stored as
    null, never zero or an empty string. Tests: a missing play caller is null, and a feature built from it carries no information.
  - 2025 is the holdout and is not loaded: every role is UNKNOWN.
  - 2026 (live) comes from data/coaches_seed.csv, the hand-built 2026 staff: head coach, offensive coordinator and offensive play caller.
  - If Hoppen grants permission, play_caller is upgraded from his file at a pinned commit, and 4d is re-tested.
- **Fixed before any 4d result.** Under option 3 the backtest knows the head coach but not the play caller. A flat 4d result is therefore
  INCONCLUSIVE about play calling, not a verdict that coaching does not matter. It does not park 4d on that evidence, and the 2026 live use of
  coaches_seed.csv stays either way. A 4d market that CLEARS gets weight 1 under the one rule, as for every layer.
- **4e (weather): not built yet.** The backtest's weather is OBSERVED (Meteostat station readings at kickoff, 2020-2025), not known before
  kickoff. The options for handling observed-vs-forecast weather are laid out for your choice before anything is built.

## 4e gate (yours, 2026-10-12: option D with A1 and A2 together; fixed before anything is built)
- **Why.** The backtest only has OBSERVED weather (Meteostat station readings at kickoff), which nobody knew before kickoff. A layer that only
  works on exact observed weather will not work live. So the gate is run on weather degraded to forecast quality, three ways.
- **Variants (outdoor games only; applied to training and test rows alike; one draw per game from a generator seeded by the game id).**
  - **A1 bins.** Wind <10 / 10-15 / 15-20 / 20+ mph; temperature <=32 / 33-50 / 51-70 / >70 F; precipitation yes / no. The model sees the bin index.
  - **A2 noise.** Wind +/- N(0, 4 mph), temperature +/- N(0, 4 F), precipitation flipped with probability 0.2, seed 20261012. The sizes are my
    assumption, not measured.
  - **A2x2.** The same draws at double size: N(0, 8 mph), N(0, 8 F), flip probability 0.4.
- **Pass rule.** A market keeps weight 1 only if the layer CLEARS under A1, under A2 AND under A2x2 (an AND, so stricter, never a pick of the
  best). CLEARS is the one rule: gain > 0, the 95% season-week interval excluding zero, >= 2 seasons won. Judged, as the build plan says,
  on outdoor games with real wind (>= 15 mph) or cold (<= 32 F), taken from the observed values; on the other rows the layer must not be
  significantly worse (interval entirely below zero). Results on exact observed weather are reported for reference only and never gate.
- **Domes and closed roofs.** An indoor game (fixed dome, or a retractable roof closed) gets fixed neutral values in every variant (wind 0,
  precipitation 0, temperature 70 F) and an explicit indoor flag. It is never binned or noised. A test changes an indoor game's observed
  weather and checks that its features do not move. Roof state unknown: the weather features are missing (null, with the completeness
  penalty), not neutral. A missing weather row never crashes the pipeline.
- **Route.** Features into the Phase 3 volume and efficiency models and a per-market ablation, as for 4a and 4c. The learned shifts inside the
  4b simulation are wired only if a market CLEARS.
- **Forecast record (option C), running from 2026-10-12.** ingest/record_forecasts.py snapshots every live-season forecast into the tracked
  records/weather_forecast_log.csv (append-only, with pulled_at and hours_before_kickoff). After a season it gives a real forecast-time
  backtest and measured forecast errors to replace A2's assumed sizes. The recurring schedule is waiting on your go-ahead.
- **Licences.** The pipeline uses Meteostat: https://dev.meteostat.net/license.html says the data is "published under the terms of the Creative
  Commons Attribution 4.0 International (CC BY 4.0) license", that you "must give credit to Meteostat and its meteorological data providers"
  (example "Source: Meteostat, Deutscher Wetterdienst"), may share and adapt it "even commercially", and that data fetched through the Python
  library comes from providers whose own licences should be reviewed (not yet done per provider). Open-Meteo, only option B and not used:
  https://open-meteo.com/en/terms says "You may only use the free API services for non-commercial purposes", and commercial examples include
  "Operating websites or apps that have subscriptions or display advertisements". If this ever becomes a paid product, Open-Meteo would
  need its paid plan; Meteostat's attribution and the providers' licences would need checking.

## 4d.1 gate (fixed before anything is built; my reading of the plan is listed for your veto)
- **What the data allows.** Under option 3 only head coaches are known. The 2020-2024 seasons have 45 head-coach change events (34 at week 1,
  11 in-season). Only 9 have an earlier head-coach stop elsewhere in the 2016+ data: McCarthy (DAL 2020), Rivera (WAS 2020), Bevell (JAX
  2021 wk 15), Bowles (TB 2022), Pederson (JAX 2022), Wilks (CAR 2022 wk 6), Payton (DEN 2023), Reich (CAR 2023), Quinn (WAS 2024). The
  other 36 were coordinators or assistants, whose play-calling stops are UNKNOWN. So the layer can act in about a fifth of the events.
- **The prior** (build plan 4d.1). From play-by-play at each prior stop (a head-coach stint with another team, never the current team):
  - **Pass rate over expected.** Mean of (pass - xpass) over designed pass and run plays (nflfastR xpass, tagged derived).
  - **Pace.** Median seconds between a team's consecutive plays in neutral states: win probability 0.2-0.8, outside the last 2 minutes of each
    half, same drive, no timeout, penalty, spike or kneel.
  - **Fourth-down aggressiveness.** Go rate minus the league's same-season go rate in the same situation, in neutral states. A situation is
    ydstogo 1 / 2 / 3-4 / 5-7 / 8+ by yardline_100 1-20 / 21-40 / 41-60 / 61-80 / 81-99.
  - **Weights.** A stop is weighted by recency, 0.5 ^ (seasons since the stop ended / 3), and by length, games / (games + 17). The coach prior is
    the weighted mean over his prior stops. Blend with the current stint: weight 6 / (6 + games under him with this team so far).
  - **Rules.** A coach with no prior stop gets no prior (null, marked no-prior). Walk-forward only: a stop counts only if it ended before the
    target game. A team without a coaching change gets no prior.
- **Where it enters.** Feature columns into the Phase 3 volume and efficiency models (coach_prior_proe, coach_prior_pace, coach_prior_go4,
  coach_prior_weight, coach_games, coach_state), then a per-market ablation as for 4a and 4c. This departs from the build plan's wording (fill the
  4b.1 hook and the 4b.3 starting values): the 4b simulation carries weight only for spread and the ratings are EPA, which none of the three
  tendencies maps onto, so the Phase 3 models are where the layer can show lift. The 4b hooks stay empty; wire them only if a market CLEARS.
- **Windows.** W1: weeks 1-6 of every season. W2: a team-game within the first six games under a new head coach. W0: everything else.
- **Pass rule per market.** CLEARS on the W1 + W2 rows (gain > 0, the 95% season-week interval excluding zero, >= 2 seasons won) AND not
  significantly worse on W0 (the interval's upper bound >= 0). Otherwise weight 0, flagged. Judged on 2020-2024 only; 2025 stays locked.
- **Reading a flat result** (as already written): inconclusive about play calling, not a verdict on coaching. With 9 events it is also a
  low-power test, so the report states n for each window.
- **A cheaper upgrade than the 640-entry hand build.** The 36 events without a prior need only the new coach's earlier play-calling stops: 36
  sourced lookups. This, or Hoppen's file, is the lever that matters for 4d.

## Decisions of 2026-10-12, second batch (yours)
- **Hoppen's play-caller file: not pursued.** The permission request and the nflverse question are not being sent; I stop raising them. 4d runs
  on the nflverse head coach plus hand-sourced prior-stop lookups only.
- **Forecast recording runs on a schedule, on its own branch.** Every ~6 hours (UTC), Wednesday through Monday, pushing only
  records/weather_forecast_log.csv to the branch data-snapshots, never the working branch. The snapshot closest to 24 hours before kickoff is
  the one that matters. The task's approval setting and the first scheduled run are checked and reported (a missed week cannot be recovered).
- **The 2026 as-of snapshots are committed.** The injury reports, depth charts and rosters as pulled on 2026-10-07 cannot be pulled again as
  they stood; they go to tracked files, and the weekly run keeps saving them the same way.
- **.env** stays uncommitted and gitignored (confirmed); **.claude/settings.local.json** stays local.
- **Coaching columns go into the Phase 3 models, not the 4b hooks** (approved). The 4b simulation carries weight only for spread and the three
  tendencies do not map onto the EPA ratings. The hooks stay empty unless a market CLEARS.
- **Heavy runs are sequential** (four at once ran a process out of memory).
- **4e: a flat result under degraded weather is inconclusive, not a verdict on weather.** There are 133 severe-weather games (14-33 per
  season) and three variants that must all pass. The real test is the recorded forecasts.

## 4d.1, the fuller table (fixed before any lookup or rerun)
- **Why.** The first 4d.1 run (below, in the Log) is flat and has a prior in only 9 of 45 head-coach changes. The other 36 new head coaches were
  mostly coordinators. Their earlier play-calling stops are the missing prior.
- **The lookups.** For each of the 36 changes in 2020-2024 without a head-coach prior: the stops where he called the offense's plays, with the
  franchise and seasons, in data/play_caller_stops.csv (tracked: it is hand-made). Columns: coach, franchise (current code), first_season,
  last_season, hired_by, hire_key, source_url, confidence (high / medium / low), note.
- **Rules.**
  1. Only career history that ended BEFORE his first game as head coach (stop's last key < hire key); the code refuses a row that does not.
  2. Each row sourced by hand with a URL and a confidence. Not Pro Football Reference / Sports Reference, and not ESPN / ABC / Disney sites: their
     terms ban automated collection and use for models (docs/coaching_sources_4d0.md). Wikipedia pages, team and league releases and press reports,
     read one fact at a time.
  3. A stop counts when he was the offense's play caller. A coordinator title without play calling is not a stop; unclear = low confidence.
  4. Only stops in the play-by-play window (2016 on) can give tendencies; earlier ones are recorded as "pre-2016, unusable".
  5. The tendencies come from play-by-play of that franchise's offense over the seasons he called plays (whole seasons). The existing prior code
     weights them (recency half-life 3 seasons, length games / (games + 17)). A stop with the hiring franchise is excluded (the rule that a prior never
     uses the current team's games).
  6. The rerun uses rows of confidence high or medium. Counts of every kind are reported.
  7. A second pass re-checks five random rows against a different source; a disagreement is resolved or the row dropped.
- **Gate and reading.** Unchanged (W1+W2, CLEARS and not worse on W0). Both results are reported, the thin and the fuller; the thin one is already in
  the Log and is not replaced.

### 4d.1 fuller table: lookups done, rule 5a (written 2026-10-12 after the lookups, BEFORE the rerun; nothing was run on them)
- **Coverage.** 36 head-coach changes without a prior (35 coaches; Raheem Morris twice). Hand lookups (six sub-agents, Wikipedia/team/press pages; no PFR,
  Sports-Reference, ESPN/ABC/Disney) found a usable play-calling stop before the hire for 9 coaches (10 rows in data/play_caller_stops.csv): Stefanski,
  Bevell, A. Smith, O'Connell, McDaniels, Hackett, Daboll, Steichen (2 stops), Canales. The other 26 coaches have none (defensive or special-teams
  coaches, college only, or a coordinator title where someone else called the plays: Sirianni, Callahan, McDaniel, and O'Connell's Rams years).
  All 35 verdicts, with source and confidence, are in data/play_caller_lookups_4d1.csv. Confidence: 7 stop rows high, 3 medium (Bevell, Smith, McDaniels).
- **Rule 5a.** Rule 5 says whole seasons. A season in which he took the play calling over or lost it part-way is included only if he called at least
  half of the team's games that season, and only when the switch week is sourced. So left out: Stefanski MIN 2018 (last 3 games), Steichen PHI 2021
  (took over "about midway", week unknown). Kept: O'Connell WAS 2019 (from week 6), Hackett JAX 2016 (from week 9) and 2018 (to week 12), Steichen LAC 2019
  (from week 9). Partial seasons still carry some of the other caller's plays; that noise is accepted and stated.
- **Second pass (rule 7).** Five rows drawn with random.Random(20261012): A. Smith TEN, Bevell SEA, O'Connell WAS, Hackett JAX, McDaniels NE, each re-checked
  against a different source than the first (search results from SI/NBC Sports, commanders.com, profootballrumors, Fox Sports, NBC Sports Boston/CBS).
  All five agree. Smith 2020 is now supported by a second report; McDaniels 2019-2020 by CBS (Newton defending his play-calling). Bevell stays medium: the
  second source calls him the play caller but only loosely (Tom Cable shared the offense).
- **Not a stop:** Bevell's DET 2019-2020 (the hiring franchise, and already the head-coach stop).

### Pre-hire audit of the play-calling stops (2026-10-12, asked by the project manager)
Rule 1 is enforced in code (manual_stops raises if a stop's last game is not before the hire) and read by hand here: every row below uses facts
about seasons that ended before the coach's first game as head coach; the sources and their publication dates (where the page shows one) are:
| coach (hire) | first source, date | second source, date | after the hire? |
|---|---|---|---|
| Stefanski (CLE 2020) | vikings.com, 2019 | NFL.com, Dec 2018 | no |
| Bevell (DET 2020, week 13) | NFL.com, Jan 2018 | MyNorthwest, Jan 2018 | no |
| A. Smith (ATL 2021) | CBS, 2019 | SI / NBC Sports press release, Aug 2020 and Oct 2020 | no (Wikipedia, undated live page, is not relied on for a date) |
| O'Connell (MIN 2022) | commanders.com, 12 Oct 2019 | commanders.com, Oct 2019 | no |
| McDaniels (LV 2022) | Boston Globe, 18 Dec 2021 | CBS, Feb 2019 and Dec 2020 | no |
| Hackett (DEN 2022) | jaguars.com, 2016 | abc7news, 2016; Fox Sports, Nov 2018 | no |
| Daboll (NYG 2022) | Bills.com, 14 Aug 2018 | **Giants.com, 28 Jan 2022: after the hire was announced (4 Jan 2022) but before his first game; it is a summary of the same Bills years, read for the dates only** | second source only |
| Steichen (IND 2023) | chargers.com, Oct 2019; CBS, Jun 2022 | NBC Sports Philadelphia, Jun 2022 | no |
| Canales (CAR 2024) | tampabay28, Aug 2023 | SI, 23 Feb 2023 | no |
What cannot be dated: Wikipedia pages (live, undated) were used as second sources only, for titles and years. No source published after a coach's first
game as head coach was used. The "none / not a caller" verdicts of the other 26 coaches rest on the same kind of pages; a missing stop adds no information
(null prior), so an error there can only leave a coach without a prior, never give him a wrong one.

### 2025+ depth-chart loader: a Phase 5 blocker, fixed (2026-10-12)
- **The problem.** nflverse's depth charts since the 2025 season have a different layout: a snapshot timestamp `dt`, no season or week, positions as
  `pos_abb`. The raw table holds 1,169,694 such rows (season NULL). Every Phase 3-4 loader filters on season and week, so they all skip these rows:
  harmless for the 2020-2024 backtest, but live predictions would see no depth chart.
- **The fix, test-first** (tests/test_depth_live.py, 6 tests): features/depth_live.py `chart_as_of(frame, cutoff, week)` gives the newest offensive
  chart per team at or before the cutoff in the old columns (season, week, team, gsis_id, position, depth_team, formation); a team with no chart yet has
  no rows. A chart belongs to the season that starts in March, so January and February 2026 charts are the 2025 season. The holdout guard: a cutoff in
  the 2025 season raises HoldoutError, and no 2025-season row is ever returned (tested with December and February charts). On the real table: the
  2026-10-07 chart gives all 32 teams, 982 offensive rows, KC QB1 = 00-0033873.
- **Not yet wired.** The Phase 5 live feature builder calls chart_as_of for each team's coming game (week from the schedule); the backtest loaders are
  unchanged, so no historical result moves.
- **A guard bug found by this work, fixed.** ingest/save_asof.py kept charts by the YEAR of `dt` >= 2026, which includes January and February 2026
  (the 2025 season, the holdout). Now it uses the season rule above (test-first). The snapshot already committed, records/asof/2026-10-07T00/depth_charts.parquet,
  holds 810,358 rows, of which 158,515 are January-February 2026 (2025-season) charts. No code reads that file's rows before March 2026 (chart_as_of and
  live_charts drop them); they are the same rows the raw database already holds. They have not been deleted from the file, because a snapshot is never rewritten
  (decided 2026-10-12: leave it as is). **Rule: that file, and any later asof depth_charts.parquet, is read only through features.depth_live
  (`read_asof_charts`, which drops the 2025-season rows).** Guard: tests/test_depth_live.py fails if any Python file outside ingest/save_asof.py,
  ingest/scheduled_snapshot.py and features/depth_live.py mentions the asof directory (mutation-checked with a module that reads the file directly).

## Phase 5 gates (fixed 2026-10-12, before anything of Phase 5 is built; spec: docs/build_plan.md "Phase 5")
**P5.1 What the weekly run produces.** `run_weekly.py` writes, for every game kicking off in the coming window, one record per (eligible player, market) and
one per (game, game market) for the 13 markets, with: prediction_id, run_id, run_timestamp (UTC), kickoff_time (UTC), version (1, 2, ...), game_id,
player_id (null for game markets), market, model_version, mean, median, quantiles (5, 10, 25, 50, 75, 90, 95), the threshold ladder, confidence components,
comp shifts and no_match flags, the injury-status snapshot used, and input_hash.
- **mean** is the Phase 3 combined prediction. **median, quantiles, ladder, confidence and comp shifts are NULL (never 0) until a calibrated source exists**
  (Phase 6 supplies them under a NEW model_version; the comps layer is parked at weight 0, so its shift columns stay null with no_match = null).
- **Layers.** model_version is a hash of the code revision and of layer_weights.csv. A layer enters a market's prediction only with weight 1 in that file
  (today: 4a for rush_att, rush_yds, targets, rec, rec_yds; 4b for spread; Phase 3 for all markets). 4c, 4d and 4e have weight 0 and are not in the run.
- **Pure track.** The prediction record has no column derived from a sportsbook line (schema allowlist, tested). The schedule line (spread, moneyline,
  total) is logged to its OWN table (lines_log), keyed by game and capture time, and is read by nothing in the prediction path.

**P5.2 Inputs are as-of stamped.** Every input table of a run is the state at run_timestamp: injuries, depth charts, rosters and schedule are read through
the as-of rule (rows published or stamped at or before run_timestamp; depth charts through features.depth_live.chart_as_of with the game's week), weather from
the recorded forecast nearest before the run. The run stores, per input table, the row count, the newest stamp used and a hash, and input_hash is the hash of
those. Nothing dated after run_timestamp can enter a feature (tested with a row stamped one second after).

**P5.3 Stored before kickoff, always, and never edited.**
- The writer takes the clock as a parameter. A record whose kickoff_time is at or before the writer's clock is REFUSED and the refusal is logged (game,
  player, market, clock, kickoff) in refusals.csv; nothing is written for it.
- Rows are append-only. A rerun for the same (game, player, market) writes version = previous max + 1 and leaves earlier versions byte-identical; a rerun
  with identical inputs and model_version writes nothing new (idempotent), so the Sunday-morning rerun adds a version only when something changed.
- Grading uses the highest version stored before kickoff (stored_at < kickoff_time). A version stored after kickoff is never graded (and cannot exist, by the
  refusal above). The store is tracked in git (records/predictions/) and pushed to the data-snapshots branch by the run, so the commit time is an outside timestamp.

**P5.4 Holdout.** No Phase 5 code path may use a 2025 outcome as a training row, a target, an evaluation or a tuning input; the loaders' cap_season guard stays.
**DECIDED 2026-10-12 (yours): 2025 and 2026-to-date games may be read as INPUT HISTORY for live targets; every model stays FIT on 2020-2024 rows only.**
Reason you gave: the holdout protects the models, not the data; reading 2025 results as recent form does not touch the question the benchmark answers; skipping
2025 would stale every veteran's features and leave 2025 rookies with none; fitting on 2025 would burn the one-time benchmark. Guardrails, each with a test:
1. **One named function** (`live/history.py: read_input_history`) is the only code allowed to read 2025+ games. It opens a scoped unlock in config
   (`config.input_history`, a context manager that lifts the loaders' season cap to the requested season for that block only), and returns rows with
   gameday strictly before the cutoff (the target's kickoff / week cutoff), marked input-only. It returns no targets and no training rows. A mutation test
   (a version that returns on-or-after-cutoff rows, or a target row) must fail the tests, and a test fails if `input_history(` is used anywhere else.
2. **The Phase 6 benchmark path** uses as-of cutoffs inside 2025 only: it never sees a 2026 row, and a 2025 week-10 target never sees week 11 or later
   (tested on `read_input_history` with 2025 cutoffs).
3. **Standardization is as-of too.** League means and standard deviations (z-scores, shrinkage, any normalisation) use only data before the target, in the
   live path as in the backtest (future-blindness test on the live feature function: appending rows dated after the cutoff changes no feature).
4. **Refit rule.** Models stay fit on 2020-2024 until the Phase 6 benchmark has run once and its result is recorded in this plan. Refitting on 2025 is then a
   separate decision of yours, and from then on 2026 is the clean holdout.

**P5.7 Wiring the weight-1 layers into the live run (decided 2026-10-12: layers first, then the schedule; fixed before the wiring is built).**
1. **The weights file decides.** A prediction for market M uses layer L exactly when layer_weights.csv has weight 1.0 for (L, M). No layer name or market list is hard-coded
   in the run: code holds only a registry of layer builders by name, and a layer with weight 1.0 but no builder stops the run with an error (never silently dropped, never
   silently added). Phase 6 changing a weight changes the run with no code change.
2. **Stack label per prediction.** Every record carries `stack` = "phase3" plus "+4a" / "+4b" for each layer actually used for that record's market (e.g. "phase3+4a", "phase3+4b").
   It is part of the record schema and of the store's identity check (a change of stack is a new version). Grading and every summary are reported per stack label and never
   blended; a week that mixes stacks shows one block per label.
3. **Exact replay with the layers in.** After wiring, the 2024 week 10 replay is rerun against the committed walk-forward predictions of every layer: the Phase 3 predictions, the
   4a "with injury features" predictions (injury_ablation_predictions_*), and the 4b simulated spread. Required: identical to within 1e-9, as for Phase 3. Any drift means the live
   path differs from the backtest and nothing is stored until it is explained.
4. **Holdout.** The inputs of 4a and 4b for 2025 and 2026-to-date (injury reports, weekly rosters, snap counts, ratings, game state) come through the scoped input-history
   function only (live/history.py); the layers' own sub-models (status, shifts, early exit; ratings and state coefficients) stay fit on 2020-2024 rows; the same mutation tests
   (cutoff comparison, no filter, over-wide unlock) apply to every new reader.
5. **Timing.** A game that kicks off before the full stack is wired gets no stored record rather than a weak-stack one (decided by you); the store's refusal rule makes this
   automatic, and nothing is stored for a game until the stack in force is final for it.
6. **Before the schedule exists:** the full test suite passes; the push path to data-snapshots has landed once; then the Friday 20:00 UTC and Sunday-morning routine is created
   and its approval setting is reported.

**P5.5 Grading (`grade_week.py`)** is idempotent (a second run adds no rows and changes none), keeps the pure and the market-aware tracks in separate tables,
grades an eligible non-player as actual = 0 with did_not_play = true and reports those rows separately, and writes the weekly summary next to the best Phase 2
baseline on the same rows.

**P5.6 Tests that must exist before the run is trusted:** a game with a past kickoff writes nothing and is logged; a rerun creates version 2 and leaves
version 1 byte-identical; an identical rerun writes nothing; the prediction table has no sportsbook-derived column; a row stamped after run_timestamp changes
no feature; the 2025 holdout guard rejects a 2025 target and a 2025 training row; a prediction stored after kickoff is never graded; grading twice is identical.

**Gate (from the build plan, unchanged):** the weekly cycle runs with no manual step beyond starting the two scripts, every prediction is stored before kickoff,
and at least three to four weeks of graded live predictions exist before Phase 6.

## Residual pattern screen (asked for by the project manager, 2026-10-12; fixed here BEFORE any candidate feature is built or any residual is looked at)
- **Question.** Which signals the Phase 3 models do not use yet carry information about the models' misses (actual - prediction)? This is a SCREEN, not a layer:
  it sets no weight and changes no prediction. A survivor is a hypothesis that earns its own layer gate and a full ablation (the feature inside the retrained model,
  the 4a/4d/4e procedure) before it can get weight 1.
- **Data.** The committed walk-forward predictions of `vol_x_eff` (eligibility_v2), 2020-2024 player-games with a prediction and an actual. Seasons <= 2024 only (asserted);
  no sportsbook line, spread, total or moneyline column is read (pure track); weather (temp, wind, roof) was settled by 4e and is not retested.
- **Markets.** targets, rush_att, pass_att, rec_yds, rush_yds, pass_yds. Each candidate is tested only on the markets listed with it.
- **Candidates (fixed, 21; each strictly as-of: schedule facts known in advance, or built only from games before the row's week cutoff).**
  - Schedule context, all six markets: X1 own rest days; X2 opponent rest days; X3 after-bye flag (rest >= 13); X4 prime-time flag (Thursday, Monday, or kickoff 19:00+);
    X5 divisional game; X6 week of the season; X7 head referee's trailing plays per game above the league mean (>= 20 earlier games); X8 head referee's trailing penalties per game.
  - Player trend, receiving markets (targets, rec_yds): X9 snap-share trend (mean offense_pct of the last 3 games - last 8; also tested on rush_att, rush_yds); X10 target-share trend
    (last 3 - last 8); X11 air-yards share, trailing 8; X12 average depth of target, trailing 8; X13 luck: trailing-8 (rec yards - expected rec yards) per game (ff_opportunity);
    X14 average separation, trailing (nextgen); X15 yards after the catch above expectation, trailing (nextgen).
  - Player trend, rushing markets (rush_att, rush_yds): X16 carry-share trend (last 3 - last 8); X17 luck: trailing-8 (rush yards - expected rush yards) per game.
  - Team tendency: X18 script-adjusted pass tendency = trailing-8 pass rate in neutral situations (win probability 0.2-0.8, quarters 1-3) minus overall pass rate, on pass_att, rush_att, targets;
    X19 no-huddle rate, trailing 8, on pass_att, rush_att.
  - Opponent defense: X20 pressure rate (QB hits + sacks per dropback), trailing 8, on pass_att, pass_yds, targets; X21 man-coverage rate (participation) and blitz rate (>= 5 rushers),
    trailing 8, on targets, rec_yds, pass_yds (counted as two candidates in the multiplicity count).
- **Test.** For each (candidate, market): rows with the candidate present (coverage < 70% of the market's rows -> reported as "insufficient coverage", not tested). Standardize the candidate on the
  training seasons only. Folds 2021, 2022, 2023, 2024: fit residual ~ a + b * z by least squares on the earlier seasons, predict the residual on the fold season, and score
  gain = mean(|r|) - mean(|r - r_hat|) (the absolute-error gain every ablation here uses). The pooled gain over the four folds gets the season-week cluster bootstrap (eval.compare.cluster_bootstrap, 10,000 resamples,
  seed config.BOOTSTRAP_SEED) at the Bonferroni level 1 - 0.05 / m, m = the number of (candidate, market) cells actually tested (printed). Squared-error gain is reported next to it, not used for the verdict.
- **Verdict.** CLEARS at the Bonferroni level: gain > 0, the interval above zero, wins in >= 2 of the 4 fold seasons, and the sign of b the same in every fold. EDGE and NO as in eval.compare.verdict.
  A candidate that clears nowhere is reported as null. No threshold is tuned afterwards to produce a survivor.
- **Amendment 1 (2026-10-12, written after the first run, BEFORE the rerun; the estimator changes, no threshold, list or verdict rule does).** The first run fit each correction by least squares, as written above.
  It was invalid by construction: every Phase 3 model is fit to the median (L1) and its residuals are right-skewed (targets: mean residual +0.31, median -0.04; rec_yds +3.39 vs -2.20), so the least-squares INTERCEPT alone
  shifts predictions toward the mean and costs absolute error (adding the training-mean residual to 2024 targets: MAE 1.730 -> 1.759; adding the median: 1.730 -> 1.730). Result: 72 of 74 cells showed a negative gain regardless of the candidate. Those numbers are kept
  as data/processed/residual_screen_cells_ols_invalid.parquet and are not evidence about any candidate. The fit is now least absolute deviations (median regression, sklearn QuantileRegressor, quantile 0.5, no penalty), the estimator that matches the metric.
  Everything else (candidates, coverage rule, folds, gain, Bonferroni level over the 74 cells, verdict) is unchanged.
- **Amendment 2 (2026-10-12, written after the second run, BEFORE the third; again the estimator/score changes, no list, level or verdict rule).** The median-regression run returned 9 CLEARS, all on rec_yds, and the
  per-season gains were nearly identical (about 0.12 per row) for unrelated candidates (rest days, week number, divisional game, after-bye flag). A gain that does not depend on the candidate is the fitted INTERCEPT
  (a median shift of the Phase 3 prediction), not the candidate. The gain is therefore scored against the intercept-only median correction fitted on the same training seasons: gain = mean(|r - a|) - mean(|r - a - b z|),
  so only what the candidate's slope adds counts. The intercept-only gain is reported beside it (it is a calibration finding about the Phase 3 product, handled separately, not a candidate). The second run's table is
  data/processed/residual_screen_cells_lad_total_gain.parquet, kept and not used for any verdict.
- **Limits, stated now.** Linear, one candidate at a time: an interaction or a non-linear effect is not tested; a candidate that fails here can still help inside a retrained tree model, and one
  that passes may not survive it. Also reported: the noise floor (what a perfect-mean Poisson outcome would miss by) beside the model's error for each volume market, so "how far from the best possible" is on the page.
- **Residual screen result (2026-10-12, third run, amendments 1 and 2 applied; 74 cells tested, Bonferroni level 0.99932; 8 cells untested for coverage < 70%: separation and YAC-above-expectation (51%), referee plays / penalties on targets and rec_yds (69%)).**
  1 CLEARS, 25 EDGE, 48 NO. The one CLEARS: X9 snap-share trend on rec_yds, gain 0.0236 yards per row (0.13% of the 17.8-yard mean miss), positive in all four fold seasons, interval [0.0069, 0.0425].
  Statistically real, practically negligible; the same candidate is EDGE on targets (+0.0029, 0.17%) and rush_yds. No other candidate (rest, bye, divisional, prime time, week, referee, air-yards share, depth of target, luck vs expectation, carry-share trend,
  neutral-script pass tendency, no-huddle, opponent pressure / man / blitz rate) cleared anywhere. Largest EDGE: opponent blitz rate on pass_yds (+0.126 yards, interval [-0.34, +0.37], 2,044 QB-games). Intercept-only (calibration) gains: rec_yds +0.083, pass_yds -0.26: nothing to recalibrate.
  Reading: the Phase 3 stack is not missing a simple linear signal in this data; the remaining miss is mostly per-game variance. Weights unchanged (the screen sets none). Open, not tested: interactions and non-linear effects,
  richer matchup data (receiver vs the specific defender, route-level), in-game events. X9 on rec_yds may enter a layer gate only if the project manager asks for one.

## Lines comparison, game markets (asked for by the project manager, 2026-10-12; fixed here BEFORE any model number is compared with any line)
- **Question.** Does the model's game forecast carry information the schedule's sportsbook line does not? This is an EVALUATION after the fact. It reads the line from the schedule table only to score, never as a model input, and nothing about it enters the model, the prediction record, or the weights (the pure track stays pure). The live line log (live/lines.py) is untouched.
- **Data.** Development seasons 2020-2024 only (asserted; the 2025 lines are not read: 2025 stays the Phase 6 holdout). Regular-season games with a line and a committed prediction. Lines: the schedule table's spread_line, total_line, home_moneyline, away_moneyline (positive spread_line = home favoured; the sign is checked against the results before anything is scored). Predictions: the committed walk-forward Phase 3 (vol_x_eff) spread, total and win probability, and the 4b simulated spread (the mean of Normal(margin_mean, margin_sd) from the game_expectations table, seeded by game id, 2,000 draws, exactly as layer_sim.spread_sim_mean).
- **Not in this test.** Player-prop lines: the project has no prop-line data. A prop source would have to be downloaded, which I do not do without asking first (a source under review). Prop numbers stay unscored against lines until the project manager picks a source.
- **Scores (all with the season-week cluster bootstrap, 10,000 resamples, seed config.BOOTSTRAP_SEED).**
  - S1, accuracy: mean absolute error against the actual margin and total, model vs line (gain = line MAE - model MAE; positive = the model beat the line), for each of the three margin forecasts and the total.
  - S2, disagreement hit rate: for each margin forecast and total, the games where the model differs from the line by at least d points, d in {1, 2, 3} (fixed now; no other d is tried), count how often the actual result falls on the model's side of the line (pushes dropped). Break-even at the standard -110 price is 52.38%. 3 forecasts x 3 thresholds for the margin + 3 thresholds for the total = 12 cells; the interval level is Bonferroni, 1 - 0.05 / 12.
  - S3, win probability: log loss of the model's win probability vs the line's de-vigged implied probability (home / (home + away) of the implied probabilities), gain = line log loss - model log loss.
- **Verdict.** An S2 cell shows an EDGE only if its hit rate's lower bound at the Bonferroni level exceeds 52.38% AND it exceeds it in at least 2 of the 5 seasons separately; S1 / S3 are CLEARS under the usual rule (gain > 0, 95% interval above 0, >= 2 seasons won). Anything else is reported as no edge. No threshold, d, or price is changed after the result is seen. The result sets no weight and changes nothing in the model; it only says whether the question is worth a Phase 6 test on the 2025 holdout.
- **Honest prior, written before the run.** Closing lines are the market's best estimate, and a model built from public box-score history has rarely beaten them. The likely result is no edge; a null result is the answer, not a failure of the build.
- **Lines comparison result, game markets (2026-10-12; run once on 2020-2024, rule above unchanged).** No edge anywhere, and the model is worse than the line. S1: the line beat every forecast on absolute error, with intervals excluding zero the wrong way
  (margin: Phase 3 10.48 vs line 9.82; 4b.2 expected margin 10.22 vs 9.77; 4b simulated spread 10.22 vs 9.77; total 10.72 vs 10.33). S3: the line's de-vigged win probability has the lower log loss (0.609 vs 0.654, interval excludes zero the wrong way).
  S2: all 12 disagreement cells were "no edge"; hit rates 49.0%-51.3% against the 52.38% break-even, none with a lower bound above it. The 4b spread's earlier CLEARS was against the Phase 3 spread, not against the line; it is a better forecast than Phase 3 and still a worse one than the market.
  Nothing changes in the model or the weights. Player-prop lines remain unscored (no source; a source is the project manager's decision). Output: data/processed/lines_comparison_games.parquet; script scripts/lines_comparison.py.

## Player-prop line logging (asked for by the project manager, 2026-10-12; fixed here BEFORE the logger is built)
- **Purpose.** Collect sportsbook player-prop lines week by week, timestamped before kickoff, in their OWN append-only table, so that a later pre-registered test can ask whether the model's player numbers carry information a prop line does not. The log is read by nothing in the prediction path (the same rule as the game-line log, live/lines.py): no model, feature, weight or stored prediction ever reads it. A test enforces it.
- **Source.** None is chosen or downloaded. The first source is manual entry: a CSV of lines the project manager (or I, from screenshots he sends) types in, one row per line as the book showed it. An API or paid-data adapter is a separate decision for the project manager, with its terms read first; it would write the same table.
- **The table** (records/lines/prop_lines_log.csv, append-only, never edited): captured_at (UTC), book, game_id, player_id, player_name (as typed), team, market (one of our player markets), book_market (the book's own name), side (over / under / at_least), line (number), price (American odds, empty if unknown), is_alt (alternate ladder rung vs the posted main line), source (manual / the adapter's name), note. A missing value is empty, never 0.
- **Rules.**
  1. A line captured at or after the game's kickoff is refused and written to the rejects file (an in-play line depends on the game state and is not comparable to a pre-game prediction).
  2. A row is written only if it differs from the last logged row with the same (book, game_id, player_id, market, side, line, is_alt): a changed price is a new row, an identical re-capture is not (counted as unchanged).
  3. A player name is matched to a gsis id by name and team against the player table; no match or more than one match is rejected with the reason, never guessed. A market the book calls by another name is mapped by an explicit table; an unmapped market is rejected.
  4. Every refusal goes to records/lines/prop_lines_rejects.csv with its reason; nothing is dropped silently.
  5. Published to data-snapshots like the other records, so the commit time is an outside timestamp of when each line was logged.
- **The test that will use it is written later, before the first look at the data.** It will be pre-registered in this plan with its thresholds, its multiplicity correction and the number of weeks of data required. Nothing here compares a model number with a line.
- **Prop-line logger built (2026-10-13), as specified above.** live/prop_lines.py (`log_prop_lines`), scripts/log_prop_lines.py (CSV in, `--push` publishes the log and the rejects file to data-snapshots), tests/test_prop_lines.py (10 tests: matched player and our market name, kickoff refusal, dedup and changed price, append-only, unmatched / ambiguous / wrong-team rejection, suffix and case matching, alt flag and unmapped market, QB rushing mapped to qb_rush_*, game by team + gameday and missing price stays empty, and that nothing in live/ models/ features/ eval/ sim/ ingest/ or the root run scripts reads the log).
  No source is chosen or downloaded; the first source is manual entry. Smoke run against the real schedule: a line for the already-started Monday night game was refused with the kickoff reason.

## Prediction schedule, calibration and grading (decided and fixed 2026-10-13, before the code below is written)
- **Schedule (created 2026-10-13, after the full suite passed (471) and the push path landed twice).** Four routines fire "PREDICT" into the dedicated predictor session (session_01EMRZGLMaRbQhgNWY6yd5Zo, which pushes to data-snapshots): Friday 19:48 UTC (the main run, after the Friday injury report), Sunday 15:47 UTC (after the 1 pm ET inactives list, which posts about 90 minutes before kickoff, and before kickoff at 17:00 UTC), Thursday 22:47 UTC and Monday 22:47 UTC (the prime-time games, about 85 minutes before the 00:15 UTC kickoff, so those games get a record too). The decision of 2026-10-12 was Friday 20:00 UTC and Sunday morning; the Sunday run moved from morning to 15:47 UTC because a morning run cannot see the inactives, and the Thursday and Monday runs are added so the prime-time games are not dropped. Each run takes about 8 minutes; a game that kicks off before its run is refused by the store. Approval setting: the routines carry no tool grants of their own (they fire into the existing session, which runs in default permission mode with Bash, Read, Write and Edit pre-approved and is told not to call claude-code-remote tools).
- **Predictive distribution (Phase 5.1 fields and Phase 6.3).** For a prediction `p` of market `m` at (season, week) the pool is every committed walk-forward row of that market (vol_x_eff, 2020-2024, prediction and actual both present) from a STRICTLY EARLIER (season, week); the K rows nearest to `p` in prediction rank are a sample of what the real number did next to a prediction like `p`. K = 600 for player markets (250 for qb_rush_att / qb_rush_yds), 200 for spread and total. Burn-in: a pool below 1,500 rows (players) / 400 (games) gives no distribution, flagged. K and the burn-in are fixed now and not tuned.
  Quantiles q05, q10, q25, q50 (= median), q75, q90, q95 are the quantiles of that sample. Ladder: P(actual >= t) for fixed round rungs: pass_att 20/25/30/35/40; pass_cmp 15/20/25/30; pass_yds 150/200/250/300/350; qb_rush_att 2/4/6/8; qb_rush_yds 10/20/30/50; rush_att 5/10/15/20/25; rush_yds 25/50/75/100/125; targets 2/4/6/8/10; rec 2/4/6/8; rec_yds 25/50/75/100/125; total 35/40/45/50/55; spread and moneyline have none (spread keeps its quantiles; the moneyline mean is already a probability). The mean is never changed by this layer.
- **Width fix.** One scale c per market multiplies each quantile's distance from the median (q' = median + c (q - median)); c is the value on the grid 0.70-1.50 (step 0.01) that minimises |cov50 - 0.5| + |cov80 - 0.8| + |cov90 - 0.9| (cov = share of actuals inside the central interval). Walk-forward check: folds 2022, 2023, 2024, c fitted on seasons before the fold, scored on the fold; c is used only if it lowers that summed coverage error in at least 2 of the 3 folds, else c = 1 and the market is flagged "width fix not helpful". Live c is fitted on all 2020-2024 rows.
- **Isotonic correction.** Per market, on the ladder probabilities (pairs: the probability for a rung, whether the actual reached it, pooled over rungs), sklearn IsotonicRegression; same folds; used only if it lowers the Brier score in at least 2 of the 3 folds, else not applied. Reliability tables (binned predicted probability vs observed share) before and after, per market, and PNG diagrams in data/processed/calibration/.
- **Rules.** 2020-2024 only (asserted); no line is read; 2025 is not read and live 2026 results never refit the calibrator (a refit is a separate decision). The calibrator is saved as an artifact with a hash recorded in each run's manifest. A market whose pool is below the burn-in or whose result is not stable is stored with null quantiles and a flag, never with a made-up distribution.
- **Grading (Phase 5.2).** grade_week.py appends to records/graded/graded_predictions.parquet (never rewritten; grading a week twice adds nothing: the key is prediction_id) for the newest version of each prediction stored before kickoff. Per row: actual, error, absolute error, the band of q05..q95 the actual landed in, the probability given to each ladder rung and its Brier term, did_not_play (an eligible player who did not play is graded with actual 0 and flagged, and reported separately), stack label. A version stored at or after kickoff is never graded. Weekly summary per market and per stack (never blended): n, MAE, Brier, coverage of the 50/80/90 intervals, and the same MAE for the best simple baseline on the same rows. Pure track only; no line enters the graded table (market comparison stays in the lines logs).
- **Calibration result (2026-10-13; run_calibration.py, twice byte-identical incl. the PNGs; calibrator hash af47a213d0fd9b46).** Fold check on 2022-2024 with the rules above, nothing tuned after seeing it.
  Coverage of the stated 50 / 80 / 90 % central intervals out of fold (2024): spread 54 / 80 / 90 and total 59 / 83 / 90 (close to stated); rec_yds 60 / 86 / 93, rush_yds 55 / 83 / 92 (a little wide); rush_att 62 / 85 / 93, targets 66 / 88 / 95, rec 71 / 91 / 95 (wide: whole-number outcomes with the quantile edge counted inside make a count interval cover more than its nominal share, and a scale on the spread cannot fix a step).
  Width fix used for rush_att (c 0.99, coverage error 0.25 -> 0.15) and rush_yds (c 0.99, 0.15 -> 0.11); not used for rec, rec_yds, targets (no help), total and spread (one fold only). Isotonic: used nowhere (Brier slightly worse in the folds). pass_att / pass_cmp / pass_yds have only 2,044 rows in all, so the 1,500-row burn-in leaves no fold: they get a distribution from the full pool with c = 1 and no isotonic, flagged "no fold check"; qb_rush_att / qb_rush_yds (310 / 293 rows) are below the burn-in and get NO distribution (null quantiles, never made up).
  Meaning for the stored numbers: the ladder and quantiles are honest and slightly conservative for counts, close to stated for yards and the game spread / total. Reliability diagrams: data/processed/calibration/*.png.

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
- 2026-10-12, retrospective check of 2026 week 5 (**NOT a track record**; scripts/retro_check_2026_week5.py, retro_check_2026_week5.parquet). The full live stack was run as of 2026-10-07 13:00 UTC
  (only data pulled by then) and graded on the 14 games played by 12 Oct against results pulled afterwards. The predictions were made after the fact, so they count toward no gate and no record.
  Played players only, MAE of the model vs each player's mean of his last four games: pass_att 6.16 vs 7.88 (n 25), pass_cmp 3.79 vs 5.05, pass_yds 57.9 vs 70.8, rec 1.33 vs 1.43 (n 187), rec_yds 18.6 vs 20.0,
  targets 1.68 vs 1.87, rush_att 3.44 vs 3.56 (n 59), rush_yds 20.8 vs 21.0, qb_rush_att 1.59 vs 1.61 (n 7), qb_rush_yds 27.7 vs 28.8; game markets (n 14): spread MAE 10.26, total MAE 10.89, moneyline Brier-style
  error 0.458 (no yardstick). 14 games is far too few to separate model from baseline, and nothing here compares with a market. Eligible players who did not play: 77 receiving, 27 rushing, 3 passing rows
  (reported separately, as in grading). Direction matches the 2020-2024 backtest; it is a sanity check of the live path on unseen games, nothing more.
- 2026-10-12, recording routine: the first design FAILED its push test and was replaced. A routine that starts a fresh session each time (trig_01AmDK2LY8cnxUoGCvKqZXBu) gets no repository in its
  session sources, so the git proxy refuses the push with a 403 ("not in this session's authorized repository set"); the commits would have been lost with the container. Test fires #1 and #2
  looked fine only because they recorded nothing to push. Fix: a dedicated long-lived session created WITH the repository as its source and data-snapshots as its outcome branch
  (session_01HEYLULjTuAturFyEuhYJWC, permission mode default with Bash/Read/Write/Edit pre-approved: it ran without stopping for approval), and a routine that messages that session
  (trig_01QnyyzrPT7HbYGzpkut8cf9, "RECORD", 7 0,6,12,18 UTC on Sun, Mon, Wed-Sat). The old trigger is disabled. **Push path proven once:** the session recorded 2 forecast rows and
  pushed commit 5187a6d822af697bfcdb51d58c2a6a7919050156 to origin/data-snapshots at 2026-10-12 15:03 UTC. The first scheduled run is 18:07 UTC today (checked 18:30 UTC).
  A weekly-prediction session is built the same way (session_01EMRZGLMaRbQhgNWY6yd5Zo); its first message is a dry run.
- 2026-10-12, Phase 5 layers wired and replayed (P5.7). Layers on/off come from layer_weights.csv (4a for rush_att, rush_yds, targets, rec, rec_yds; 4b for spread); each record carries its
  stack label ("phase3", "phase3+4a", "phase3+4b"); grading and summaries are per label. **Replay of the whole live path on 2024 week 10** (outcomes of the week removed from a working
  copy of the database, targets from the schedule and chart, features from history, models fit on rows before the cutoff): the Phase 3 markets, the 4a markets and the 4b spread equal
  the committed walk-forward predictions to < 1e-9 (the 4b expected margin and SD too). Defects the replay found and that are fixed: (1) 2025+ injury rows have no posting time
  (date_modified is NULL in nflverse's new format): a row in a pull made by the run clock now counts as posted (2020-2024 byte-identical); (2) a team on a bye in the target week got a wrong
  next-report flag: the future schedule is now passed (2020-2024 byte-identical); (3) LightGBM fits depend on training-row order (a catch-rate model moved by up to 0.03): fits now join rows as
  the walk-forward harness does; (4) a traded player was predicted for both teams: a player on any chart is not also a recent player of his old team (a stale chart can still list him).
  Notes: the live spread is the seeded simulated mean (backtest mode, 2,000 draws), so it carries the same fixed-seed Monte Carlo wobble as the validated layer (SD about 0.3); a game with no
  margin SD yet gets no 4b record and no phase-3 fallback. The 4a/4b sub-models (status, shifts, early exit; margin scaling terms and SD) are fit on games before 2025-03-01 only.
- 2026-10-12, Phase 5 status (built test-first; nothing of it has written a real prediction yet). **Built:** the prediction store (refuses and logs a started game or a missing
  mean, append-only versions, identical rerun writes nothing, gradable = newest version stored before kickoff, schema allowlist, lines in their own log); grading
  (idempotent, did-not-play separate); the scoped 2025+ input-history unlock with its guardrail tests; lineups for 2025+ games from timestamped charts; the scoring core
  (live/predict.py) whose replay of 2024 week 10 reproduces the walk-forward harness exactly (866 + 42 rows, < 1e-9); live targets (97.6% of the standard eligible
  set, plus eligible players who did not play); the weekly run (live/weekly.py, run_weekly.py) and publishing to data-snapshots through a temporary worktree.
  **Checks on real data:** a dry run on the 2026-10-07 pull gave 1,180 predictions for the 15 games of week 5 (13 markets, none null); a fresh-container bootstrap
  (schedules, player stats, depth charts, play-by-play for 2020-2026: about 50 seconds) gave predictions identical to the main database's (max difference 0.0) for the
  one game still unplayed (BUF @ LA). **Not yet done:** (1) the weight-1 layers (4a for rush_att, rush_yds, targets, rec, rec_yds; 4b for spread) are not in the live
  run, so every stored prediction is model_version "phase3" until they are wired as new versions; (2) the scheduled routine for the Friday and Sunday-morning
  runs is not created (needs your go: it publishes to data-snapshots unattended); (3) quantiles, ladder, confidence stay null (Phase 6); (4) the weekly summary
  against the best Phase 2 baseline on the same rows; (5) the weather and injury-status snapshot columns.
- 2026-10-12, 4d.1 coaching prior, the fuller table (thin table plus the 10 hand-sourced play-calling stops of data/play_caller_stops.csv, confidence
  high or medium, rule 5a). Two runs byte-identical (results sha256 4d6ef4d70b8ba999..., data fingerprint 80bf534e6b9d18dd); coach_ablation_results_eligibility_v2_fuller.parquet
  and coach_stops_fuller.parquet (113 stops: 103 head-coach stops + 10 play-calling). Team-games with a prior: 270 (thin: 131); new coach without one:
  395 (thin: 534); W1+W2 games 985, post-change 260. **No market clears: all 13 weight 0, flagged, markets kept.** Thin to fuller on W1+W2: no gain interval
  excludes zero on the positive side in either run; EDGE in pass_att, pass_cmp, rush_att, rush_yds, targets, rec, rec_yds and moneyline (gains of
  +0.006 pass attempts [-0.073, +0.076], +0.003 rush attempts [-0.019, +0.025], +0.0035 receiving yards [-0.028, +0.036]); pass_yds, qb_rush_att
  (-0.049 [-0.092, -0.003]), qb_rush_yds, spread (-0.023 [-0.152, +0.108]) and total are NO. Spread, significantly worse in the thin run (-0.10 [-0.21, -0.01]),
  is no longer significant (the shuffled control gives -0.10 in the fuller run too: it was the noise level, not the prior). The null control
  gives gains of similar size to the real ones. Read as the plan says: inconclusive about play calling (only 9 of 36 missing coaches had a
  play-calling stop to add: 27 of the 45 changes in 2020-2024 still have no prior), not a verdict on coaching. The coaching columns stay out of the V1 weights.
- 2026-10-12, 4e weather ablation, run1 (observed weather degraded to forecast quality; gate as fixed in the plan). Two runs byte-identical (data
  fingerprint 80bf534e6b9d18dd); weather_ablation_results_eligibility_v2_run1.parquet. 1,343 backtest games, 446 indoor (neutral, flagged), 133 severe
  (outdoor, wind >= 15 mph or <= 32 F). **No market CLEARS under A1: all 13 weight 0, flagged, markets kept.** Under A1 on the severe rows pass_yds
  has the largest gain (+1.43 yards [-0.08, +3.03], EDGE), pass_att, pass_cmp, rec_yds and rush_yds are EDGE, rush_att / targets / rec / qb_rush_*
  are NO. Under A2 only pass_yds clears (+1.81 [+0.32, +3.37]); under A2x2 pass_cmp, pass_yds and rec_yds clear; with exact weather none clears.
  A layer that clears at the doubled noise but not at the exact values is a sign of bootstrap noise, not signal; the rule needs all three modes, so
  nothing passes and the thresholds are not touched. The null control (the weather columns shuffled) is flat (pass_yds +0.44 [-1.05, +2.07]).
  **Read as the plan says: flat under degraded weather is inconclusive, not a verdict on weather.** 133 severe games (14-33 per season) is low power
  (qb_rush has 33 rows). Re-test on the recorded forecasts after a season of forecast-time records.
- 2026-10-12, 4d.1 coaching prior, first run (the thin table: head-coach stops only). Two runs byte-identical (results ac8bdfa7bb884b80); both then
  crashed on a path-building slip while writing the predictions file (fixed test-first), so the predictions file of this run was not saved; the
  results file and coach_stops.parquet are. 2,686 team-games: 131 with a prior (9 changes), 534 new coach without one; W1+W2 985 games, 260
  post-change. **No market clears: all weight 0, flagged.** EDGE in pass_cmp, rush_att, rush_yds, rec and rec_yds with intervals across zero
  (rush_att +0.0065 [-0.015, +0.028]); NO elsewhere; spread is significantly worse on W1+W2 (-0.10 points [-0.21, -0.01]). The null control gives
  gains of the same size. Read as the plan says: inconclusive about play calling, not a verdict on coaching.
- 2026-10-12, scan of the gitignored files (data/ is ignored as a whole, plus .env, .claude/ and caches).
  - **Hand-made inputs:** only data/coaches_seed.csv, the 2026 staff, now tracked (24d7477). data/coaches_history.csv is generated and tracked too.
  - **Rebuildable by scripts, left ignored:** data/raw/raw.duckdb (462 MB, the nflverse and Meteostat pulls), data/processed (218 MB), the
    DuckDB tables derived from them, data_audit.json.
  - **Not re-obtainable as they stood:** every raw table is one pull, stamped 2026-10-07. Historical seasons can be pulled again, but upstream
    corrections would change the bytes (the data fingerprints would move). What cannot be pulled again at all is anything that is only true
    as of a moment: the 15 stored 2026 weather forecasts and the 2026 injury reports, depth charts and rosters as of 2026-10-07. The forecast
    recorder keeps its snapshots in the tracked records/weather_forecast_log.csv for this reason. The 2026 injury / depth-chart / roster
    snapshots are not saved anywhere outside the raw database.
  - **Not committed on purpose:** .env (a contact string for the weather API) and .claude/settings.local.json (local permissions).
- 2026-10-09, 4a ablation rerun on the current code (run_injury_ablation.py --tag rerun_20261009). Two runs byte-identical (results
  b5a9ff05a5fe5b52, predictions 1c6805ccef66799d). **The rerun is identical to the first run (335752f)**: every result column except the new
  weight column, and all 89,599 predictions. The role-share fixes made after the first run sit on the 4b simulation path (role-aware baselines,
  a promoted backup taking the role's average share); the 4a injury features never call that code, so the first run was not understated.
  - Final 4a weights under the one rule (layer_weights.csv, source the rerun file): rush_att, rush_yds, targets, rec, rec_yds CLEARS -> 1.
  - pass_att, spread and moneyline are EDGE -> 0 (the provisional values confirmed, flag removed); everything else NO -> 0.
  - Items 1-3 before Phase 6 are done. 4d.0 is listed (docs/coaching_sources_4d0.md) and stopped for review.
- 2026-10-09, per-unit threshold sweep (the searches per unit / pair, healthy vectors, count weights, recency_weighted window). Two runs
  byte-identical (rows 1c0939f3d67d32a8); the summary under the tightened rule was recomputed twice from the rows (identical):
  comp_threshold_sweep_perunit.parquet. **No threshold passes.**
  - The per-unit design matches far more: 287,671 scored rows at 0.40 against 71,816 for the averaged design.
  - At every threshold the comps beat random pairing (0.40: +0.0056 [+0.0040, +0.0071]) but lose to a zero shift (0.40: -0.0154 [-0.0177,
    -0.0132]; 0.70: -0.0119 [-0.0155, -0.0081]). Every search, both sides, has an error above the zero shift's, except S5 volume at 0.40
    (0.937 vs 0.944, a difference random pairing nearly matches).
  - Unit diagnostic at 0.70 (comp_unit_diagnostic_recency_weighted.parquet, Bonferroni over 115 cells): 0 WORKS, 28 EDGE, 87 NO. The
    best-looking EDGE cells rest on 1-3 rows.
  - **Stopping rule met** (the fingerprint diagnostic failed and no threshold passes): **the comparables are parked at weight 0 for V1.**
    SIM_THRESHOLD stays 0.70. Nothing is deleted. The layer stays built and tested, and is re-tested after later layers. Next: 4d.0.
- 2026-10-09, fingerprint diagnostic (as fixed above). Two runs byte-identical (comp_unit_signal.parquet 633b7cf5c7139b40).
  **By the fixed rule, every unit fails:** no unit's 20 fingerprint neighbours beat the unit's own trailing average on its single-game headline
  (Bonferroni intervals all include or lie below zero; rb_rotation, receiver_usage and coverage_mix are significantly worse). Reported
  alongside, not gated: against RANDOM neighbours the fingerprint neighbours are better with a 95% interval above zero for 5 of 9 units
  (run_offense, pass_offense, rb_rotation, ol_protection, coverage_mix), so teams that look alike on a unit do behave alike. They never beat
  the team's own recent average, which already carries that information: the fingerprints are informative but redundant with the team's own
  history, rather than empty.
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

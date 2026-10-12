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

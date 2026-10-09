# Overnight plan (written while you sleep; updated as each step finishes)

You asked me to keep going one phase at a time, and not to start a phase until the previous one is bug-proof. This file is the plan and the
log. Every step is committed and pushed as it finishes. Standing rules I keep: the pure track (no sportsbook lines); 2025 stays locked (every loader
is capped at 2024); raw pulls stay append-only; results are byte-identical on rerun before they are committed; older result versions are never
overwritten; no market is dropped; no threshold is tuned to flip a verdict; no PRs.

## Where I stop, and why
The repo and your messages contain the specs for 4c.0-4c.3 only. The Phase 4 layering message names 4d = coaching priors and 4e = weather, but the
texts of 4c.4, 4c.5, 4c.6, 4d and 4e were never pasted. I won't invent those specs and build on them. Section D drafts proposals for 4c.4-4c.6 for you
to approve or replace. More important, 4c.3 has hit a gate-level finding: at SIM_THRESHOLD = 0.70 almost every search returns no_match (see C). If
that stands, 4c.4-4c.6 would only compute shift = 0, so the calibration decision is yours before anything is built on top of it.

## A. Finish and verify 4c.3 (in progress)
- [x] Search engine built, 50 tests passing, first full search log (33,485 targets x 5 searches) run twice: byte-identical.
- [x] Bug found by that log and fixed: the archetype pool was "players with a touch". It dropped every zero-target game (12.7% of `targets`
      targets had no vector, and the pool's outcomes were biased upward). The pool is now "played (snap counts) or had a touch".
- [ ] Rebuild the vectors twice from scratch with the fix, sigma included; byte-compare; install.
- [ ] Rerun the full search log twice; byte-compare; commit `comp_search_log.parquet`.

## B. Bug-proof 4c.1-4c.3
- [ ] Line-by-line self-review of `features/comps_ledger.py` and `models/comps.py` against your 4c.1 / 4c.2 / 4c.3 texts.
- [ ] An independent review by a fresh agent that sees the code and the spec, not my conclusions.
- [ ] Every confirmed bug: a failing test first, then the fix, then rebuild only what it touches, with the byte-identical check.
- [ ] Full test suite once at the end.

## C. The no-match finding: a decision memo (analysis only; nothing changed)
- [ ] Measured on the 2020-2024 walk-forward search summary only: what share of targets would match under each option. The options are the
      threshold (your constant, tunable only on 2020-2024 walk-forward results, never per target), the cross-unit combination (mine: the mean of
      unit similarities, which the spec doesn't fix), and the recency half-life inside n_eff.
- [ ] Whether matched comparables carry signal at all: their weighted outcomes against the targets' actual outcomes, as a diagnostic, not a gate.

## D. Proposed next prompts (DRAFTS for your approval; not built)
- 4c.4 (proposed): the shift of a matched search = the weighted mean over its matches of (observed outcome - that observation's own pre-game
  expectation), shrunk by n_eff / (n_eff + SHRINK_K), with no team above TEAM_CAP_PER_SEARCH of a search's weight. Open question for you: the
  expectation for 2016-2019 pool games, which have no Phase 3 prediction (Phase 2 best baseline? a trailing mean?).
- 4c.5 (proposed): combine the five searches' shifts, weighting each by n_eff and (1 - completeness penalty), with the average team share across
  searches capped at TEAM_CAP_AVG_ACROSS_SEARCHES. A no-match search contributes 0 and widens the uncertainty. Store `retrieval_change` (healthy
  vs adjusted) for every search.
- 4c.6 (proposed): choose one window per unit and market walk-forward, then backtest Phase 3 + comparables vs Phase 3 on 2020-2024 with the
  CLEARS / EDGE / NO standard. Full weight only on CLEARS; EDGE and NO stay at weight 0, flagged, not dropped.
- 4d (coaching priors) and 4e (weather): need your prompts.

## Log
(entries added as steps finish)

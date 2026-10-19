# Comparables, version 2 (started 2026-10-19)

Asked for by the project manager after week 6: the whole comps layer (Phase 4c) rebuilt, plus the role-change case (Braelon Allen with Breece Hall out)
and the played-hurt case (Jalen Coker), and "keep trying structures until it works".

## The rule that makes "until it works" honest
- **Practice seasons 2021-2023**: any number of structures may be tried here. Every try is logged in `records/tuning/comps2_ladder.csv`.
- **Exam seasons 2025 + 2026 (weeks 1-6)**: never used to choose anything. The structure is fixed before the exam runs; a structure that fails the exam
  goes back to practice, and the exam is not tuned on. 2024 is reported alongside (also not used to choose).
- The bar: Mark 3's boosted-parlay return with the comps vs without, same picker, same seeds (two groups of five), paired week bootstrap.

## What v1 showed (docs/overnight_plan.md)
Comps beat random pairing but lost to a zero shift at every threshold; no unit cleared; through the stage-2 stacker the comps block scored like a shuffled block.

## v2 structure (comps2/)
| file | what |
|---|---|
| `expectations.py` | the comp-free expectation and the actual for every player-game 2020-2026 (2020-24 walk-forward Phase 3 + 4a; 2025 stored holdout; 2026 stored retro + live); `rel` = actual vs expected as a fraction |
| `fingerprints.py` | per team and game, offense and defense (allowed) unit stats from play-by-play (run 16 stats, pass 22, pace 3), recency-weighted over EARLIER games only, z against the same week's league |
| `archetypes.py` | per player, the same for his profile (RB 8 stats, WR/TE 8, QB 7), z within his family |
| `searches.py` | own (S1: his games vs similar defenses), dfn (S2: tonight's defense vs similar offenses + similar players), int (league-wide similar offense-defense pairs + similar players); kernels, recency, shrinkage, effective sample size; every search has a no-similarity control |
| `situations.py` | role change (carry / target share vacated by teammates not playing, his own share) and injury (status this week; played hurt last game; did not practice = at least doubtful) |
| `metric.py` | learns similarity weights from pairs of games: which fingerprint differences predict that two games' residuals move together |
| `residual_model.py` | the matchup model: LightGBM on the residual from both fingerprints, their mismatches, the archetype, the situation and the comp components; walk-forward by season |
| `scripts/comps2_exam.py` | the exam through Mark 3's picker |

## Practice findings so far (2021-2023)
- Correlation of each search's shift with the residual (records/tuning/comps2_ladder.csv): the defense's recent "allowed vs expectation" carries a small
  signal without similarity (0.03-0.05); league-wide interaction similarity at h = 0.75 beats its control in rush (0.039 vs 0.001), receiving (0.037 vs 0.022)
  and passing (0.035 vs -0.005); the player's own history does not (except passing, without similarity).
- Situation search (role change / hurt): rushing 0.09 with similarity vs -0.02 without, positive in every practice season; receiving 0.03 vs 0.01.
- Learned similarity weights: pair-level residual co-movement explained by fingerprint distance is no larger than with shuffled residuals (R^2 ~ 1e-4);
  what structure exists sits in player-archetype features, not team units. One game's residual is mostly noise, which is why averaging a few similar games
  (v1) could not work and why v2 moves to one model trained on all games at once.

## Residual model, practice (2022-2023; 2021 has no earlier comps)
Locked structure: the search outputs (own, dfn, int, sit: shift and n_eff) + the situation + market, family, expectation. Practice: rush corr 0.089 (both
seasons), over-rate lift top vs bottom fifth +12.8 points; receiving 0.056 / +7.9; passing 0.067 / +7.6. Adding raw fingerprints, archetypes and
mismatch columns made every group worse (rush 0.054).

## Final exam (2025 + 2026 weeks 1-6; records/tuning/comps2_exam.json; structure fixed before it ran)
| setup | seeds | exam parlays | 2022-24 parlays | exam singles |
|---|---|---|---|---|
| Mark 3 | 1-5 | +29.4% (42) | +71.3% | -3.8% |
| Mark 3 | 11-15 | +38.4% (43) | +56.6% | -11.7% |
| + comps (all outputs) | 1-5 | +6.9% (49) | +44.9% | -5.4% |
| + comps (all outputs) | 11-15 | +28.5% (58) | +48.0% | +5.1% |
| + comps (shift + adjusted gap) | 1-5 | +36.4% (35) | +36.2% | -0.3% |
| + comps (shift + adjusted gap) | 11-15 | +40.3% (42) | +24.1% | -0.9% |

Chance of beating Mark 3 on the exam weeks (paired week bootstrap): all outputs, parlays 34% / 42%; shift + gap, parlays 59% / 52%, singles 63% / 87%.
**Verdict: fails.** The lean version is level with Mark 3 on the exam (a coin flip) and well behind on 2022-2024; the full version is worse. The comps
predict the residual a little (practice) but not enough to change which parlays win. Mark 3 stays as it is. Back to practice for the next structure.

## Comps scored against the line (comps2/line_comps.py, scripts/comps3_exam.py; 2026-10-19)
Practice (2022-2023, outcome = actual / line - 1): the defense's recent results vs the line, no similarity: receiving corr 0.079 (both seasons >= 0.078),
over-rate lift +6.4 points; rushing 0.044 / +9.1. Similar offense-defense pairs vs the line (h 0.75): receiving 0.048. Own history and situations vs the
line: flat or negative. Locked: two setups, fixed before the exam.

| setup | seeds | exam parlays | 2022-24 parlays | exam singles | 2022-24 singles |
|---|---|---|---|---|---|
| Mark 3 | 1-5 | +29.4% | +71.3% | -3.8% | +6.3% |
| Mark 3 | 11-15 | +38.4% | +56.6% | -11.7% | +2.4% |
| + defense vs line + matchup vs line | 1-5 | +20.3% | +25.6% | +0.8% | +4.7% |
| + defense vs line + matchup vs line | 11-15 | +32.9% | +31.4% | -4.3% | +7.7% |
| + defense vs line | 1-5 | +86.3% | +42.0% | +10.5% | +9.2% |
| + defense vs line | 11-15 | +18.5% | +65.6% | -0.8% | +10.5% |

Chance of beating Mark 3 (paired week bootstrap), defense vs line: parlays exam 86% / 34%, 2022-24 10% / 70% -> seed-dependent, NOT a pass for parlays.
Singles exam 88% / 80%, 2022-24 68% / 93% -> better in all four cells. Defense + matchup: parlays worse (2022-24 1% / 13%).
**Verdict:** parlays unchanged (Mark 3 stays the bet). The defense-vs-line input is the first change that improves STRAIGHT bets consistently
(exam average about +5% vs -8%; 2022-24 about +10% vs +4%). Tracked from week 7 as a straight-bet candidate, not bet until live weeks agree.

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

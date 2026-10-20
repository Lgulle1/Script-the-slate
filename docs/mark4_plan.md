# Mark 4 (started 2026-10-19)

Marks 1, 2 and 3 are frozen exactly as they are and keep being tracked side by side every week. Mark 4 is built from scratch around one goal:

**Know these players better than the sportsbook does.** Not "predict the stat", but find where FanDuel's line is systematically wrong, the way
Moneyball found what the market undervalued.

## Rules for Mark 4 (the project manager's, 2026-10-19)
1. **Recent seasons count more.** The game changes. Training and judging both weight a game by how recent it is (half-life about one season), so 2025-26
   count most and 2021 least. Every result is shown both recency-weighted and season by season.
2. **Every candidate model gets its own betting-strategy tournament** (straights, parlays, 2 or 3 legs, markets, sides, edge cutoffs, boosts); nothing is
   carried over from Mark 3's cadence.
3. **Everything visible.** The Lab page shows every number: trends by season, where the book misses, every test and its result, each Mark's record.
4. Exam discipline stays: structures are chosen on earlier seasons; a fixed choice is then checked on later weeks it never saw.

## Steps
1. Lab page (records/lab/): trends, blind spots, tests, Marks; recency-weighted.
2. Blind-spot map (mark4/blindspots.py): every FanDuel line 2021-2026 tagged with its situation (role change, injury, QB change, new coach, rest, weather,
   primetime, divisional, game script, line vs recent form, streaks, price movement, defense vs line); recency-weighted hit rate and return per tag, market
   and side; season-by-season persistence; shrunk estimates (many tags are tried, so small samples are pulled toward the base rate).
3. Book lag: after a role change, how many games until FanDuel's line catches up.
4. Recency-weighted training of the picker and the projections vs equal weights.
5. Strategy tournament per candidate, recency-weighted scoring.
6. Case files on every pick.

## Strategy tournament and Mark 4i (2026-10-20)

### What changed in how strategies are judged
- **First pass (mark4/tournament.py `run`)**: 4,400 formats per Mark (straights, 2/3/4-leg parlays, 1/2/4/8 a week, pools of 6-16 legs, pairing,
  same/mixed markets, unders/overs/either side, 4 or 7 markets, edge bars 0-8%, boosted or not). Picking the best past format before each season
  (nested) lost money for every Mark: with ~70 parlays a season, a format's past record is mostly luck, so the pick chases last year's lucky format.
- **All-combo scoring (`combo_ev`)**: a parlay format is scored by the average of EVERY parlay its leg pool could build that week (different games),
  not the 1-2 it happened to build. That is the format's expected return measured on what actually happened, with far less luck.
- **New dial: how legs are ranked.** Mark 3 ranks legs by win chance. Ranking by expected value (chance x decimal price - 1) or by edge beats it for every
  Mark: the safest legs are the ones the book already prices right.

### What the dials say (every Mark agrees)
- Unders or either side >> overs (overs lose for every Mark).
- Mixing markets inside a parlay beats same-market parlays (same-market legs miss together).
- More legs = higher return per unit when the legs have an edge (a parlay multiplies the per-leg edge), and the 30% boost is worth more on longer parlays;
  the price is variance (4-leg seasons range from -26u to +75u).
- Pool depth 8-12 legs a week is best; 4 is too thin, 16 dilutes.

### Mark 4i: Mark 3 + Mark 4h + FanDuel's own price
p = b x (wm x Mark 3 + (1 - wm) x Mark 4h) + (1 - b) x FanDuel's no-vig chance (pre-kickoff price only). Both models alone are overconfident; leaning on
the price fixes it. wm and b are chosen before each season from earlier seasons only (records/lab/mark4i_weights.json: book weight 0.5-0.7).
- Log loss vs the book (positive = knows more than the price), per 1,000 lines, forward: 2023 +0.52, 2024 +0.17, 2025 +0.76, 2026 +0.36.
  First version ever to beat the price in every forward season (Mark 3: -0.18, -0.96, +1.18, -0.35).

### Head to head, paired week bootstrap vs Mark 3's rule (records/lab/strat_h2h.csv, all-combo, through 2026 week 6)
| format (EV-ranked, either side, top 12 legs, 7 markets) | boosted return | no boost | seasons up (boosted) | chance better than Mark 3 rule (boosted) |
|---|---|---|---|---|
| Mark 3 rule (2-leg, top 8 unders by chance, 4% edge) | +9.0% | -11.0% | 4/5 | - |
| Mark 4i 2-leg | +64.1% | +32.9% | 5/5 | 99.8% |
| Mark 4i 3-leg | +65.1% | +30.0% | 4/5 | 96.7% |
| Mark 4i 4-leg | +97.9% | +54.0% | 4/5 | 92.0% |
| Blend (Mark 3 + 4h, no book) 2-leg | +43.4% | +16.2% | 5/5 | 98.0% |
| Mark 3's own probabilities, EV-ranked 2-leg | +29.6% | +5.0% | 5/5 | 94.0% |

### Real cards, played week by week (mark4/cards.py, records/lab/cards.csv, through 2026 week 6)
| card | return all seasons | recency-weighted | seasons up | worst drawdown | 2026 wk 1-6 | chance better than Mark 3 |
|---|---|---|---|---|---|---|
| Mark 3 rule (frozen) | +14.8% | +34.8% | 3/5 | 20.4u | +10.6u | - |
| Mark 4i: 4 boosted 2-leg | +15.9% | +44.7% | 4/5 | 21u | +17.2u | 61% |
| Mark 4i: 4 boosted 3-leg | +23.0% | +50.0% | 4/5 | 25u | +17.1u | 61% |
| Mark 4i: 4 boosted 4-leg | +70.3% | +189.7% | 4/5 | 41u | +74.6u | 93% |
| Mark 4i: 2 boosted 2-leg + 2 boosted 4-leg | +37.6% | +123.7% | 3/5 | 36u | +60.4u | 85% |
| Mark 4i: top 10 straights only | +0.3% | +10.1% | 3/5 | 24.6u | +12.0u | 9% |

Straights alone still do not beat the vig enough to matter; the edge is in boosted multi-leg parlays built from EV-ranked legs.

### Markets
- Passing TDs (mark4/pass_td.py): FanDuel prices them well. The model alone loses to the price every season 2021-2026; mixed with the price it gains a
  little, but bets break even (+2% at any edge, -0.5% at 2%+). Not added to the pool; kept for retest.
- Rush + receiving yards: FanDuel has no such line in any of the exports, so it cannot be tested.
- Anytime TD (scripts/td_model.py, earlier): the recalibrated book price alone beats the model's log loss. Not added.

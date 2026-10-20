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

Correction (2026-10-20): the "worst drawdown" column above is the worst drop WITHIN one season. Over the whole history (records/lab/strat_cards.csv):
| card | worst losing run | when | bets in it | winning weeks |
|---|---|---|---|---|
| Mark 3 rule (frozen) | -21.3u | 2022 wk 12 to 2023 wk 15 (22 weeks) | 79 | 37 of 78 |
| Mark 4i: 4 boosted 2-leg | -22.0u | 2022 wk 2 to 2023 wk 13 (30 weeks) | 110 | 35 of 73 |
| Mark 4i: 4 boosted 3-leg | -25.1u | 2023 wk 18 to 2025 wk 9 (25 weeks) | 64 | 17 of 73 |
| Mark 4i: 2 boosted 2-leg + 2 boosted 4-leg | -36.0u | 2022 wk 2 to 2022 wk 10 (9 weeks) | 36 | 24 of 73 |
| Mark 4i: 4 boosted 4-leg | -60.0u | 2023 wk 3 to 2024 wk 14 (30 weeks) | 60, all lost | 11 of 73 |
The 4-leg card's +116.7u rests on 11 winning weeks; it went 60 parlays without a hit. It is the highest return per unit and the least proven.

Straights alone still do not beat the vig enough to matter; the edge is in boosted multi-leg parlays built from EV-ranked legs.

### Markets
- Passing TDs (mark4/pass_td.py): FanDuel prices them well. The model alone loses to the price every season 2021-2026; mixed with the price it gains a
  little, but bets break even (+2% at any edge, -0.5% at 2%+). Not added to the pool; kept for retest.
- Rush + receiving yards: FanDuel has no such line in any of the exports, so it cannot be tested.
- Anytime TD (scripts/td_model.py, earlier): the recalibrated book price alone beats the model's log loss. Not added.

### Balancing return against losing streaks: stake by parlay length (mark4/portfolio.py, records/lab/portfolio.csv)
Same Mark 4i legs; long parlays staked smaller. History 2022 to 2026 week 6, played week by week:
| portfolio (4 boosts a week) | won | return per unit | worst losing run | parlays hit |
|---|---|---|---|---|
| A. 4 x 2-leg @ 1u | +38.2u | +16% | -22.0u | 49 of 240 |
| B. 4 x 4-leg @ 1u | +116.7u | +70% | -60.0u | 11 of 166 |
| C. 4 x 4-leg @ 0.25u | +29.2u | +70% | -15.0u | 11 of 166 |
| E. 2 x 2-leg @ 1u + 2 x 4-leg @ 0.25u | +34.7u | +23% | -22.5u | 30 of 209 |
| G. 2 x 2-leg @ 1u + 1 x 3-leg @ 0.5u + 1 x 4-leg @ 0.25u | +41.1u | +25% | -24.8u | 34 of 220 |
| Mark 3 rule (frozen), for reference | +35.1u | +15% | -21.3u | |
Simulated 18-week seasons (weeks resampled): with every season weighted equally, every card loses in about 1 season in 3-4; leaving 2026 out, about
1 in 2.5. 2026 has been the best stretch by far, especially for 4-leg parlays (3 of its 6 weeks hit), so the recency-weighted numbers flatter the long parlays.
Recommendation: G (or E): the 2-leg core keeps weekly hits, the 3- and 4-leg slots at a half and a quarter unit keep the long-parlay upside at a
fraction of the swing; worst run about the same as Mark 3's. Size the unit to the bankroll so a 30u run is survivable (unit = bankroll / 75 to / 100).

### Mix search (mark4/mix_search.py, records/lab/mix_search*.csv): 29,982 mixes, chosen forward
Every split of the 4 boosts among 2/3/4/5-leg parlays, built long-first or short-first, plain parlays on top, stakes 0.25-1u per group.
Honest test: before each season, the mix with the most recency-weighted units over EARLIER seasons is chosen and bet. Chosen forward, 2023-26:
+151u to +169u on 143-167u bet (+100%), vs +78.5u for 4 x 4-leg @ 1u and +43u for the earlier recommended mix. From 2024 on it picked the same
family every year: **1 x 5-leg + 1 x 3-leg + 2 x 2-leg, built long-first** (the best expected-value legs go into the long parlay; the 2-leg parlays
take the next legs, which are shorter-priced and hit 37% of the time vs 20% when built from the top legs).

| mix (long-first) | won 2022-26 | return | worst losing run (all) | since 2025 | every season | simulated: lose a season | 1-in-20 bad season |
|---|---|---|---|---|---|---|---|
| 1 x 5-leg @ 1u + 1 x 3-leg @ 1u + 2 x 2-leg @ 1u | +243.9u | +121% | -12.4u | -2.0u | -10.3, +113.9, +2.1, +65.9, +72.3 | 9% | -6.4u |
| 1 x 5-leg @ 0.5u + 1 x 3-leg @ 1u + 2 x 2-leg @ 1u | +179.4u | +103% | -8.6u | -2.0u | -1.3, +77.3, +10.6, +43.6, +49.2 | 5% | +0.2u |
| **1 x 5-leg @ 0.25u + 1 x 3-leg @ 0.5u + 2 x 2-leg @ 1u** | **+128.8u** | **+98%** | **-5.8u** | **-1.5u** | **+12.2, +49.8, +6.3, +29.4, +31.1** | **2%** | **+5.1u** |
| 4 x 4-leg @ 1u (reference) | +116.7u | +70% | -60.0u | -5.0u | +38.2, +15.5, -26.2, +14.5, +74.6 | 25% | -31.0u |
Caveats: the 5-leg slot hit 3 times in 55 (one each in 2023, 2025, 2026), so its share of the profit rests on 3 hits; at 0.25-0.5u a cold run of
5-leg parlays costs little. Mark 4i bet thinly in 2025 (about 4 qualifying legs a week: it leaned 70% on the book that season), so "since 2025"
runs are measured on few bets. Simulated seasons with 2023 and 2026 (the two best) removed: the 0.25u version still loses only 9% of seasons.

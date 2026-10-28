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

### Role-change engine (mark4/rolechange.py, 2026-10-20)
Pre-game only (roster + final injury report, announced starting QBs, earlier usage). Per player-game: his carry / target share when playing, vacancy
left by OUT regulars of his role, his rank among the available, competition, teammates back, games since he last played, game N of the vacancy,
QB change, new team. A forward model (each 6-week block fitted on earlier games, 2018 on) predicts his new share.
- New share: 5-7% more accurate than his old share (2019-2026, forward), in role-change games too.
- As a full projection (share x team volume x efficiency) it is LESS accurate than FanDuel's line in every market, and betting its disagreements loses
  about the vig: FanDuel's lines already move for role changes, and the engine leans over (60%) where overs lose.
- **The edge: FanDuel overreacts to promotions.** Compare how far the line jumped (this line / his previous line) with how far the engine says his
  role grows. Vacancy games 1-3, line moved 35%+ more than the role: under hits 58.0%, +7.2% (369 bets; game 1 only: 60.4%, +12.3% on 187, 5 of 6
  seasons up). Dose-response: 10% overreaction -5%, 20% -1%, 35% +12% (game 1). Chasing the promotion (over) loses 10-15% everywhere.
- Mark 4k (role inputs fed to the picker): worse log loss 2022-24 (diluted among 50 inputs). Not kept.
- **Mark 4l = Mark 4i + overreaction unders** (the under's chance = the hit rate of such lines in earlier seasons only, shrunk with 100 lines; 54-56%).
  All-combo boosted return by season, 2022-2026:
  | legs | Mark 4i | Mark 4l |
  |---|---|---|
  | 2 | +14 +79 +9 +120 +63 | +16 +49 +20 +54 +57 |
  | 3 | +18 +179 -22 +83 +88 | +20 +84 +25 +121 +72 |
  | 4 | +27 +118 -52 +312 +122 | +28 +149 +27 +219 +76 |
  | 5 | +43 +288 -79 +1177 +182 | +42 +166 +22 +279 +66 |
  Mark 4l is positive in EVERY season at every parlay length (Mark 4i lost 2024 for 3-5 legs); fewer giant seasons (Mark 4i's 2025 rests on ~4
  qualifying legs a week). More qualifying legs: 2025 6.0 a week vs 3.9. Log loss vs the book: 2023 +0.59, 2024 +0.02, 2025 +0.75, 2026 +0.57.
- Caution learned: the real-card mix results (e.g. +128.8u for 5/3/2/2) swing a lot when the leg pool changes slightly (Mark 4l reshuffles which
  legs go into the 5-leg: +113.6u at 0.25u, but +134.6u at 1u vs +243.9u). Those single-path numbers are fragile; all-combo is the steadier guide.

### Edge hunt round 2 (2026-10-20): what survived the forward test and what did not
Rule learned the hard way: a pattern that looks consistent across 6 seasons after slicing lines many ways is often luck; only a rule chosen from
EARLIER seasons and then bet on the next one counts.
- mark4/hunt.py (descriptive, every line 2021-26): overs are overpriced almost everywhere (every under about -2 to -3%, every over -7 to -9%).
  Hot players (beating their lines) get overpriced (6/6 seasons); defenses whose opponents go under keep doing so (6/6); Thursday leans over (4/6).
- Comps v5, comps of the BOOK's behaviour (mark4/context.py: kNN and gradient boosting on line context: line vs form, line jump, last game vs line,
  player / defense record vs the line, role change, game script, price shape): forward, worse than the book's own price; betting their predicted errors
  loses 3-4% (they mostly rediscover "overs a bit overpriced", which is smaller than the vig).
- Shape (line vs his median of last 12): descriptively unders win when the line sits 5-30% above the median (receptions 6/6, rushing 4/5). Forward:
  as a probability correction (Mark 4m) the weight chosen from earlier seasons is mostly 0 and log loss does not improve; as a rule (bet the unders of
  groups profitable in earlier seasons) -4.3% on 3,832 bets. Look-back illusion. Not kept.
- Promotion overreaction, cutoff chosen forward each season: games 1-3 +4.5% on 312 bets (3 of 5 seasons up; the 35% cutoff chosen every year);
  game 1 only -1.7%. Smaller than the +7.2% picked with hindsight, still positive: kept in Mark 4l.
- Kelly-style stakes per parlay (Mark 4l, 5/3/2/2 mix): same return per unit (+70-72%), more money in, worst run grows with it (fixed 18.7u on +106.8u;
  quarter-Kelly-ish 35.7u on +256.1u). Stake size is a risk choice, not an edge.
What still stands: Mark 4l (Mark 3 + 4h + FanDuel's price, + forward-tested promotion-overreaction unders), legs ranked by expected value, either side,
all markets, boosted parlays built long-first. Mark 4l 5/3/2/2 at 0.25/0.5/1u: +106.8u on 151.8u 2022-26, up every season (+13, +22, +12, +25, +35).

### Reusing legs and adding straights (mark4/reuse.py, records/lab/reuse_live.csv, 2026-10-22)
The PM's question: put the best legs in a short parlay too, or bet them straight, instead of using each leg once. Same Mark 4l legs, the live card's
own construction, 2022 to 2026 week 6:
| strategy | won | bet | return | worst run | chance better than now |
|---|---|---|---|---|---|
| now: 5/3/2/2, each leg once | +101.3u | 151.8u | +67% | 18.7u | - |
| 5-leg's top 2 also as a 2-leg | +79.9u | 174.8u | +46% | 14.8u | 2% |
| nested (5-leg, its top 3, top 2, legs 3+4) | +54.8u | 187.8u | +29% | 30.2u | 0% |
| now + every card leg straight @ 0.5u | +108.2u | 513.2u | +21% | 24.7u | 0% |
| now + every card leg straight @ 1u | +115.1u | 874.8u | +13% | 31.5u | 0% |
| 5-leg + its legs straight, no other parlays | +4.4u | 345.2u | +1% | 20.7u | 0% |
Reusing legs puts several boosts on the same few outcomes; straights earn only the legs' 1-3% edge without the boost (+7 to +14u on 360-720 more units bet).
The 4 boosts are worth the most spread over different legs. Kept: each leg once. Also learned: building the same card a slightly different way (one leg
per game across the whole card vs per parlay) moved the history from +101u to +38u, so these single-path numbers are fragile.
Week page: a unit box (1 unit = $X) shows each parlay's stake in dollars and what it wins with the boost.

## 2026-10-22: Mark 4i is the bet, Mark 4l dropped (before week 7's first game)

Mark 4l = Mark 4i plus promotion-overreaction unders, which replaced the model's chance on flagged lines with the earlier-season hit rate (about 55.8%),
so those legs went to the top of every card. Tested on the real 5/3/2/2 card, 2022 to 2026 week 6, every card on the same seasons (mark4/cards.py):

| Overreaction strength (mark4/nudge.py) | Units | Staked | Return | 2025 on |
|---|---|---|---|---|
| 0 (Mark 4i) | +128.8 | 131.8 | +98% | +60.4 on 22.8 |
| 0.25 | +96.3 | 138.5 | +70% | +37.1 on 23.0 |
| 0.5 | +80.5 | 147.8 | +54% | +35.1 on 28.5 |
| 0.75 | +83.9 | 149.8 | +56% | +28.7 on 29.5 |
| 1 (Mark 4l) | +101.3 | 151.8 | +67% | +54.6 on 31.5 |

Chosen forward from earlier seasons, the strength is 0 for 2025 and 2026. Every parlay the legs could build (2022-2026, 2021 masked): 4i beats 4l at
2 legs (90%) and 5 legs (91%; 97% on 2025-26), level at 3 and 4. 4l's case was steadiness (up every season at every length; 4i lost 2024 at 3-5 legs).
Not proof (under 95%), but every test that measures money on this card points the same way. The week 7 4l card is kept as
records/cards/2026_wk07_card_mark4_v1_4l_20261022.csv; the live card is now Mark 4i (mark4/live.py VERSION).

## 2026-10-23: per-market settings and de-vig method tested (mark4/per_market.py): all rejected; leg-level proof

One-size settings audited: the model-vs-book weight, the Mark 3 vs 4h weight, a bias correction, and the de-vig method. Each chosen forward per season
(earlier seasons only, recency-weighted log loss), played on the real 5/3/2/2 card 2022 - 2026 week 6 and leg by leg (the week's top-12 legs, straight, at price):

| Variant | Card | Legs | Chance better than Mark 4i (card / legs) |
|---|---|---|---|
| Mark 4i (one setting for every market) | +128.8 on 131.8 | +1.41% on 666 | - |
| per-market weights | +39.1 on 170.5 | -0.83% on 806 | 1% / 17% |
| per-market weights + bias | +35.6 on 213.5 | -5.38% on 935 | 0% / 3% |
| de-vig chosen forward | same as Mark 4i: proportional is the most accurate every season | | |

Per-market weights overfit the smaller samples and trust the model more where it is not better, so more fake edges reach the card.

Leg-level proof (top-12 legs straight, weeks resampled, recency-weighted): Mark 4i makes money 95% (2025 on: +21.9% on 118 legs); Mark 4i beats Mark 3
94%, Mark 4h 91%, Mark 4l 78%.

## 2026-10-28: build order tested after week 7 (the 1u 2-leg got the weakest legs; Marks 1-2 won with their best two in a 2-leg)

Mark 4i, same stakes, 2022 - 2026 week 6:

| Build order | Units | Staked | 2025 on | Worst run | Chance better than long-first |
|---|---|---|---|---|---|
| long-first 5 / 3 / 2 / 2 (now) | +128.8 | 131.8 | +60.4 on 22.8 | 5.8 | - |
| short-first 2 / 2 / 3 / 5 | +27.3 | 165.8 | +25.3 on 32.0 | 24.8 | 0% |
| 2-legs, then 5, then 3 | +19.4 | 165.0 | +8.9 on 31.8 | 20.6 | 0% |
| one 2-leg first, then 5 / 3 / 2 | +74.8 | 153.5 | +42.4 on 28.5 | 16.5 | 2% |

Long-first stays. The best legs belong in the longest parlay: a parlay's edge compounds leg by leg and the 30% profit boost is worth most on the
biggest payout. Long-first also stakes less in thin weeks (only the 0.25u 5-leg fills), which are the weak weeks.

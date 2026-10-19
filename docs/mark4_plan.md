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

# Lineup-adjusted target vectors: what they add (analysis only; nothing was changed)

**What you need to decide:** how the lineup-adjusted target vectors of 4c.1.4 should be built. Every comparable search runs on them, per the
plan. On 2020-2024 they describe the coming game worse than the plain (healthy) vectors do. Nothing below changed any stored vector: the searches,
the no-match log and 4c.4 all still run on the vectors as built.

Found while reading comparable cards (the plan's manual step). The worst card: SF, 2024 week 16, rushing. The adjusted vector has the RB target share
at 0.000 (z = -4, the clip), against 0.19 in the healthy vector. In the game itself it was 0.053.

## 1. How the adjusted vector is built today
For a team-game the 4a layer gives each listed player a baseline share (b_*, his healthy-lineup share) and an expected share for this game
(exp_*, after his status and the redistribution). The adjusted vector is

    healthy window value + sum over players of (exp - b) x (his value - the team's value)

(and, for the share-structure features, the structure evaluated on exp minus the structure on b). The healthy window value is the team's own recent
form, so it already reflects anyone who has been out for weeks. The 4a baseline does not: it still counts that player at full strength. His absence
is therefore subtracted a second time. At SF in 2024, McCaffrey (out most of the season) still carries a 0.14 target share and a 0.29 carry share
in the 4a baseline. That is how 0.19 - 0.22 ended below zero.

## 2. Measurements (2020-2024, the 2,686 team-games with 4a expectations, recency_weighted window)
All numbers are in league standard deviations of the window values. Variants compared:
- **now**: the current construction.
- **window**: the 4a expected shares minus the window's normal shares.
- **rebased**: each player's window share x his 4a expected / baseline ratio.
- **participation**: the pool's own "who actually played" recipe, with the 4a play probability in place of who played.

**a. Against the pool's own "who actually played" vector** (what the pool rows look like), mean absolute distance:

| unit | healthy | now | window | rebased | participation |
|---|---|---|---|---|---|
| pass_offense | 0.358 | 0.494 | 0.295 | 0.485 | 0.239 |
| rb_rotation | 0.760 | 0.774 | 0.644 | 0.616 | 0.368 |
| receiver_usage | 0.878 | 0.710 | 0.638 | 0.621 | 0.403 |
| run_offense | 0.225 | 0.222 | 0.234 | 0.192 | 0.117 |

The current construction moves pass_offense away from the pool's vector (0.494 against 0.358 for doing nothing). This is the double counting.
"Participation" is closest to the pool's vector everywhere, partly because it is built the same way.

**b. Against what happened in the game** (that game's own value of each feature), root mean square error:

| unit | healthy | now | window | rebased | participation |
|---|---|---|---|---|---|
| pass_offense | 2.657 | 2.725 | 2.716 | 2.791 | 2.703 |
| rb_rotation | 1.882 | 2.107 | 2.189 | 2.274 | 2.139 |
| receiver_usage | 2.208 | 2.407 | 2.549 | 2.686 | 2.475 |
| run_offense | 2.331 | 2.345 | 2.360 | 2.368 | 2.356 |

The healthy vector is best for every one of the 29 lineup-corrected features. In the 10% of games with the largest lineup change, the gap widens.
For pass_offense: healthy 2.86, now 3.30, participation 3.20. For the top-3 target share: healthy 2.30, now 4.32, participation 5.98.

**c. Calibration.** Regress the game's change (game value - healthy) on the predicted change (adjusted - healthy). A slope of 1 means
calibrated, and 0 means no information. Median slopes: pass_offense 0.02 now, 0.13 participation; rb_rotation 0.15 / 0.16; receiver_usage
0.13 / 0.16; run_offense 0.06 / -0.11. Correlations are 0.01-0.16. The predicted moves are several times too large and carry little signal.

**d. Shrinking each player's value toward his team harder** (SHRINK_K = 5 opportunities today, an approved 4c.1 default) only makes the moves
smaller. pass_offense RMSE against the game: K=5 2.725, K=50 2.671, K=200 2.659, against 2.657 for healthy. run_offense never improves. At no K does
an adjusted vector beat the healthy one.

The defense units have no lineup adjustment at all: there is no player-level decomposition of a defense. The plan's manual step expects a
"key defender out" card in which the adjusted and healthy vectors retrieve different comparables. That cannot happen with the current units.

## 3. Options (your call; I have not chosen)
A. **Keep it as built** (plan-literal). Searches run on vectors whose lineup moves are mostly noise and partly double-counted. The healthy run is
   stored next to them (4c.4: top-10 overlap, change in the shifts), so you can see what they change.
B. **Fix the frame, keep the structure**: use the "participation" construction, which is consistent with the pool rows and drops the double
   counting, and shrink harder (K around 200). The moves become small (about 0.06 SD for pass_offense) and close to harmless, but they still add
   nothing measurable.
C. **Calibrate**: adjusted = healthy + lambda x (the move), with lambda per unit estimated walk-forward on earlier weeks only (about 0.1-0.2 from
   the slopes above, 0 for run_offense). This is a new constant, tuned on 2020-2024 walk-forward results as the plan allows.
D. **Search on the healthy vectors** until the injury layer predicts unit-level change better. This deviates from 4c.1.4, which says every
   search runs on the adjusted vector.

Practical effect today is small: at SIM_THRESHOLD = 0.70 nearly every search is no_match (docs/nomatch_memo.md), so neither version moves any
shift yet. It will matter once the threshold decision is made.

Reproduce with `python run_lineup_adjustment_check.py`. It prints every table above and writes nothing. Two runs gave identical output. Inputs are
capped at 2024.

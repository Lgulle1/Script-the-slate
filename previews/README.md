# Predictions preview (2026-10-12)

NOT stored records and NOT a track record: display files made at 2026-10-12 ~15:45 UTC from the full live stack (Phase 3 + 4a for rush/receiving + 4b spread).

- `monday_night_BUF_at_LA_*` : tonight's game (kickoff 2026-10-13 00:15 UTC), real run on current data.
- `week6_preview_*` : the week-6 slate (15-19 Oct), built as of 2026-10-13 12:00 UTC before Monday night's result exists (so BUF and LA are one game stale).
- `*_games.csv` : predicted home margin (home minus away; positive = home wins by that), total points, home win probability.
- `*_players.csv` : predicted mean for each eligible player in each market he is eligible for (blank = not eligible for that market). A listed player may not play (inactive, injured): those rows are graded as 0 and flagged did-not-play once the weekly record exists.
Each number is a mean, not a line, and there are no sportsbook lines anywhere in the model.

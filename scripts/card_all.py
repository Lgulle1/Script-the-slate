"""One command for a week's card, in the order that keeps it honest (yours, 2026-10-19: week 6 was built on a Thursday injury report and never refreshed).

  python scripts/card_all.py --season 2026 --week 7 --books /path/NFL_2026_Weeks1-7_All_Books_Player_Props.csv [--skip-refresh]

  1. refresh the data and store the live predictions (run_weekly.py --refresh): new injury report, new inactives, new lines for games that have not started;
  2. stop if the injury report is older than MAX_AGE_HOURS (the refresh must have worked) and print who is out around the teams playing;
  3. warn if FanDuel lines for the week in prop_lines_log.csv are older than LINES_WARN_HOURS (send fresh ones);
  4. rebuild the line study when it lacks this week's lines or last week's results (scripts/line_study.py build), and carry the saved forward chances
     onto the rebuilt table by line (mark4/remap_forward.py; 2026-10-29: week 8's first build found no week 8 rows);
  5. build Mark 2's card (week_card.py), Mark 3's card (schedule_picker.py), Mark 4's card (mark4/live.py, the bet), the week page and the Track Record.
Run it again the morning of the games, after the inactive lists (about 90 minutes before kickoff): the page says how old the injury report is.
"""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from live import role_flags as RF

HERE = Path(__file__).resolve().parent
ROOT = config.ROOT
MAX_AGE_HOURS = 3.0
LINES_WARN_HOURS = 6.0


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"step failed: {' '.join(str(c) for c in cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--books", type=Path, required=True, help="the all-books export for the week (best prices shown on Mark 2's card)")
    ap.add_argument("--skip-refresh", action="store_true", help="rebuild from the data already in the database (still checks its age)")
    ap.add_argument("--allow-stale", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    if not a.skip_refresh:
        run([py, "run_weekly.py", "--refresh", "--no-sync"])
    age = RF.data_age(a.season, a.week)
    print("\ninjury report pulled:", age["injuries_pulled_utc"], f"({age['injuries_age_hours']:.1f} hours old)" if age["injuries_age_hours"] is not None else "(none)")
    if age["injuries_age_hours"] is None or (age["injuries_age_hours"] > MAX_AGE_HOURS and not a.allow_stale):
        raise SystemExit(f"injury report is older than {MAX_AGE_HOURS} hours or missing: refresh did not work. Not building a card on stale data (--allow-stale to override).")
    flags = RF.teammate_flags(a.season, a.week)
    print("\nregulars out or not practicing this week:")
    for team, fl in sorted(flags.items()):
        print(f"  {team}: " + "; ".join(RF.label(f) for f in fl))
    log = ROOT / "records" / "lines" / "prop_lines_log.csv"
    ln = pl.read_csv(log, infer_schema_length=0).filter(pl.col("game_id").str.starts_with(f"{a.season}_{a.week:02d}_"))
    if ln.height:
        last = ln["captured_at"].str.to_datetime(time_zone="UTC", strict=False).max()
        hrs = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        print(f"\nlatest FanDuel line for week {a.week}: {last:%Y-%m-%d %H:%M} UTC ({hrs:.1f} hours old)" + ("  <-- send fresh lines" if hrs > LINES_WARN_HOURS else ""))
    else:
        print(f"\nNO FanDuel lines for week {a.week} in the log: import the week's file first (scripts/import_bettingpros.py).")
    ls = config.PROCESSED_DIR / "line_study_sel.parquet"
    st = pl.read_parquet(ls).filter(pl.col("season") == a.season)
    have = st.filter(pl.col("week") == a.week).height
    ungraded = st.filter((pl.col("week") == a.week - 1) & pl.col("won").is_null()).height
    if have == 0 or ungraded:
        print(f"\nline study: {have} lines for week {a.week}, {ungraded} ungraded in week {a.week - 1}: rebuilding it")
        prev = ls.with_name("line_study_sel.prev.parquet")
        prev.write_bytes(ls.read_bytes())
        run([py, "scripts/line_study.py", "build"])
        run([py, "-m", "mark4.remap_forward", str(prev)])
    run([py, "scripts/week_card.py", "--season", str(a.season), "--week", str(a.week), "--books", str(a.books)])
    run([py, "scripts/schedule_picker.py", "--season", str(a.season), "--week", str(a.week)])
    run([py, "-m", "mark4.live", "--season", str(a.season), "--week", str(a.week)])          # Mark 4: the bet from 2026 week 7
    run([py, "scripts/make_week_page.py", "--season", str(a.season), "--week", str(a.week)])
    run([py, "scripts/week_page_mark4.py", "--season", str(a.season), "--week", str(a.week)])
    run([py, "scripts/make_record_page.py", "--season", str(a.season), "--live-week", str(a.week)])
    print("\ndone: records/cards/%d_wk%02d_picks.html and track_record.html" % (a.season, a.week))


if __name__ == "__main__":
    main()

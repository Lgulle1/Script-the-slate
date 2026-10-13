"""Log player-prop lines from a CSV into records/lines/prop_lines_log.csv (live/prop_lines.py; rules in docs/overnight_plan.md).

CSV columns: book, player_name, book_market, side, line  [price, note, is_alt] and the game as game_id OR team + gameday (YYYY-MM-DD).
Example row:  bookA,Puka Nacua,Receiving Yards,over,75.5,-115,,,2026-10-18,LA   (header: book,player_name,book_market,side,line,price,note,is_alt,gameday,team)

  python scripts/log_prop_lines.py lines.csv                 # logs, stamped now
  python scripts/log_prop_lines.py lines.csv --push          # also publishes the log + rejects to data-snapshots (an outside timestamp)
Lines captured at or after kickoff are refused; unmatched players / markets go to prop_lines_rejects.csv with the reason.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from live import prop_lines as PL
from live import publish
from live import targets as TG


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--captured-at", help="ISO UTC time the lines were seen (default: now)")
    ap.add_argument("--source", default="manual")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--log", type=Path, default=config.ROOT / "records" / "lines" / "prop_lines_log.csv")
    a = ap.parse_args(argv)
    if a.push:
        publish.sync_records(config.ROOT, ["records/lines"])
    now = datetime.fromisoformat(a.captured_at).astimezone(timezone.utc) if a.captured_at else datetime.now(timezone.utc)
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    pcon = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    players = pcon.execute("SELECT DISTINCT gsis_id, display_name, latest_team, position FROM players WHERE gsis_id IS NOT NULL").pl()
    pcon.close()
    rows = pl.read_csv(a.csv, infer_schema_length=0).with_columns(pl.all().replace("", None))
    if "is_alt" in rows.columns:
        rows = rows.with_columns(is_alt=pl.col("is_alt").str.to_lowercase().is_in(["true", "1", "yes"]))
    res = PL.log_prop_lines(rows, now, a.log, TG.schedule_frame(sched), players, source=a.source)
    print(f"written {res.written}, unchanged {res.unchanged}, refused {res.refused}")
    if a.push:
        files = [p for p in (a.log, a.log.with_name("prop_lines_rejects.csv")) if p.exists()]
        print("published:", publish.publish(config.ROOT, files, f"prop lines {now.strftime('%Y-%m-%dT%H:%MZ')}: {res.written} written, {res.refused} refused"))
    return res


if __name__ == "__main__":
    main()

"""Anytime touchdown market (yours, 2026-10-19): import FanDuel's `touchdown-scored` lines for 2021-2026, match the players, grade them, and show how the market prices.

The BettingPros exports already on disk (records/lines/imports) hold three touchdown markets: anytime (`touchdown-scored`), first and last. This script uses the anytime one.
In those rows the `selection` column is the player (the `player` column is a stray name from the game), `line` is 1 (one or more touchdowns) and `american_odds` is the price of "yes";
there is no "no" price, so the break-even chance is 1 / decimal odds, overround included.

Rules (the same as the over/under lines):
  - regular season, FanDuel rows only; the game is resolved on season + week + the two teams;
  - the price is the last one posted up to 300 s after kickoff; a later stamp (the feed overwrites a line once the game starts) falls back to the opening price when the file has one, else the row is dropped;
  - a name is matched inside the game's played roster (QB / RB / WR / TE with a stat line); a name that does not match, or matches twice, is left out: a player who did not play is void anyway;
  - scored = rushing + receiving + special-teams touchdowns >= 1 (passing touchdowns do not count for the passer).

  python scripts/td_import.py        # writes records/lines/td_lines_graded.parquet
"""
import glob
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from live import history as H
from live import prop_lines as PL

IMPORTS = config.ROOT / "records" / "lines" / "imports"
OUT = config.ROOT / "records" / "lines" / "td_lines_graded.parquet"
TEAM_FIX = {"JAC": "JAX", "LAR": "LA"}
GRACE = 300


def load_files() -> pl.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(IMPORTS / "*FanDuel_NFL_20*_Player_Props.csv"))):
        d = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy")
        d = d.rename({c: c.lstrip("﻿") for c in d.columns})
        for pre in ("book_opening_", "selected_opening_"):
            if "fanduel_opening_odds" not in d.columns and f"{pre}odds" in d.columns:
                d = d.rename({c: c.replace(pre, "fanduel_opening_") for c in d.columns if c.startswith(pre)})
        d = d.filter((pl.col("market") == "touchdown-scored") & (pl.col("season_type") == "REG"))
        if "sportsbook" in d.columns:
            d = d.filter(pl.col("sportsbook") == "FanDuel")
        parts.append(d.select("season", "week", "away_team", "home_team", "selection", "american_odds", "game_start_utc", "line_updated_utc", "fanduel_opening_odds",
                              "is_off", "is_active", "retrieved_at_utc").with_columns(file=pl.lit(Path(f).name)))
    return pl.concat(parts, how="diagonal_relaxed")


def main():
    d = load_files()
    print("anytime-TD rows in the files:", d.height)
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, home_team, away_team FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    ps = con.execute("SELECT DISTINCT ON (player_id, season, week) player_id, season, week, rushing_tds, receiving_tds, special_teams_tds FROM player_stats "
                     "WHERE season_type = 'REG' ORDER BY player_id, season, week, pulled_at DESC").pl()
    con.close()
    fix = lambda c: pl.col(c).replace(TEAM_FIX)
    d = d.with_columns(away=fix("away_team"), home=fix("home_team"), season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32))
    games = sched.with_columns(season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32))
    d = d.join(games, left_on=["season", "week", "home", "away"], right_on=["season", "week", "home_team", "away_team"], how="left")
    print("no game found:", d["game_id"].null_count())
    d = d.filter(pl.col("game_id").is_not_null())
    price = lambda c: pl.col(c).cast(pl.Float64, strict=False)
    ts = lambda c: pl.col(c).str.to_datetime(time_zone="UTC", strict=False)
    d = d.with_columns(last=price("american_odds"), open=price("fanduel_opening_odds"), kick=ts("game_start_utc"), upd=ts("line_updated_utc"))
    d = d.with_columns(late_s=(pl.col("upd") - pl.col("kick")).dt.total_seconds())
    d = d.with_columns(use=pl.when(pl.col("late_s").is_null() | (pl.col("late_s") <= GRACE)).then(pl.col("last")).otherwise(pl.col("open")),
                       tier=pl.when(pl.col("late_s").is_null()).then(pl.lit("pre")).when(pl.col("late_s") <= 0).then(pl.lit("pre")).when(pl.col("late_s") <= GRACE).then(pl.lit("grace"))
                       .when(pl.col("open").is_not_null()).then(pl.lit("opening (late stamp)")).otherwise(pl.lit("dropped")))
    d = d.filter(pl.col("use").is_not_null() & (pl.col("tier") != "dropped"))
    d = d.sort("upd", descending=True).unique(["game_id", "selection"], keep="first")
    print("rows with a usable price:", d.height)
    # player matching inside each game's played roster
    out = []
    for season in sorted(d["season"].unique().to_list()):
        roster = PL_roster(season)
        rm = {}
        for r in roster.iter_rows(named=True):
            rm.setdefault((r["game_id"], PL.normalize_name(r["display_name"])), []).append(r)
        for r in d.filter(pl.col("season") == season).iter_rows(named=True):
            hit = rm.get((r["game_id"], PL.normalize_name(r["selection"])), [])
            if len(hit) == 1:
                out.append({**{k: r[k] for k in ("season", "week", "game_id", "selection", "use", "open", "tier", "late_s")}, "player_id": hit[0]["gsis_id"], "team": hit[0]["team"], "position": hit[0]["position"]})
    m = pl.DataFrame(out).rename({"selection": "player_name", "use": "price", "open": "open_price"})
    print("matched to a played player:", m.height, f"({m.height / d.height:.0%} of priced rows)")
    m = m.join(ps.with_columns(season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32)), on=["player_id", "season", "week"], how="left")
    m = m.with_columns(tds=(pl.col("rushing_tds").fill_null(0) + pl.col("receiving_tds").fill_null(0) + pl.col("special_teams_tds").fill_null(0)))
    m = m.with_columns(scored=(pl.col("tds") >= 1).cast(pl.Float64), dec=pl.when(pl.col("price") > 0).then(1 + pl.col("price") / 100).otherwise(1 + 100 / -pl.col("price")))
    m = m.with_columns(implied=1 / pl.col("dec")).drop("rushing_tds", "receiving_tds", "special_teams_tds")
    m.write_parquet(OUT)
    print("wrote", OUT, m.height)
    print(m.group_by("season").agg(lines=pl.len(), scored=pl.col("scored").mean(), implied=pl.col("implied").mean()).sort("season"))


def PL_roster(season):
    """Who played in each game of `season`: the played log (game_id, gsis_id, team, position) with the player master's names."""
    plog = H.read_played_logs(season)[0].filter(pl.col("season") == season).select("game_id", gsis_id="player_id", team="team", position="family")
    pcon = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    names = pcon.execute("SELECT DISTINCT gsis_id, display_name FROM players WHERE gsis_id IS NOT NULL").pl()
    pcon.close()
    return plog.join(names, on="gsis_id", how="inner").unique(["game_id", "gsis_id"])


if __name__ == "__main__":
    main()

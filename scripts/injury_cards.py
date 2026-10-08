"""Five before-and-after injury cards from 2024 (the 4a.4 manual check).

  python scripts/injury_cards.py

Each card is one real team-game: the injured player's status and P(play), the team's expected carry / target / dropback shares
BEFORE redistribution (trailing baseline) and AFTER (4a.2, statuses as of the main run), and what actually happened in the game.
Everything is computed with data before that week's first kickoff.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from features import phase4_inputs as p4
from models import injuries as ij

CARDS = [  # (title, game_id, team, role of the injured player)
    ("RB1 out", None, None, "RB1"), ("WR1 out", None, None, "WR1"), ("QB switch", None, None, "QB"),
    ("Questionable, DNP in practice", None, None, None), ("OL starter out", None, None, "OL")]


def main():
    f = ij.build_role_frame(max_season=2024)
    pw = ij.build_player_weeks(max_season=2024)
    injr, ros = ij.load_injury_rows(max_season=2024), ij.load_roster_status(max_season=2024)
    status_fn = ij.make_status_lookup(injr)
    wcs = p4.week_cutoffs(max_season=2024)
    names = {}
    try:
        import duckdb

        import config
        c = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
        names = dict(c.execute("SELECT DISTINCT gsis_id, full_name FROM rosters_weekly WHERE gsis_id IS NOT NULL").fetchall())
    except Exception:
        pass
    nm = lambda i: names.get(i, i)
    g24 = f.filter(pl.col("season") == 2024)
    picks = []
    for role in ("RB1", "WR1", "QB", "OL"):
        # a role holder who did not play, and was on the report as Out / Doubtful / Questionable when the main run was made
        cand = g24.filter((pl.col("role") == role) & (~pl.col("played")) & (pl.col("base_snap") > 0.5) & (pl.col("week") >= 4)).sort("week")
        best = None
        for r in cand.iter_rows(named=True):
            rep, pra = status_fn(r["player_id"], r["season"], r["week"], ij.main_run_as_of(r["gameday"]))
            if rep in ("Out", "Doubtful") and r["base_snap"] > 0.5:
                best = r
                break
        picks.append((role, best))
    q = (pw.filter((pl.col("season") == 2024) & (pl.col("report_status") == "Questionable") & (pl.col("practice_status") == "DNP")
                   & (pl.col("normal_snap_pct") > 0.6) & pl.col("group").is_in(["WR", "RB", "TE", "QB"]) & (pl.col("week") >= 4)).sort("week"))
    qrow = q.row(0, named=True) if q.height else None
    for title, (role, best) in zip(("RB1 out", "WR1 out", "QB switch", "OL starter out"), picks):
        if best:
            card(title, f, pw, injr, ros, status_fn, wcs, best["game_id"], best["team"], best["season"], best["week"], best["gameday"],
                 best["player_id"], nm)
    if qrow:
        g = g24.filter((pl.col("team") == qrow["team"]) & (pl.col("week") == qrow["week"])).row(0, named=True)
        card("Questionable, DNP in practice", f, pw, injr, ros, status_fn, wcs, g["game_id"], qrow["team"], 2024, qrow["week"], g["gameday"],
             qrow["gsis_id"], nm)


def card(title, f, pw, injr, ros, status_fn, wcs, game_id, team, season, week, gameday, pid, nm):
    wc = wcs.filter((pl.col("season") == season) & (pl.col("week") == week))["cutoff_date"][0]
    model, shifts = ij.fit_status_model(pw, wc), ij.fit_shifts(f, wc)
    blocked = set(ros.filter((pl.col("season") == season) & (pl.col("week") == week)
                             & pl.col("roster_status").is_in(list(ij.BLOCKED_ROSTER_STATUSES)))["gsis_id"].to_list())
    groups = dict(pw.sort("gameday").group_by("gsis_id", maintain_order=True).agg(pl.col("group").last()).iter_rows())
    out = ij.game_expected_shares(f.filter(pl.col("team") == team), shifts, model, None, None, game_id, team, season, week, gameday,
                                  ij.main_run_as_of(gameday), wc, status_fn=status_fn, blocked_ids=blocked, player_groups=groups, with_eff=False,
                                  exit_model=ij.fit_exit_model(pw, wc))
    rep, pra = status_fn(pid, season, week, ij.main_run_as_of(gameday))
    me = out.filter(pl.col("player_id") == pid)
    actual = f.filter((pl.col("game_id") == game_id) & (pl.col("team") == team) & pl.col("played"))
    print(f"\n=== {title}: {nm(pid)} ({team}, 2024 week {week}, {game_id}) ===")
    print(f"status as of main run: report={rep} practice={pra}   P(out beyond healthy)={me['p_out'][0] if me.height else None}   "
          f"expected_snap_share={me['exp_snap_share'][0] if me.height else None}")
    show = (out.filter(pl.col("player_id") != "rest").with_columns(name=pl.col("player_id").map_elements(nm, return_dtype=pl.String))
            .join(actual.select("player_id", act_carry="carry_share", act_target="target_share", act_dropback="dropback_share"), on="player_id", how="left")
            .with_columns([pl.col(c).round(3) for c in ("base_carry", "exp_carry", "base_target", "exp_target", "base_dropback", "exp_dropback", "act_carry",
                                                         "act_target", "act_dropback")])
            .sort(pl.col("exp_target") + pl.col("exp_carry") + pl.col("exp_dropback"), descending=True).head(8))
    with pl.Config(tbl_cols=12, tbl_width_chars=200, tbl_hide_dataframe_shape=True, fmt_str_lengths=18):
        print(show.select("name", "role", "base_carry", "exp_carry", "act_carry", "base_target", "exp_target", "act_target",
                          "base_dropback", "exp_dropback", "act_dropback"))


if __name__ == "__main__":
    main()

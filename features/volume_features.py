"""Leakage-safe feature tables for the volume models (models/volume.py).

Every row's features use only games dated strictly BEFORE that row's week cutoff (the first
kickoff of its week -- the same cutoff the walk-forward harness uses), so a table built once
can be sliced by cutoff for training without ever showing a model the future.

Feature groups
--------------
own history    recency-weighted (and recency x continuity-weighted) trailing means of the player's own
               stats, usage share, last value, weight mass, history length
team rates     the player's team's recency-weighted rush attempts, dropbacks, plays, pass rate, points
opp allowed    the opponent's recency-weighted allowed rush attempts / dropbacks / plays / pass rate /
               points, plus what it allowed the same position+slot group over the trailing year
context        depth slot, position family, home/away, team game number, week

Weights come from features/weights.py semantics: recency half-life 6 team games with an 8-game
offseason; continuity penalties from config.CONTINUITY_PENALTIES for the market a quantity maps to
(config.BASELINE_TO_PENALTY_MARKET). Continuity factors QB / OL / RB_group / WR_TE_group come from
depth-chart lineups (features/lineups.py), "role" from the slot bucket, and a team change flags all
factors. HC and OC are never flagged -- there is no trustworthy historical coaching source yet.
No market lines (spread/total/odds) are used anywhere: this is the pure track.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

import config
from features import lineups as lu

OFFSEASON = float(config.OFFSEASON_GAP_GAMES)
HALF_LIFE = float(config.RECENCY_HALF_LIFE_GAMES)
FAMILY_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

# quantity -> player-level spec. `stats` are the own-history columns averaged; `share` is the
# usage share (stat / team total of the same thing) averaged as an extra feature.
PLAYER_SPECS = {
    "pass_att": dict(label="attempts", stats=["attempts", "completions", "passing_yards", "carries"],
                     share=("attempts", "team_att")),
    "rush_att": dict(label="carries", stats=["carries", "rushing_yards", "targets"],
                     share=("carries", "team_rush")),
    "targets": dict(label="targets", stats=["targets", "receptions", "receiving_yards"],
                    share=("targets", "team_att")),
}
TEAM_STATS = ["rush_att", "dropbacks", "plays", "pass_rate", "pts"]


@dataclass
class FeatureTables:
    players: dict      # quantity -> DataFrame (keys + features + label)
    teams: pl.DataFrame  # one row per team-game: features + label `plays`


def build_team_volume(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """Team-game rush attempts / dropbacks / plays, summed over ALL players' stats (every position).

    dropbacks = pass attempts + sacks suffered; plays = rush attempts + dropbacks.
    """
    cap = "" if max_season is None else f" AND season <= {int(max_season)}"
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT game_id, team, CAST(sum(carries) AS DOUBLE) AS rush_att, CAST(sum(attempts) AS DOUBLE) AS pass_att, "
            "CAST(sum(coalesce(sacks_suffered, 0)) AS DOUBLE) AS sacks FROM "
            "(SELECT DISTINCT ON (player_id, season, week) * FROM player_stats "
            " ORDER BY player_id, season, week, pulled_at DESC) "
            f"WHERE season_type = 'REG'{cap} GROUP BY game_id, team").pl()
    finally:
        con.close()
    return d.with_columns(dropbacks=pl.col("pass_att") + pl.col("sacks"),
                          plays=pl.col("rush_att") + pl.col("pass_att") + pl.col("sacks")).drop("sacks")


def _day(dates) -> np.ndarray:
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


class _Clock:
    """Maps (season, team_game_num) to an absolute game index so that
    index differences = team games elapsed, with each offseason counting as 8 games."""

    def __init__(self, seasons):
        self.cum, self.idx, total = {}, {}, 0
        for k, s in enumerate(sorted(set(seasons))):
            self.cum[s], self.idx[s] = total, k
            total += 17 if s >= 2021 else 16

    def a(self, season, game_num):
        season = np.asarray(season)
        return (np.vectorize(self.cum.get)(season) + OFFSEASON * np.vectorize(self.idx.get)(season)
                + np.asarray(game_num)).astype(float)


def _wavg(w, x):
    """Weighted mean ignoring NaN values; NaN if no weight mass."""
    m = ~np.isnan(x)
    ws = w[m].sum()
    return float((w[m] * x[m]).sum() / ws) if ws > 0 else np.nan


# ------------------------------------------------------------------ team-level tables
def _key(k):
    return k[0] if isinstance(k, tuple) else k


def build_team_side_features(team_log, team_volume, week_cutoff, clock) -> pl.DataFrame:
    """Per (team, season, week): the team's own rates (off_*) and what it allowed (def_*).

    Past games are those strictly before the week's cutoff; recency decay is in that team's own games.
    """
    tg = (team_log.join(team_volume, on=["game_id", "team"], how="left")
          .with_columns(pass_rate=pl.col("dropbacks") / pl.col("plays"), pts=pl.col("pf").cast(pl.Float64))
          .sort("team", "gameday"))
    nums = tg.select("game_id", opponent=pl.col("team"), o_num=pl.col("team_game_num"))
    allowed = (tg.join(nums, on=["game_id", "opponent"], how="left")
               .select(team=pl.col("opponent"), gameday="gameday", season="season", team_game_num=pl.col("o_num"),
                       rush_att="rush_att", dropbacks="dropbacks", plays="plays", pass_rate="pass_rate",
                       pts=pl.col("pa").cast(pl.Float64))
               .sort("team", "gameday"))
    keys = tg.select("team", "season", "week", "team_game_num").unique().sort("team", "season", "week")
    tables = []
    for src, prefix in ((tg, "off_"), (allowed, "def_")):
        by_team = {_key(k): g for k, g in src.partition_by("team", as_dict=True).items()}
        feats = []
        for r in keys.iter_rows(named=True):
            row = {"team": r["team"], "season": r["season"], "week": r["week"]}
            g = by_team.get(r["team"])
            j = 0
            if g is not None:
                gd = _day(g["gameday"])
                j = int(np.searchsorted(gd, week_cutoff[(r["season"], r["week"])].toordinal(), side="left"))
            if not j:
                feats.append(row | {prefix + s: np.nan for s in TEAM_STATS})
                continue
            ago = clock.a(r["season"], r["team_game_num"]) - clock.a(g["season"].to_numpy()[:j], g["team_game_num"].to_numpy()[:j])
            w = 0.5 ** (np.maximum(ago, 1.0) / HALF_LIFE)
            feats.append(row | {prefix + s: _wavg(w, g[s].to_numpy()[:j].astype(float)) for s in TEAM_STATS})
        tables.append(pl.DataFrame(feats))
    return tables[0].join(tables[1], on=["team", "season", "week"])


# ------------------------------------------------------------------ continuity
def continuity_weights(penalties, lin_past, lin_target, slot_p, slot_t, fam_p, fam_t, team_changed) -> np.ndarray:
    """Per past game: product of the penalties for every factor that changed vs the target game.

    QB / OL / RB_group / WR_TE_group: the lineup-group id differs (a missing chart, id -1, on either
    side is never a change). role: slot bucket or position family differs. A team change flags every
    factor, HC and OC included. HC / OC are otherwise never flagged (no trusted historical source).
    """
    flags = {"HC": team_changed, "OC": team_changed, "role": (slot_p != slot_t) | (fam_p != fam_t) | team_changed}
    for factor, comp in (("QB", "QB"), ("OL", "OL"), ("RB_group", "RB"), ("WR_TE_group", "WRTE")):
        t = lin_target[comp]
        flags[factor] = ((lin_past[comp] != t) & (lin_past[comp] != -1) & (t != -1)) | team_changed
    w = np.ones(len(team_changed))
    for factor, changed in flags.items():
        w = w * np.where(changed, penalties[factor], 1.0)
    return w


# ------------------------------------------------------------------ player-level tables
def build_player_features(quantity, player_log, team_log, team_volume, lineups, week_cutoff, clock, team_side,
                          eligible) -> pl.DataFrame:
    """Feature rows for one volume quantity: one row per eligible player-game."""
    spec = PLAYER_SPECS[quantity]
    pen = config.CONTINUITY_PENALTIES[config.BASELINE_TO_PENALTY_MARKET[quantity]]
    stats, label_col = spec["stats"], spec["label"]
    tv = team_volume.rename({"rush_att": "team_rush", "pass_att": "team_att"}).select("game_id", "team", "team_rush", "team_att")
    base = (player_log.join(tv, on=["game_id", "team"], how="left")
            .join(team_log.select("game_id", "team", "is_home"), on=["game_id", "team"], how="left")
            .with_columns(_share=pl.col(spec["share"][0]) / pl.col(spec["share"][1]).clip(lower_bound=1))
            .sort("player_id", "gameday"))
    lineup_ids = {(r["team"], r["season"], r["week"]): r for r in lineups.iter_rows(named=True)}
    team_codes = {t: i for i, t in enumerate(sorted(base["team"].unique()))}

    # trailing-year "allowed to this role" means, via cumulative sums over date-sorted rows
    role = {}
    for k, g in base.sort("gameday").partition_by("opponent", "family", "slot", as_dict=True).items():
        role[tuple(k)] = (_day(g["gameday"]), np.concatenate([[0.0], np.cumsum(g[label_col].to_numpy().astype(float))]))

    rows = []
    for k, g in base.partition_by("player_id", as_dict=True, maintain_order=True).items():
        pid = _key(k)
        n = g.height
        gd, season, week, tgn = _day(g["gameday"]), g["season"].to_numpy(), g["week"].to_numpy(), g["team_game_num"].to_numpy()
        teams_l, opp_l, fam_l = g["team"].to_list(), g["opponent"].to_list(), g["family"].to_list()
        team = np.array([team_codes[t] for t in teams_l])
        slot = g["slot"].to_numpy()
        fam = np.array([FAMILY_CODE[f] for f in fam_l])
        A = clock.a(season, tgn)
        X = np.column_stack([g[s].to_numpy().astype(float) for s in stats])
        share, label, home = g["_share"].to_numpy().astype(float), g[label_col].to_numpy().astype(float), g["is_home"].to_numpy()
        lin = {c: np.array([(lineup_ids.get((teams_l[i], int(season[i]), int(week[i]))) or {}).get(c, -1)
                            for i in range(n)], dtype=np.int64) for c in lu.COMPONENTS}
        for i in range(n):
            if quantity not in eligible(fam_l[i], int(slot[i])):
                continue
            cutoff = week_cutoff[(int(season[i]), int(week[i]))].toordinal()
            j = int(np.searchsorted(gd, cutoff, side="left"))
            r = {"player_id": pid, "game_id": g["game_id"][i], "gameday": g["gameday"][i], "season": int(season[i]),
                 "week": int(week[i]), "team": teams_l[i], "opponent": opp_l[i], "label": label[i],
                 "slot": int(slot[i]), "family": int(fam[i]), "is_home": float(home[i]),
                 "team_game_num": int(tgn[i]), "n_hist": j}
            if j:
                team_changed = team[:j] != team[i]
                traded_same_season = (season[:j] == season[i]) & team_changed
                ago = np.where(traded_same_season, np.maximum(1.0, week[i] - week[:j]), np.maximum(A[i] - A[:j], 1.0))
                w_rec = 0.5 ** (ago / HALF_LIFE)
                w_cont = continuity_weights(pen, {c: v[:j] for c, v in lin.items()}, {c: v[i] for c, v in lin.items()},
                                            slot[:j], slot[i], fam[:j], fam[i], team_changed)
                w_both = w_rec * w_cont
                for c, s in enumerate(stats):
                    r[f"rec_{s}"], r[f"rc_{s}"] = _wavg(w_rec, X[:j, c]), _wavg(w_both, X[:j, c])
                r.update(rec_share=_wavg(w_rec, share[:j]), rc_share=_wavg(w_both, share[:j]), last_val=float(label[j - 1]),
                         last3_val=float(np.mean(label[max(0, j - 3):j])), w_mass=float(w_both.sum()),
                         cont_mean_last3=float(w_cont[-3:].mean()))
            else:
                r.update({f"{p}_{s}": np.nan for s in stats for p in ("rec", "rc")})
                r.update(rec_share=np.nan, rc_share=np.nan, last_val=np.nan, last3_val=np.nan, w_mass=0.0,
                         cont_mean_last3=np.nan)
            d, cs = role.get((opp_l[i], fam_l[i], int(slot[i])), (None, None))
            r["opp_role_mean"], r["opp_role_n"] = np.nan, 0
            if d is not None:
                lo = int(np.searchsorted(d, cutoff - config.ROLE_WINDOW_DAYS, "left"))
                hi = int(np.searchsorted(d, cutoff, "left"))
                if hi > lo:
                    r["opp_role_mean"], r["opp_role_n"] = float((cs[hi] - cs[lo]) / (hi - lo)), hi - lo
            rows.append(r)
    df = pl.DataFrame(rows)
    own = team_side.select("team", "season", "week", **{f"team_{c}": pl.col(c) for c in team_side.columns if c.startswith("off_")})
    opp = team_side.select(opponent="team", season="season", week="week",
                           **{f"opp_{c}": pl.col(c) for c in team_side.columns if c.startswith("def_")})
    return (df.join(own, on=["team", "season", "week"], how="left")
              .join(opp, on=["opponent", "season", "week"], how="left").sort("player_id", "gameday"))


# ------------------------------------------------------------------ team-game table (game volume)
def build_team_game_features(team_log, team_volume, week_cutoff, team_side) -> pl.DataFrame:
    """One row per team-game: the team's rates vs the opponent's allowed rates; label = team plays."""
    tg = team_log.join(team_volume.select("game_id", "team", "plays"), on=["game_id", "team"], how="left")
    own = team_side.select("team", "season", "week", **{f"team_{c}": pl.col(c) for c in team_side.columns if c.startswith("off_")})
    opp = team_side.select(opponent="team", season="season", week="week",
                           **{f"opp_{c}": pl.col(c) for c in team_side.columns if c.startswith("def_")})
    return (tg.select("game_id", "team", "opponent", "season", "week", "gameday", "team_game_num",
                      is_home=pl.col("is_home").cast(pl.Float64), label=pl.col("plays").cast(pl.Float64))
              .join(own, on=["team", "season", "week"], how="left")
              .join(opp, on=["opponent", "season", "week"], how="left").sort("team", "gameday"))


def build_feature_tables(player_log, team_log, team_volume, lineups, eligible) -> FeatureTables:
    """Build every quantity's feature table from already-loaded frames (no database access)."""
    wk = team_log.group_by("season", "week").agg(cutoff=pl.col("gameday").min())
    week_cutoff = {(r["season"], r["week"]): r["cutoff"] for r in wk.iter_rows(named=True)}
    clock = _Clock(team_log["season"].unique().to_list())
    side = build_team_side_features(team_log, team_volume, week_cutoff, clock)
    players = {q: build_player_features(q, player_log, team_log, team_volume, lineups, week_cutoff, clock, side, eligible)
               for q in PLAYER_SPECS}
    return FeatureTables(players, build_team_game_features(team_log, team_volume, week_cutoff, side))


def load_feature_tables(data, raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> FeatureTables:
    """Feature tables for a backtest.BacktestData (seasons capped in SQL for the volume/lineup inputs)."""
    from eval import backtest as bt
    cap = max_season if max_season is not None else bt.MAX_BACKTEST_SEASON
    tv = build_team_volume(raw_db, cap)
    ln = lu.build_lineups(raw_db, cap)
    for f in (tv.join(data.team_log.select("game_id", "season").unique(), on="game_id", how="inner"), ):
        bt.assert_no_holdout(f)
    return build_feature_tables(data.player_log, data.team_log, tv, ln, bt.eligible_markets)

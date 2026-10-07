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
(config.MARKETS[market]["penalty_key"]). Continuity detection and the penalty product live in
features/weights.py (the single implementation); the group ids come from features/lineups.py.
No market lines (spread/total/odds) are used anywhere: this is the pure track.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

import config
from features import lineups as lu
from features.weights import continuity_flags, continuity_weight

OFFSEASON = float(config.OFFSEASON_GAP_GAMES)
HALF_LIFE = float(config.RECENCY_HALF_LIFE_GAMES)
FAMILY_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

# quantity -> player-level spec. `stats` are the own-history columns averaged; `share` is the
# usage share (stat / team total of the same thing) averaged as an extra feature.
# Spec keys: market (eligibility + continuity penalties), stats (columns averaged), share, and either
# `label` (a stat column, volume) or `ratio` = (numerator, denominator) columns (efficiency; label =
# num/den, undefined when den == 0). `pooled` = (name, num, den) pooled recency-weighted ratios.
PLAYER_SPECS = {
    "pass_att": dict(market="pass_att", label="attempts", stats=["attempts", "completions", "passing_yards", "carries"],
                     share=("attempts", "team_att")),
    "rush_att": dict(market="rush_att", label="carries", stats=["carries", "rushing_yards", "targets"],
                     share=("carries", "team_rush")),
    "targets": dict(market="targets", label="targets", stats=["targets", "receptions", "receiving_yards"],
                    share=("targets", "team_att")),
}
TEAM_STATS = ["rush_att", "dropbacks", "plays", "pass_rate", "pts"]
TEAM_EFF_STATS = ["ypc", "comp_pct", "yds_per_cmp", "ypa", "pts_per_play"]


@dataclass
class FeatureTables:
    players: dict      # quantity -> DataFrame (keys + features + label)
    teams: pl.DataFrame  # one row per team-game: features + label `plays`


def build_team_volume(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """Team-game rush attempts / dropbacks / plays, summed over ALL players' stats (every position).

    dropbacks = pass attempts + sacks suffered; plays = rush attempts + dropbacks. Also carries the team's
    rushing / passing yards and completions (for team efficiency rates).
    """
    cap = config.season_sql(max_season)  # FEATURE_HISTORY_START .. cap; raises config.HoldoutError for 2025+
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT game_id, team, CAST(sum(carries) AS DOUBLE) AS rush_att, CAST(sum(attempts) AS DOUBLE) AS pass_att, "
            "CAST(sum(coalesce(sacks_suffered, 0)) AS DOUBLE) AS sacks, "
            "CAST(sum(rushing_yards) AS DOUBLE) AS rush_yds, CAST(sum(passing_yards) AS DOUBLE) AS pass_yds, "
            "CAST(sum(completions) AS DOUBLE) AS completions FROM "
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


def _team_frame(team_log, team_volume) -> pl.DataFrame:
    """Team-game rows with every derived rate (volume and efficiency)."""
    return (team_log.join(team_volume, on=["game_id", "team"], how="left")
            .with_columns(pass_rate=pl.col("dropbacks") / pl.col("plays"), pts=pl.col("pf").cast(pl.Float64),
                          ypc=pl.col("rush_yds") / pl.col("rush_att").clip(lower_bound=1),
                          comp_pct=pl.col("completions") / pl.col("pass_att").clip(lower_bound=1),
                          yds_per_cmp=pl.col("pass_yds") / pl.col("completions").clip(lower_bound=1),
                          ypa=pl.col("pass_yds") / pl.col("pass_att").clip(lower_bound=1),
                          pts_per_play=pl.col("pf") / pl.col("plays"))
            .sort("team", "gameday"))


def build_team_side_features(team_log, team_volume, week_cutoff, clock, stats=None) -> pl.DataFrame:
    """Per (team, season, week): the team's own rates (off_*) and what it allowed (def_*).

    Past games are those strictly before the week's cutoff; recency decay is in that team's own games.
    `def_*` for team O is the offence-side stat of the teams that FACED O (so def_pts = points O allowed).
    """
    stats = stats or TEAM_STATS
    tg = _team_frame(team_log, team_volume)
    nums = tg.select("game_id", opponent=pl.col("team"), o_num=pl.col("team_game_num"))
    allowed = (tg.join(nums, on=["game_id", "opponent"], how="left")
               .with_columns(team_game_num=pl.col("o_num"), team=pl.col("opponent"))
               .select("team", "gameday", "season", "team_game_num", *stats).sort("team", "gameday"))
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
                feats.append(row | {prefix + s: np.nan for s in stats})
                continue
            ago = clock.a(r["season"], r["team_game_num"]) - clock.a(g["season"].to_numpy()[:j], g["team_game_num"].to_numpy()[:j])
            w = 0.5 ** (np.maximum(ago, 1.0) / HALF_LIFE)
            feats.append(row | {prefix + s: _wavg(w, g[s].to_numpy()[:j].astype(float)) for s in stats})
        tables.append(pl.DataFrame(feats))
    return tables[0].join(tables[1], on=["team", "season", "week"])


# ------------------------------------------------------------------ player-level tables
def build_player_features(quantity, spec, player_log, team_log, team_volume, lineups, week_cutoff, clock, team_side,
                          eligible, team_cols=None) -> pl.DataFrame:
    """Feature rows for one quantity: one row per eligible player-game.

    Volume specs have `label` (a stat column); efficiency specs have `ratio` = (num, den) and the row's
    `label` is num/den (NaN when den == 0), with `den` kept as the row's training weight. Rows with den == 0
    stay in the table (a target's denominator is unknown before the game); trainers drop them.
    """
    market = spec["market"]
    penalty_key = config.MARKETS[market]["penalty_key"]
    stats, pooled = spec["stats"], spec.get("pooled", [])
    num_col, den_col = spec["ratio"] if "ratio" in spec else (spec["label"], None)
    tv = (team_volume.rename({"rush_att": "team_rush", "pass_att": "team_att"})
          .select("game_id", "team", "team_rush", "team_att"))
    base = (player_log.join(tv, on=["game_id", "team"], how="left")
            .join(team_log.select("game_id", "team", "is_home"), on=["game_id", "team"], how="left")
            .sort("player_id", "gameday"))
    has_share = bool(spec.get("share"))
    if has_share:
        base = base.with_columns(_share=pl.col(spec["share"][0]) / pl.col(spec["share"][1]).clip(lower_bound=1))
    else:
        base = base.with_columns(_share=pl.lit(np.nan))
    lineup_ids = {(r["team"], r["season"], r["week"]): r for r in lineups.iter_rows(named=True)}
    team_codes = {t: i for i, t in enumerate(sorted(base["team"].unique()))}

    def num_den(g):
        n = g[num_col].to_numpy().astype(float)
        d = g[den_col].to_numpy().astype(float) if den_col else np.ones(len(n))
        return n, d

    # trailing-year "allowed to this role" pooled means, via cumulative sums over date-sorted rows
    role = {}
    for k, g in base.sort("gameday").partition_by("opponent", "family", "slot", as_dict=True).items():
        n, d = num_den(g)
        role[tuple(k)] = (_day(g["gameday"]), np.concatenate([[0.0], np.cumsum(n)]), np.concatenate([[0.0], np.cumsum(d)]))

    rows = []
    for k, g in base.partition_by("player_id", as_dict=True, maintain_order=True).items():
        pid = _key(k)
        n_rows = g.height
        gd, season, week, tgn = _day(g["gameday"]), g["season"].to_numpy(), g["week"].to_numpy(), g["team_game_num"].to_numpy()
        teams_l, opp_l, fam_l = g["team"].to_list(), g["opponent"].to_list(), g["family"].to_list()
        team = np.array([team_codes[t] for t in teams_l])
        slot = g["slot"].to_numpy()
        fam = np.array([FAMILY_CODE[f] for f in fam_l])
        A = clock.a(season, tgn)
        X = np.column_stack([g[s].to_numpy().astype(float) for s in stats])
        P = [(nm, g[a].to_numpy().astype(float), g[b].to_numpy().astype(float)) for nm, a, b in pooled]
        share, home = g["_share"].to_numpy().astype(float), g["is_home"].to_numpy()
        lab_n, lab_d = num_den(g)
        label = np.where(lab_d > 0, lab_n / np.where(lab_d > 0, lab_d, 1), np.nan) if den_col else lab_n
        lin = {c: np.array([(lineup_ids.get((teams_l[i], int(season[i]), int(week[i]))) or {}).get(c, -1)
                            for i in range(n_rows)], dtype=np.int64) for c in lu.COMPONENTS}
        for i in range(n_rows):
            if market not in eligible(fam_l[i], int(slot[i])):
                continue
            cutoff = week_cutoff[(int(season[i]), int(week[i]))].toordinal()
            j = int(np.searchsorted(gd, cutoff, side="left"))
            r = {"player_id": pid, "game_id": g["game_id"][i], "gameday": g["gameday"][i], "season": int(season[i]),
                 "week": int(week[i]), "team": teams_l[i], "opponent": opp_l[i], "label": label[i],
                 "slot": int(slot[i]), "family": int(fam[i]), "is_home": float(home[i]),
                 "team_game_num": int(tgn[i]), "n_hist": j}
            if den_col:
                r["den"] = float(lab_d[i])
            if j:
                team_changed = team[:j] != team[i]
                traded_same_season = (season[:j] == season[i]) & team_changed
                ago = np.where(traded_same_season, np.maximum(1.0, week[i] - week[:j]), np.maximum(A[i] - A[:j], 1.0))
                w_rec = 0.5 ** (ago / HALF_LIFE)
                w_cont = continuity_weight(continuity_flags({c: v[:j] for c, v in lin.items()}, {c: v[i] for c, v in lin.items()},
                                                            slot[:j], slot[i], fam[:j], fam[i], team_changed), penalty_key)
                w_both = w_rec * w_cont
                for c, s in enumerate(stats):
                    r[f"rec_{s}"], r[f"rc_{s}"] = _wavg(w_rec, X[:j, c]), _wavg(w_both, X[:j, c])
                for nm, pn, pd_ in P:  # pooled ratios: sum(w*num) / sum(w*den), plus the weight mass behind them
                    for tag, w in (("rec", w_rec), ("rc", w_both)):
                        mass = float((w * pd_[:j]).sum())
                        r[f"{tag}_{nm}"] = float((w * pn[:j]).sum() / mass) if mass > 0 else np.nan
                        r[f"{tag}_{nm}_mass"] = mass
                if has_share:
                    r.update(rec_share=_wavg(w_rec, share[:j]), rc_share=_wavg(w_both, share[:j]))
                r.update(last_val=float(label[j - 1]), w_mass=float(w_both.sum()), cont_mean_last3=float(w_cont[-3:].mean()))
                sl = slice(max(0, j - 3), j)
                dsum = lab_d[sl].sum()
                r["last3_val"] = float(lab_n[sl].sum() / dsum) if dsum > 0 else np.nan
            else:
                r.update({f"{p}_{s}": np.nan for s in stats for p in ("rec", "rc")})
                for nm, _, _ in P:
                    r.update({f"{p}_{nm}": np.nan for p in ("rec", "rc")} | {f"{p}_{nm}_mass": 0.0 for p in ("rec", "rc")})
                if has_share:
                    r.update(rec_share=np.nan, rc_share=np.nan)
                r.update(last_val=np.nan, last3_val=np.nan, w_mass=0.0, cont_mean_last3=np.nan)
            d, cn, cd = role.get((opp_l[i], fam_l[i], int(slot[i])), (None, None, None))
            r["opp_role_mean"], r["opp_role_n"] = np.nan, 0
            if d is not None:
                lo = int(np.searchsorted(d, cutoff - config.ROLE_WINDOW_DAYS, "left"))
                hi = int(np.searchsorted(d, cutoff, "left"))
                if hi > lo and cd[hi] - cd[lo] > 0:
                    r["opp_role_mean"], r["opp_role_n"] = float((cn[hi] - cn[lo]) / (cd[hi] - cd[lo])), hi - lo
            rows.append(r)
    df = pl.DataFrame(rows)
    return _attach_team_sides(df, team_side)


def _attach_team_sides(df, team_side):
    own = team_side.select("team", "season", "week", **{f"team_{c}": pl.col(c) for c in team_side.columns if c.startswith("off_")})
    opp = team_side.select(opponent="team", season="season", week="week",
                           **{f"opp_{c}": pl.col(c) for c in team_side.columns if c.startswith("def_")})
    return (df.join(own, on=["team", "season", "week"], how="left")
              .join(opp, on=["opponent", "season", "week"], how="left").sort(*[c for c in ("player_id", "team") if c in df.columns], "gameday"))


# ------------------------------------------------------------------ team-game table (game volume / efficiency)
def build_team_game_features(team_log, team_volume, team_side, label="plays") -> pl.DataFrame:
    """One row per team-game: the team's rates vs the opponent's allowed rates.

    label = "plays" (volume) or "pts_per_play" (efficiency; `den` = plays is the training weight).
    """
    tg = _team_frame(team_log, team_volume)
    cols = ["game_id", "team", "opponent", "season", "week", "gameday", "team_game_num",
            pl.col("is_home").cast(pl.Float64).alias("is_home"), pl.col(label).cast(pl.Float64).alias("label")]
    if label != "plays":
        cols.append(pl.col("plays").alias("den"))
    return _attach_team_sides(tg.select(cols), team_side)


def week_cutoffs(team_log):
    wk = team_log.group_by("season", "week").agg(cutoff=pl.col("gameday").min())
    return {(r["season"], r["week"]): r["cutoff"] for r in wk.iter_rows(named=True)}


def build_feature_tables(player_log, team_log, team_volume, lineups, eligible) -> FeatureTables:
    """Volume feature tables from already-loaded frames (no database access)."""
    week_cutoff = week_cutoffs(team_log)
    clock = _Clock(team_log["season"].unique().to_list())
    side = build_team_side_features(team_log, team_volume, week_cutoff, clock)
    players = {q: build_player_features(q, spec, player_log, team_log, team_volume, lineups, week_cutoff, clock, side, eligible)
               for q, spec in PLAYER_SPECS.items()}
    return FeatureTables(players, build_team_game_features(team_log, team_volume, side, "plays"))


def load_feature_tables(data, raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> FeatureTables:
    """Volume feature tables for a backtest.BacktestData (seasons capped in SQL for the volume/lineup inputs)."""
    from eval import backtest as bt
    cap = config.cap_season(max_season)  # raises config.HoldoutError for 2025+
    tv = build_team_volume(raw_db, cap)
    ln = lu.build_lineups(raw_db, cap)
    bt.assert_no_holdout(tv.join(data.team_log.select("game_id", "season").unique(), on="game_id", how="inner"))
    return build_feature_tables(data.player_log, data.team_log, tv, ln, bt.eligible_markets)

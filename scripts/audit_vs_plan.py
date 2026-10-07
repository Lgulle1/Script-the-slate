"""Audit the built code against the tightened plan (report only).

  python scripts/audit_vs_plan.py [--skip-rerun] [--skip-tests]

Prints a table and writes docs/audit_vs_plan.md. For each item: what the plan says, what the code
does (file:line), and MATCH or DIFF. This script changes NO model, feature, baseline, config value or
result: it only reads source files, the raw database and committed result files, and (unless skipped)
re-runs the Phase-3 backtest in memory WITHOUT saving, to test the determinism claim.
"""
from __future__ import annotations

import argparse
import inspect
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

import config  # noqa: E402

ITEMS: list["Item"] = []


@dataclass
class Item:
    id: str
    item: str
    plan: str
    code: str
    where: str
    status: str


def loc(path: str, pattern: str, nth: int = 0) -> str:
    """'file:line' of the nth line matching `pattern` ('-' if none)."""
    hits = [i for i, line in enumerate((ROOT / path).read_text().splitlines(), 1) if re.search(pattern, line)]
    return f"{path}:{hits[nth]}" if len(hits) > nth else f"{path}:-"


def add(id_, item, plan, code, where, ok):
    ITEMS.append(Item(id_, item, plan, code, where, "MATCH" if ok else "DIFF"))


def has(path: str, pattern: str) -> bool:
    return bool(re.search(pattern, (ROOT / path).read_text()))


# ---------------------------------------------------------------------------------------------- 1. config
def audit_config():
    c = config
    add("1a", "DB_PATH", "config.DB_PATH names the DuckDB file", 
        f"no DB_PATH; has DUCKDB_PATH={c.DUCKDB_PATH.name} (players, coaches) and RAW_DUCKDB_PATH={c.RAW_DUCKDB_PATH.name} (all pulls)",
        loc("config.py", r"^DUCKDB_PATH"), hasattr(c, "DB_PATH"))
    add("1b", "DATA_START_SEASON", "= 2016", f"{getattr(c, 'DATA_START_SEASON', 'not defined')}", "config.py:-", getattr(c, "DATA_START_SEASON", None) == 2016)
    add("1c", "BACKTEST_SEASONS", "2020 through 2024", f"{c.BACKTEST_SEASONS}", loc("config.py", r"^BACKTEST_SEASONS"),
        c.BACKTEST_SEASONS == [2020, 2021, 2022, 2023, 2024])
    guard = [loc("eval/backtest.py", r"locked"), loc("eval/backtest.py", r"def assert_no_holdout")]
    loaders_guarded = has("eval/baselines.py", r"HOLDOUT_SEASON")
    try:
        from eval import backtest as bt
        try:
            bt.walk_forward("x", None, seasons=[2025])
            raised = False
        except ValueError:
            raised = True
    except Exception as e:  # pragma: no cover
        raised = False
    add("1d", "HOLDOUT_SEASON", "= 2025; any request for it raises an error",
        f"HOLDOUT_SEASON={c.HOLDOUT_SEASON}; walk_forward(seasons=[2025]) raises ValueError={raised}; assert_no_holdout guards loaded frames; "
        f"but the data loaders (build_team_game_log / build_player_game_log, max_season=) have no HOLDOUT check={not loaders_guarded}",
        ", ".join(guard), c.HOLDOUT_SEASON == 2025 and raised and loaders_guarded)
    add("1e", "PROSPECTIVE_START_SEASON", "= 2026", f"{getattr(c, 'PROSPECTIVE_START_SEASON', 'not defined')}", "config.py:-",
        getattr(c, "PROSPECTIVE_START_SEASON", None) == 2026)
    from ingest import ids
    add("1f", "MAX_UNMATCHED_FRACTION", "= 0.01 (in config)",
        f"config has it={hasattr(c, 'MAX_UNMATCHED_FRACTION')}; defined in ingest/ids.py as {ids.MAX_UNMATCHED_FRACTION}",
        loc("ingest/ids.py", r"^MAX_UNMATCHED_FRACTION"), getattr(c, "MAX_UNMATCHED_FRACTION", None) == 0.01)
    add("1g", "GAME_MARGIN_SD", "= 13.5", f"{c.GAME_MARGIN_SD}", loc("config.py", r"^GAME_MARGIN_SD"), c.GAME_MARGIN_SD == 13.5)
    add("1h", "ROLE_WINDOW_DAYS", "= 365", f"{c.ROLE_WINDOW_DAYS}", loc("config.py", r"^ROLE_WINDOW_DAYS"), c.ROLE_WINDOW_DAYS == 365)
    plan_keys = {"pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds", "total", "spread", "moneyline"}
    m = getattr(c, "MARKETS", None)
    add("1i", "MARKETS", "dict with exactly the 11 keys, each carrying its player pool and continuity-penalty key",
        "no MARKETS in config. Pieces exist elsewhere: market list = eval/baselines.PLAYER_MARKETS + GAME_MARKETS; "
        "player pools = eval/backtest.eligible_markets(); penalty keys = config.BASELINE_TO_PENALTY_MARKET",
        loc("config.py", r"^BASELINE_TO_PENALTY_MARKET"), isinstance(m, dict) and set(m) == plan_keys)
    plan_map = {"pass_att": "pass_yds", "pass_cmp": "pass_yds", "pass_yds": "pass_yds", "rush_att": "rush_att", "rush_yds": "rush_yds",
                "targets": "rec", "rec": "rec", "rec_yds": "rec_yds", "total": "game_total", "spread": "game_total", "moneyline": "game_total"}
    add("1j", "market -> penalty key", "pass_att/pass_cmp/pass_yds->pass_yds; rush_att->rush_att; rush_yds->rush_yds; targets,rec->rec; "
        "rec_yds->rec_yds; total/spread/moneyline->game_total", f"BASELINE_TO_PENALTY_MARKET == plan mapping: {c.BASELINE_TO_PENALTY_MARKET == plan_map}",
        loc("config.py", r"^BASELINE_TO_PENALTY_MARKET"), c.BASELINE_TO_PENALTY_MARKET == plan_map)
    p = c.CONTINUITY_PENALTIES
    ok = len(p) == 11 and all(set(v) == set(c.CONTINUITY_FACTORS) and all(0 < x <= 1 for x in v.values()) for v in p.values())
    add("1k", "CONTINUITY_PENALTIES", "eleven penalty rows", f"{len(p)} rows ({', '.join(p)}), each with the 7 factors in (0,1]",
        loc("config.py", r"^CONTINUITY_PENALTIES"), ok)


# ---------------------------------------------------------------------------------------------- 2. pulls
def _table_seasons(con, t):
    cols = {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}
    if "season" in cols:
        q = f"SELECT min(TRY_CAST(season AS INTEGER)), max(TRY_CAST(season AS INTEGER)) FROM {t}"
    elif "nflverse_game_id" in cols:
        q = f"SELECT min(TRY_CAST(left(nflverse_game_id,4) AS INTEGER)), max(TRY_CAST(left(nflverse_game_id,4) AS INTEGER)) FROM {t}"
    elif "game_id" in cols:
        q = f"SELECT min(TRY_CAST(left(game_id,4) AS INTEGER)), max(TRY_CAST(left(game_id,4) AS INTEGER)) FROM {t}"
    else:
        return None
    lo, hi = con.execute(q).fetchone()
    return lo, hi


def audit_pulls():
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()]
    expected_min = {"ftn_charting": 2022, "pfr_advstats_pass": 2018, "pfr_advstats_rec": 2018, "pfr_advstats_rush": 2018, "participation": 2016}
    spans, late = {}, []
    for t in tables:
        if t == "weather":
            continue
        s = _table_seasons(con, t)
        spans[t] = s
        want = expected_min.get(t, 2016)
        if s and s[0] is not None and s[0] > want:
            late.append(f"{t} starts {s[0]} (plan {want})")
    add("2a", "start season of every pulled table", "every table from 2016 (FTN from 2022, PFR from 2018, participation 2016-2025) through the current season",
        f"{len(late)} of {len(spans)} tables start later than the plan: " + "; ".join(late) if late else "all start at or before the plan",
        loc("ingest/pull_nflverse.py", r"def pull_seasons") + " -> pull_seasons() = BACKTEST_SEASONS + HOLDOUT + current = 2020+", not late)
    no_pulled = [t for t in tables if "pulled_at" not in {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}]
    add("2b", "pulled_at on every raw table", "append-only with pulled_at", f"raw.duckdb tables missing pulled_at: {no_pulled or 'none'}",
        loc("ingest/pull_nflverse.py", r"def _append_raw"), not no_pulled)
    main = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    pcols = {r[0] for r in main.execute("DESCRIBE players").fetchall()}
    add("2c", "players (master id table) append-only", "every table pulled is append-only with pulled_at",
        f"players is rebuilt with CREATE OR REPLACE on every pull (replaces prior pulls); has pulled_at={'pulled_at' in pcols}; "
        f"coaches is also rebuilt each load (by design: hand-corrected rows are preserved)",
        loc("ingest/pull_nflverse.py", r"CREATE OR REPLACE TABLE"), "pulled_at" in pcols and not has("ingest/pull_nflverse.py", r"CREATE OR REPLACE TABLE"))
    part = spans.get("participation")
    cols = {r[0] for r in con.execute("DESCRIBE participation").fetchall()}
    add("2d", "participation", "2016-2025 only, tagged training-labels-only",
        f"seasons {part[0]}-{part[1]}; training_labels_only column present={'training_labels_only' in cols}; "
        f"seasons outside 2016-2025 are skipped with a warning", loc("ingest/pull_nflverse.py", r"def pull_participation"),
        part == (2016, 2025) and "training_labels_only" in cols)
    ftn = spans.get("ftn_charting")
    add("2e", "FTN from 2022", "FTN pulled from 2022", f"ftn_charting seasons {ftn[0]}-{ftn[1]}; FTN_MIN_SEASON=2022 clips earlier seasons",
        loc("ingest/pull_nflverse.py", r"^FTN_MIN_SEASON"), ftn[0] == 2022)
    pfr = {t: spans[t] for t in spans if t.startswith("pfr_advstats")}
    add("2f", "PFR from 2018", "PFR advanced stats from 2018", f"PFR_MIN_SEASON=2018 is set, but the default pull starts at 2020, so tables begin {sorted(set(v[0] for v in pfr.values()))}",
        loc("ingest/pull_nflverse.py", r"^PFR_MIN_SEASON"), all(v[0] == 2018 for v in pfr.values()))


# ---------------------------------------------------------------------------------------------- 3. weather
def audit_weather():
    src = (ROOT / "ingest/pull_weather.py").read_text()
    add("3a", "weather source", "Meteostat", f"imports meteostat={'import meteostat' in src}; references to open-meteo/nws: {bool(re.search('open-meteo|api.weather.gov', src, re.I))}",
        loc("ingest/pull_weather.py", r"import meteostat"), "import meteostat" in src and not re.search("open-meteo|api.weather.gov", src, re.I))
    add("3b", "one pull per stadium-year", "one request per stadium per year", "venue key = (rounded lat, lon, kickoff year); one Meteostat load per key (stadium identity via coordinates, not stadium_id)",
        loc("ingest/pull_weather.py", r"key = \(round"), has("ingest/pull_weather.py", r"round\(r\[\"lat\"\], 4\), round\(r\[\"lon\"\], 4\), _naive_utc"))
    add("3c", "closest hourly observation", "closest hourly observation to kickoff", f"smallest time gap within +/-1 h (MAX_HOUR_GAP), then nearest of 4 stations, requiring temperature and wind",
        loc("ingest/pull_weather.py", r"^MAX_HOUR_GAP"), has("ingest/pull_weather.py", r"MAX_HOUR_GAP = 1"))
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    cols = {r[0] for r in con.execute("DESCRIBE weather").fetchall()}
    srcs = [r[0] for r in con.execute("SELECT DISTINCT source FROM weather ORDER BY 1").fetchall()]
    add("3d", "source tag", "meteostat or missing", f"distinct source values in the weather table: {srcs}", loc("ingest/pull_weather.py", r"^SOURCE_METEOSTAT"),
        set(srcs) == {"meteostat", "missing"})
    nn = con.execute("SELECT count(precip_prob_pct) FROM weather").fetchone()[0]
    add("3e", "precip_prob_pct", "always null", f"non-null values in the weather table: {nn}", loc("ingest/pull_weather.py", r"precip_prob_pct=None"), nn == 0)
    indoor = [c for c in cols if "indoor" in c or "roof" in c or "dome" in c]
    add("3f", "indoor flag column", "each row carries an indoor flag", f"weather columns: {sorted(cols)}; indoor/roof column present: {indoor or 'no'} "
        f"(roof state lives in schedules; docstring says the features layer must handle it)", loc("ingest/pull_weather.py", r"domed venues"), bool(indoor))
    row = con.execute("""SELECT count(*), sum((w.source='missing')::int), min(s.gameday) FILTER (WHERE w.source <> 'missing'),
        max(s.gameday) FILTER (WHERE w.source <> 'missing'), max(w.pulled_at) FROM weather w JOIN
        (SELECT DISTINCT ON (game_id) game_id, gameday, home_score FROM schedules ORDER BY game_id, pulled_at DESC) s USING (game_id)
        WHERE s.home_score IS NULL""").fetchone()
    n, miss, lo, hi, pulled = row
    detail = (f"{miss} of {n} games without a final score are 'missing'. The other {n - miss} are NOT past games: kickoffs {lo} to {hi}, "
              f"weather pulled {str(pulled)[:10]}, yet source = 'meteostat' with values -- Meteostat returns model/forecast data for the coming week, "
              f"so the 'meteostat' tag mixes observed readings (past games) with forecasts (next ~8 days)") if n != miss else f"all {n} unplayed games are 'missing'"
    add("3g", "unplayed games", "written as missing", detail, loc("ingest/pull_weather.py", r"_missing\(rows"), n == miss)


# ---------------------------------------------------------------------------------------------- 4-5. ids, weights
def audit_ids_weights():
    import polars as pl
    from ingest import ids
    master = pl.DataFrame({"gsis_id": ["G1"], "pfr_id": ["P1"], "espn_id": ["1"]})
    over = pl.DataFrame({"pfr_id": ["P1"] * 98 + ["x", "y"]})
    under = pl.DataFrame({"pfr_id": ["P1"] * 99 + ["x"]})
    try:
        ids.add_canonical_gsis_id(over, master=master)
        raised = False
    except ids.IdMatchError:
        raised = True
    try:
        ids.add_canonical_gsis_id(under, master=master)
        quiet = True
    except ids.IdMatchError:
        quiet = False
    add("4a", "add_canonical_gsis_id", "raises IdMatchError above MAX_UNMATCHED_FRACTION", f"2% unmatched raises={raised}; exactly 1% does not={quiet}; threshold constant {ids.MAX_UNMATCHED_FRACTION}",
        loc("ingest/ids.py", r"if frac > max_unmatched"), raised and quiet)

    from features import weights as w
    ok = abs(w.recency_weight(6) - 0.5) < 1e-12 and abs(w.recency_weight(12) - 0.25) < 1e-12 and has("features/weights.py", r"0\.5 \*\* \(games_ago / half_life\)")
    add("5a", "recency weight", "0.5 ** (games_ago / 6)", "0.5 ** (games_ago / half_life), half_life = config.RECENCY_HALF_LIFE_GAMES = 6", loc("features/weights.py", r"0\.5 \*\*"), ok)
    add("5b", "offseason gap", "offseason counts as 8 games", f"games_elapsed(2023,17,2024,1) = {w.games_elapsed(2023, 17, 2024, 1)} (8 offseason + 1); config.OFFSEASON_GAP_GAMES={config.OFFSEASON_GAP_GAMES}",
        loc("features/weights.py", r"OFFSEASON_GAP_GAMES"), w.games_elapsed(2023, 17, 2024, 1) == 9)
    add("5c", "where continuity factors are detected", "features/weights.py", "weights.continuity_weight() only multiplies caller-supplied boolean flags and is NOT used by the feature builders; "
        "detection and a second, vectorised penalty product live in features/volume_features.py:continuity_weights and features/lineups.py",
        loc("features/volume_features.py", r"def continuity_weights"), False)
    add("5d", "factor QB", "QB1 differs", "depth-chart QB slot 1 (weekly chart) id differs between past game and target", loc("features/lineups.py", r'"QB": \(pl.col'), True)
    add("5e", "factor RB group", "top 2 RBs differ", "RB depth slots 1-2 (position 'RB', FBs are listed as RB) as a set", loc("features/lineups.py", r'"RB": \(pl.col'), True)
    add("5f", "factor WR/TE group", "top 4 WR/TE differ", "WR depth 1-3 plus TE depth 1 as a set (four players, but fixed 3 WR + 1 TE, not the top four WR/TE by slot)", loc("features/lineups.py", r'"WRTE"'), False)
    add("5g", "factor OL", "top 5 OL differ", "T, G and C at depth 1 (the five starters) as a set", loc("features/lineups.py", r'"OL": pl.col'), True)
    add("5h", "factor role", "own depth-chart slot differs", "slot bucket (1, 2, 3, 0=unlisted) or position family differs, or team changed", loc("features/volume_features.py", r'flags = \{"HC"'), True)
    add("5i", "HC and OC", "only on a team change until coaches_history exists", "HC and OC flagged only when the player's team changed", loc("features/volume_features.py", r'"HC": team_changed'), True)
    q = config.QUALITY_WEIGHTS
    used = subprocess.run(["grep", "-rn", "quality_weight", "--include=*.py", str(ROOT)], capture_output=True, text=True).stdout
    used = [l for l in used.splitlines() if "tests/" not in l and "weights.py" not in l and "audit_vs_plan" not in l]
    add("5j", "quality weight", "observed 1.0 / derived 0.9 / estimated 0.65", f"values {q}; quality_weight() is defined but called nowhere outside tests ({len(used)} call sites): no model or baseline applies it yet",
        loc("features/weights.py", r"def quality_weight"), q == {"observed": 1.0, "derived": 0.9, "estimated": 0.65})
    add("5k", "market -> penalty key through config.MARKETS", "every market resolves to a penalty key via config.MARKETS",
        "feature builders use config.BASELINE_TO_PENALTY_MARKET[spec['market']] (same mapping, different name; config.MARKETS does not exist)", loc("features/volume_features.py", r"BASELINE_TO_PENALTY_MARKET"), hasattr(config, "MARKETS"))


# ---------------------------------------------------------------------------------------------- 6-7. baselines, harness
def audit_baselines_harness():
    from eval import backtest as bt, baselines as bl, grade
    preds = pl.read_parquet(ROOT / "data/processed/baseline_predictions.parquet") if (ROOT / "data/processed/baseline_predictions.parquet").exists() else None
    combos = preds.select("method", "market").unique().height if preds is not None else None
    add("6a", "five baselines x eleven markets", "last3, season_avg, recency, 70/30 blend, role average on all 11 markets",
        f"METHODS={bl.METHODS}; markets={len(bl.PLAYER_MARKETS)} player + {len(bl.GAME_MARKETS)} game; (method, market) pairs in the saved backtest = {combos}",
        loc("eval/baselines.py", r"^METHODS"), len(bl.METHODS) == 5 and len(bl.PLAYER_MARKETS) + len(bl.GAME_MARKETS) == 11 and combos == 55)
    rb_slots = [s for s in range(0, 5) if "rush_att" in bt.eligible_markets("RB", s)]
    add("6b", "RB group", "RB1, RB2, other RBs and fullbacks", f"history log holds every RB/HB/FB at every slot (FB mapped to RB), but the SCORED rushing/receiving pool is RB slots {rb_slots} only: RB3+ and unlisted RBs are never graded",
        loc("eval/backtest.py", r'if family == "RB"'), rb_slots == [1, 2, 3, 0] or set(rb_slots) == {0, 1, 2, 3})
    add("6c", "targets pool", "WR, TE and RB", f"WR slots {[s for s in range(5) if 'targets' in bt.eligible_markets('WR', s)]}, TE slots {[s for s in range(5) if 'targets' in bt.eligible_markets('TE', s)]}, RB slots {[s for s in range(5) if 'targets' in bt.eligible_markets('RB', s)]}",
        loc("eval/backtest.py", r"def eligible_markets"), all("targets" in bt.eligible_markets(f, 1) for f in ("WR", "TE", "RB")))
    add("6d", "role / opponent-allowed window", "use ROLE_WINDOW_DAYS", f"_window() subtracts config.ROLE_WINDOW_DAYS ({config.ROLE_WINDOW_DAYS}) days for role average, opponent-allowed average and the league-average game baseline",
        loc("eval/baselines.py", r"config.ROLE_WINDOW_DAYS"), has("eval/baselines.py", r"timedelta\(days=config\.ROLE_WINDOW_DAYS\)"))
    import math
    g = bl.predict_game("last3", __import__("polars").DataFrame({"team": ["H", "A"] * 3, "opponent": ["A", "H"] * 3,
        "gameday": [__import__("datetime").date(2024, 9, d) for d in (1, 1, 8, 8, 15, 15)], "season": [2024] * 6, "team_game_num": [1, 1, 2, 2, 3, 3],
        "pf": [30, 20] * 3, "pa": [20, 30] * 3}), bl.GameTarget(__import__("datetime").date(2024, 9, 22), 2024, "H", "A", 4, 4))
    want = 0.5 * (1 + math.erf(10 / config.GAME_MARGIN_SD / math.sqrt(2)))
    add("6e", "moneyline", "normal CDF of expected margin / GAME_MARGIN_SD", f"predict_game moneyline = {g['moneyline']:.6f}; CDF(10/13.5) = {want:.6f}", loc("eval/baselines.py", r"math.erf"), abs(g["moneyline"] - want) < 1e-12)
    sigs = {n: list(inspect.signature(f).parameters) for n, f in bl.PLAYER_BASELINES.items()}
    add("6f", "pure, cutoff-based", "all baselines are cutoff-based pure functions", f"player baselines take {set(map(tuple, sigs.values()))} and read no database; cutoff may not exceed the game date (raises)",
        loc("eval/baselines.py", r"def _check_cutoff"), all(s == ["log", "target", "cutoff"] for s in sigs.values()))
    add("7a", "harness never touches 2025", "backtest/grade harness never touches 2025", "loaders cap seasons in SQL (max_season=2024), assert_no_holdout re-checks, walk_forward rejects seasons outside BACKTEST_SEASONS",
        loc("eval/backtest.py", r"def assert_no_holdout"), has("eval/backtest.py", r"max_season=MAX_BACKTEST_SEASON") and has("eval/backtest.py", r"assert_no_holdout"))
    fns = ["mean_absolute_error", "brier_score", "log_loss", "calibration_table", "pit_histogram", "pit_flatness", "predictive_scores", "skill_vs_baseline", "grade_backtest"]
    add("7b", "grading functions", "as in 2.4: MAE, Brier per rung, log loss, calibration (20% bands), whole-curve check, skill vs baseline", f"present: {[f for f in fns if hasattr(grade, f)]}; missing: {[f for f in fns if not hasattr(grade, f)] or 'none'}",
        loc("eval/grade.py", r"def calibration_table"), all(hasattr(grade, f) for f in fns))
    br = pl.read_parquet(ROOT / "baseline_results.parquet")
    has_best = "best_baseline" in set(br["metric"].to_list()) or any("best" in m for m in br["metric"].unique().to_list())
    add("7c", "best baseline per market stored", "run_baseline_backtest.py stores the best baseline per market on identical rows",
        f"baseline_results.parquet metrics: {sorted(br['metric'].unique().to_list())}: per-method scores only, no best-baseline record. The best baseline per market is chosen later, in eval/compare.py, and stored in volume_efficiency_results.parquet",
        loc("eval/compare.py", r"best = min\(losses"), has_best)


# ---------------------------------------------------------------------------------------------- 8-9. models, comparison
def audit_models_compare(rerun: bool):
    from models import combine, efficiency, volume
    P = volume.PARAMS
    add("8a", "hyperparameters", "~150 trees, 8 leaves, min 30 rows per leaf, L1", f"objective={P['objective']}, n_estimators={P['n_estimators']}, num_leaves={P['num_leaves']}, min_child_samples={P['min_child_samples']} (plus learning_rate={P['learning_rate']}, colsample_bytree={P['colsample_bytree']}, seed {P['random_state']})",
        loc("models/volume.py", r"^PARAMS"), (P["objective"], P["n_estimators"], P["num_leaves"], P["min_child_samples"]) == ("l1", 150, 8, 30))
    from features import volume_features as vf
    add("8b", "volume quantities", "one LightGBM per volume quantity", f"{list(vf.PLAYER_SPECS)} + team_plays (game markets)", loc("features/volume_features.py", r"^PLAYER_SPECS"),
        set(vf.PLAYER_SPECS) == {"pass_att", "rush_att", "targets"})
    add("8c", "efficiency quantities", "one LightGBM per efficiency quantity", f"{list(efficiency.EFFICIENCY_SPECS)} + pts_per_play (game markets)", loc("models/efficiency.py", r"^EFFICIENCY_SPECS"),
        set(efficiency.EFFICIENCY_SPECS) == {"comp_pct", "yds_per_cmp", "ypc", "catch_pct", "yds_per_rec"})
    add("8d", "features", "own recency+continuity-weighted stats, team rates (rush attempts, dropback rate), opponent-allowed rates",
        f"team stats {vf.TEAM_STATS} (off_* and def_*), own history rec_*/rc_* (recency and recency x continuity), opp_role_mean; no comparables/injuries/simulation/market lines",
        loc("features/volume_features.py", r"^TEAM_STATS"), "rush_att" in vf.TEAM_STATS and "pass_rate" in vf.TEAM_STATS)
    v = {"pass_att": 30.0, "rush_att": 12.0, "targets": 6.0}
    e = {"comp_pct": 0.6, "yds_per_cmp": 11.0, "ypc": 4.5, "catch_pct": 0.7, "yds_per_rec": 10.0}
    c = combine.combine_player(v, e)
    ok = (abs(c["pass_cmp"] - 18) < 1e-9 and abs(c["pass_yds"] - 198) < 1e-9 and abs(c["rush_yds"] - 54) < 1e-9 and abs(c["rec"] - 4.2) < 1e-9 and abs(c["rec_yds"] - 42) < 1e-9)
    gm = combine.combine_game({"plays_home": 64.0, "plays_away": 60.0}, {"ppp_home": 0.4, "ppp_away": 0.3})
    ok = ok and abs(gm["total"] - 43.6) < 1e-9 and abs(gm["spread"] - 7.6) < 1e-9
    add("8e", "combine formulas", "volume x efficiency (cmp = att x comp%; yds = att x comp% x yds/cmp; rush = att x ypc; rec = tgt x catch%; rec_yds = tgt x catch% x yds/rec; games = plays x points/play, spread = home - away)",
        "all verified with a worked example", loc("models/combine.py", r"def combine_player"), ok)
    from eval import compare as cp
    add("9a", "bootstrap", "season-week bootstrap, 10,000 resamples, fixed seed", f"resamples={config.BOOTSTRAP_RESAMPLES}, seed={config.BOOTSTRAP_SEED}, cluster = season*100+week", loc("eval/compare.py", r"def cluster_bootstrap"),
        config.BOOTSTRAP_RESAMPLES == 10000 and has("eval/compare.py", r"cluster = \(t\[\"season\"\]"))
    ve = pl.read_parquet(ROOT / "volume_efficiency_results.parquet")
    s = ve.filter(pl.col("table") == "summary")
    mets = set(s["metric"].to_list())
    add("9b", "row-level interval, per-season gains, mean bias", "all three reported", f"row-level: {'ci_lo_iid' in mets}; per-season gains: {ve.filter((pl.col('table') == 'by_season') & (pl.col('metric') == 'gain')).height} rows; mean bias: {'model_bias' in mets} (moneyline has none: it is a probability)",
        loc("eval/compare.py", r"iid_bootstrap\(diff\)"), {"ci_lo_iid", "model_bias"} <= mets)
    vs = [cp.verdict(0.5, 0.1, 0.9, 3), cp.verdict(0.62, -0.02, 1.27, 4), cp.verdict(-0.2, -0.6, 0.1, 1)]
    add("9c", "verdicts", "CLEARS, EDGE, NO as in 3.3", f"CLEARS = gain>0, CI excludes zero and wins >= {config.MIN_SEASONS_WON} seasons; EDGE = gain>0 but CI touches zero or too few seasons; NO otherwise. Test cases give {vs}",
        loc("eval/compare.py", r"def verdict"), vs == ["CLEARS", "EDGE", "NO"])
    if rerun:
        import run_volume_efficiency_backtest as rv
        _, again, _ = rv.run(save=False)
        same = again.equals(ve)
        add("9d", "second run identical", "a second run gives a byte-identical parquet", f"fresh in-memory re-run == committed volume_efficiency_results.parquet (all {ve.height} rows, exact float equality): {same}. "
            "(Byte equality of the written file was checked separately with cmp when the results were produced.)", loc("run_volume_efficiency_backtest.py", r"pq.write_table"), same)
    else:
        add("9d", "second run identical", "a second run gives a byte-identical parquet", "SKIPPED (--skip-rerun)", "-", False)


# ---------------------------------------------------------------------------------------------- 10. tests
def audit_tests(run_suite: bool):
    names = {
        "future-blindness": ("tests/test_volume.py", r"def test_features_never_see_the_future"),
        "future-blindness (efficiency)": ("tests/test_efficiency.py", r"def test_efficiency_features_never_see_the_future"),
        "holdout lock": ("tests/test_volume.py", r"locked"),
        "determinism": ("tests/test_volume.py", r"a\.equals\(b\)"),
        "recency matches baseline": ("tests/test_volume.py", r"def test_recency_feature_matches_phase2_baseline"),
    }
    found = {k: loc(*v) for k, v in names.items()}
    add("10a", "named tests exist", "future-blindness, holdout lock, determinism, recency-matches-baseline", "; ".join(f"{k}: {v}" for k, v in found.items()), "tests/",
        all(not v.endswith(":-") for v in found.values()))
    if run_suite:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
        last = [l for l in r.stdout.strip().splitlines() if "passed" in l or "failed" in l or "error" in l]
        add("10b", "full suite passes", "the full suite passes", last[-1] if last else r.stdout[-200:], "pytest", r.returncode == 0)
    else:
        add("10b", "full suite passes", "the full suite passes", "SKIPPED (--skip-tests)", "-", False)


def render(items):
    wid = {"id": 4, "item": 30, "status": 6}
    lines = []
    for i in items:
        lines.append(f"[{i.status:5}] {i.id:<4} {i.item}\n         plan: {i.plan}\n         code: {i.code}\n         where: {i.where}")
    return "\n".join(lines)


def write_md(items, path):
    d = [i for i in items if i.status == "DIFF"]
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    out = ["# Audit of built code vs the tightened plan", "",
           "Generated by `scripts/audit_vs_plan.py`. Report only: no model, feature, baseline, config value or result was changed.", "",
           f"**{len(items) - len(d)} MATCH, {len(d)} DIFF** across {len(items)} items.", "",
           "| # | Item | Plan says | Code does | Where | |", "|---|---|---|---|---|---|"]
    for i in items:
        out.append(f"| {i.id} | {esc(i.item)} | {esc(i.plan)} | {esc(i.code)} | `{i.where}` | {'**DIFF**' if i.status == 'DIFF' else 'MATCH'} |")
    out += ["", "## DIFF items", ""] + [f"- **{i.id} {i.item}** — {i.code}" for i in d]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rerun", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    audit_config(); audit_pulls(); audit_weather(); audit_ids_weights(); audit_baselines_harness()
    audit_models_compare(not a.skip_rerun); audit_tests(not a.skip_tests)
    print(render(ITEMS))
    write_md(ITEMS, ROOT / "docs/audit_vs_plan.md")
    d = [i for i in ITEMS if i.status == "DIFF"]
    print(f"\n{len(ITEMS) - len(d)} MATCH, {len(d)} DIFF of {len(ITEMS)} items -> docs/audit_vs_plan.md")
    print("DIFF: " + ", ".join(f"{i.id} {i.item}" for i in d))

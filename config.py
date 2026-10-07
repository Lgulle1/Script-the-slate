"""Central constants for script-the-slate."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

# Local secrets/settings live in a git-ignored .env, if present.
if (ROOT / ".env").exists():
    load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
PROCESSED_DIR = DATA_DIR / "processed"

DUCKDB_PATH = DATA_DIR / "script_the_slate.duckdb"
# Raw, append-only nflverse pulls live in their own DuckDB file under data/raw/.
RAW_DUCKDB_PATH = RAW_DIR / "raw.duckdb"

DATA_START_SEASON = 2016  # first season the plan wants pulled (pulls still start at 2020 -- see docs/audit_vs_plan.md 2a)
# First season any model, baseline or feature loader reads. Data before it (2016-2019) is pulled and stored but not
# read by anything yet; moving this to DATA_START_SEASON is a separate decision.
FEATURE_HISTORY_START = 2020
BACKTEST_SEASONS = list(range(2020, 2025))  # 2020-2024 inclusive
HOLDOUT_SEASON = 2025  # locked: never used for tuning or model selection
PROSPECTIVE_START_SEASON = 2026  # first season predicted live, out of sample

# A master-id join may leave at most this share of a batch unmatched before it raises (ingest/ids.py).
MAX_UNMATCHED_FRACTION = 0.01


class HoldoutError(ValueError):
    """A training or evaluation loader was asked for the locked holdout season (or later)."""


def cap_season(max_season=None) -> int:
    """The season cap every training/evaluation loader applies.

    Raises HoldoutError for HOLDOUT_SEASON or later, exactly as the walk-forward harness does. ``None``
    means "the last backtest season", never "everything", so a loader cannot return holdout rows by
    default. The raw pull functions are exempt: they must keep fetching 2025 and 2026 data.
    """
    if max_season is None:
        return max(BACKTEST_SEASONS)
    if int(max_season) >= HOLDOUT_SEASON:
        raise HoldoutError(f"max_season={max_season} reaches season {HOLDOUT_SEASON}+, which is the locked holdout; "
                           f"training/evaluation loaders stop at {max(BACKTEST_SEASONS)}")
    return int(max_season)

OUTDOORS = "outdoors"
DOME = "dome"
RETRACTABLE = "retractable"

# Used only to locate the weather API call. For retractable roofs, the actual
# game-day roof state comes from the schedules table's roof column.
# team -> (latitude, longitude, fixed_roof)
# NOTE: coordinates entered from memory; spot-check against a map before relying on them.
stadium_coordinates = {
    "Arizona Cardinals": (33.5276, -112.2626, RETRACTABLE),
    "Atlanta Falcons": (33.7554, -84.4008, RETRACTABLE),
    "Baltimore Ravens": (39.2780, -76.6227, OUTDOORS),
    "Buffalo Bills": (42.7738, -78.7870, OUTDOORS),
    "Carolina Panthers": (35.2258, -80.8528, OUTDOORS),
    "Chicago Bears": (41.8623, -87.6167, OUTDOORS),
    "Cincinnati Bengals": (39.0954, -84.5160, OUTDOORS),
    "Cleveland Browns": (41.5061, -81.6995, OUTDOORS),
    "Dallas Cowboys": (32.7473, -97.0945, RETRACTABLE),
    "Denver Broncos": (39.7439, -105.0201, OUTDOORS),
    "Detroit Lions": (42.3400, -83.0456, DOME),
    "Green Bay Packers": (44.5013, -88.0622, OUTDOORS),
    "Houston Texans": (29.6847, -95.4107, RETRACTABLE),
    "Indianapolis Colts": (39.7601, -86.1639, RETRACTABLE),
    "Jacksonville Jaguars": (30.3239, -81.6373, OUTDOORS),
    "Kansas City Chiefs": (39.0489, -94.4839, OUTDOORS),
    "Las Vegas Raiders": (36.0909, -115.1833, DOME),
    "Los Angeles Chargers": (33.9535, -118.3392, DOME),
    "Los Angeles Rams": (33.9535, -118.3392, DOME),
    "Miami Dolphins": (25.9580, -80.2389, OUTDOORS),
    "Minnesota Vikings": (44.9736, -93.2575, DOME),
    "New England Patriots": (42.0909, -71.2643, OUTDOORS),
    "New Orleans Saints": (29.9511, -90.0812, DOME),
    "New York Giants": (40.8128, -74.0742, OUTDOORS),
    "New York Jets": (40.8128, -74.0742, OUTDOORS),
    "Philadelphia Eagles": (39.9008, -75.1675, OUTDOORS),
    "Pittsburgh Steelers": (40.4468, -80.0158, OUTDOORS),
    "San Francisco 49ers": (37.4030, -121.9700, OUTDOORS),
    "Seattle Seahawks": (47.5952, -122.3316, OUTDOORS),
    "Tampa Bay Buccaneers": (27.9759, -82.5033, OUTDOORS),
    "Tennessee Titans": (36.1665, -86.7713, OUTDOORS),
    "Washington Commanders": (38.9078, -76.8644, OUTDOORS),
}


# nflverse schedule abbreviation -> stadium_coordinates key (full team name).
# Verified against nflreadpy.load_schedules for 2020-2024 (note LA, LV, JAX, WAS).
TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}
TEAM_NAME_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_NAME.items()}

# Neutral-site / international venues, keyed by the schedules table's stadium_id.
# stadium_id (not home team) is what locates a game, since for a neutral-site game
# the "home" team's own stadium is the wrong place. Entries: stadium_id -> (lat, lon).
# International coordinates entered from memory; spot-check against a map.
# Domestic neutral/relocated games reuse the home stadium's coordinates above.
_DOMESTIC_VENUE_TEAMS = {
    "PHO00": "Arizona Cardinals", "TAM00": "Tampa Bay Buccaneers",
    "JAX00": "Jacksonville Jaguars", "LAX01": "Los Angeles Rams",
    "DET00": "Detroit Lions", "VEG00": "Las Vegas Raiders",
    "NOR00": "New Orleans Saints", "CLE00": "Cleveland Browns",
    "IND00": "Indianapolis Colts", "MIA00": "Miami Dolphins",
    "NYC01": "New York Giants", "PIT00": "Pittsburgh Steelers",
    "SFO01": "San Francisco 49ers",
}
venue_coordinates = {
    **{sid: stadium_coordinates[team][:2] for sid, team in _DOMESTIC_VENUE_TEAMS.items()},
    "LON00": (51.5560, -0.2796),    # Wembley Stadium, London
    "LON02": (51.6043, -0.0664),    # Tottenham Hotspur Stadium, London
    "GER00": (48.2188, 11.6247),    # Allianz Arena, Munich
    "MUN01": (48.2188, 11.6247),    # FC Bayern Munich Stadium (Allianz Arena)
    "FRA00": (50.0686, 8.6455),     # Deutsche Bank Park, Frankfurt
    "MEX00": (19.3029, -99.1505),   # Estadio Azteca / Banorte, Mexico City
    "SAO00": (-23.5453, -46.4742),  # Arena Corinthians, Sao Paulo
    "MAD01": (40.4531, -3.6883),    # Santiago Bernabeu, Madrid
    "MEL00": (-37.8200, 144.9834),  # Melbourne Cricket Ground
    "PAR00": (48.9245, 2.3601),     # Stade de France, Paris
    "RIO00": (-22.9122, -43.2302),  # Maracana, Rio de Janeiro
}


def ensure_data_dirs() -> None:
    """Create the (git-ignored) data folders if they don't exist yet."""
    for d in (RAW_DIR, SNAPSHOT_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def season_sql(max_season=None) -> str:
    """The SQL season window every training/evaluation loader applies: FEATURE_HISTORY_START through
    cap_season(max_season). Raises HoldoutError for the holdout season or later."""
    return f" AND season >= {FEATURE_HISTORY_START} AND season <= {cap_season(max_season)}"


# --- Phase 2 weighting constants --------------------------------------------
RECENCY_HALF_LIFE_GAMES = 6
OFFSEASON_GAP_GAMES = 8  # an offseason counts as this many team games of decay
QUALITY_WEIGHTS = {"observed": 1.0, "derived": 0.9, "estimated": 0.65}

# Continuity penalties: market -> {factor: multiplier in (0, 1]}, applied when that
# factor changed between a past game and the game being predicted. Factors:
# QB, HC, OC, OL, role, RB_group, WR_TE_group.
# Starting values from the guide. A market missing from this table makes
# continuity_weight() raise, rather than silently meaning "no penalty".
CONTINUITY_FACTORS = ("QB", "HC", "OC", "OL", "role", "RB_group", "WR_TE_group")
CONTINUITY_PENALTIES: dict[str, dict[str, float]] = {
    "pass_yds":         {"QB": 0.25, "HC": 0.85, "OC": 0.65, "OL": 0.90, "role": 0.50, "RB_group": 0.95, "WR_TE_group": 0.80},
    "pass_tds":         {"QB": 0.25, "HC": 0.85, "OC": 0.65, "OL": 0.90, "role": 0.50, "RB_group": 0.95, "WR_TE_group": 0.75},
    "rush_yds":         {"QB": 0.90, "HC": 0.85, "OC": 0.80, "OL": 0.65, "role": 0.35, "RB_group": 0.65, "WR_TE_group": 0.95},
    "rush_att":         {"QB": 0.90, "HC": 0.85, "OC": 0.80, "OL": 0.65, "role": 0.35, "RB_group": 0.65, "WR_TE_group": 0.95},
    "rec_yds":          {"QB": 0.55, "HC": 0.85, "OC": 0.70, "OL": 0.95, "role": 0.35, "RB_group": 0.90, "WR_TE_group": 0.65},
    "rec":              {"QB": 0.55, "HC": 0.85, "OC": 0.70, "OL": 0.95, "role": 0.35, "RB_group": 0.90, "WR_TE_group": 0.65},
    "rec_tds":          {"QB": 0.50, "HC": 0.85, "OC": 0.70, "OL": 0.95, "role": 0.30, "RB_group": 0.88, "WR_TE_group": 0.60},
    "anytime_td":       {"QB": 0.60, "HC": 0.85, "OC": 0.70, "OL": 0.90, "role": 0.30, "RB_group": 0.70, "WR_TE_group": 0.70},
    "game_total":       {"QB": 0.55, "HC": 0.80, "OC": 0.70, "OL": 0.90, "role": 0.90, "RB_group": 0.95, "WR_TE_group": 0.90},
    "first_half_total": {"QB": 0.55, "HC": 0.80, "OC": 0.70, "OL": 0.90, "role": 0.90, "RB_group": 0.95, "WR_TE_group": 0.90},
    "team_total":       {"QB": 0.50, "HC": 0.80, "OC": 0.68, "OL": 0.88, "role": 0.90, "RB_group": 0.92, "WR_TE_group": 0.88},
}

# --- Phase 2 baseline constants ---------------------------------------------
# The 8 player markets and 3 game markets the baselines cover (names differ from the
# continuity-penalty market names above; this maps the former onto the latter).
# NOTE: this mapping is an assumption -- confirm it before continuity weights are
# applied to baselines/models.
# The single market table. Per market: kind ("player"/"game"), `baseline_family` (which baseline routine
# family scores it: the per-player functions or predict_game), `stat` (player_stats column for player
# markets), `pool` (position family -> depth-chart slots that are scored; empty for game markets) and
# `penalty_key` (its row in CONTINUITY_PENALTIES). Key order is iteration order everywhere.
_RECV_POOL = {"WR": (1, 2, 3), "TE": (1, 2), "RB": (1, 2)}
MARKETS = {
    "pass_att": dict(kind="player", baseline_family="player", stat="attempts", pool={"QB": (1,)}, penalty_key="pass_yds"),
    "pass_cmp": dict(kind="player", baseline_family="player", stat="completions", pool={"QB": (1,)}, penalty_key="pass_yds"),
    "pass_yds": dict(kind="player", baseline_family="player", stat="passing_yards", pool={"QB": (1,)}, penalty_key="pass_yds"),
    "rush_att": dict(kind="player", baseline_family="player", stat="carries", pool={"RB": (1, 2)}, penalty_key="rush_att"),
    "rush_yds": dict(kind="player", baseline_family="player", stat="rushing_yards", pool={"RB": (1, 2)}, penalty_key="rush_yds"),
    "targets": dict(kind="player", baseline_family="player", stat="targets", pool=_RECV_POOL, penalty_key="rec"),
    "rec": dict(kind="player", baseline_family="player", stat="receptions", pool=_RECV_POOL, penalty_key="rec"),
    "rec_yds": dict(kind="player", baseline_family="player", stat="receiving_yards", pool=_RECV_POOL, penalty_key="rec_yds"),
    "spread": dict(kind="game", baseline_family="game", stat=None, pool={}, penalty_key="game_total"),
    "moneyline": dict(kind="game", baseline_family="game", stat=None, pool={}, penalty_key="game_total"),
    "total": dict(kind="game", baseline_family="game", stat=None, pool={}, penalty_key="game_total"),
}
# Alias kept for existing callers: market -> continuity-penalty key.
BASELINE_TO_PENALTY_MARKET = {m: spec["penalty_key"] for m, spec in MARKETS.items()}
BLEND_OWN_WEIGHT = 0.7  # player/team own average; the rest is the opponent-allowed average
ROLE_WINDOW_DAYS = 365  # trailing window for role and opponent-allowed averages
GAME_MARGIN_SD = 13.5   # NFL final-margin std dev (pts); turns a predicted margin into P(home win)

# --- Phase 2 grading constants ------------------------------------------------
# Threshold ladders: rung t means the event "actual > t" (half-integers, so no ties).
# The rung values are my starting picks -- adjust to the lines you care about.
# moneyline has a single event, "home team wins", predicted directly by the baseline.
LADDERS = {
    "pass_att": [24.5, 29.5, 34.5, 39.5],
    "pass_cmp": [14.5, 19.5, 24.5, 29.5],
    "pass_yds": [174.5, 224.5, 274.5, 324.5],
    "rush_att": [7.5, 12.5, 17.5, 22.5],
    "rush_yds": [29.5, 49.5, 69.5, 99.5],
    "targets": [3.5, 5.5, 7.5, 9.5],
    "rec": [2.5, 4.5, 6.5, 8.5],
    "rec_yds": [29.5, 49.5, 74.5, 99.5],
    "spread": [-6.5, -3.5, -0.5, 2.5, 6.5],   # home margin > t
    "total": [38.5, 41.5, 44.5, 47.5, 50.5],
}
SKILL_REFERENCE_METHOD = "season_avg"  # the baseline every method's Brier skill is measured against
MIN_RESIDUALS = 200       # prior residuals needed before a baseline gets a predictive distribution
MIN_STRATUM_RESIDUALS = 60  # prior residuals needed in a predicted-value stratum before it is used
N_STRATA = 3              # predicted-value strata (terciles of earlier predictions)
PIT_JITTER_SEED = 20240    # seeded continuity jitter for PIT of integer outcomes (reproducibility)
CALIBRATION_BAND_WIDTH = 0.2

# --- Phase 3 comparison constants ----------------------------------------------
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20241
BOOTSTRAP_CI = 0.95
MIN_SEASONS_WON = 2  # the phase-3 gate: the model must beat the best baseline in more than one season separately

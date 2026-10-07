"""Central constants for script-the-slate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
PROCESSED_DIR = DATA_DIR / "processed"

DUCKDB_PATH = DATA_DIR / "script_the_slate.duckdb"

BACKTEST_SEASONS = list(range(2020, 2025))  # 2020-2024 inclusive
HOLDOUT_SEASON = 2025  # locked: never used for tuning or model selection

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

# TODO(1.6): neutral-site / international venues (London, Germany, Brazil,
# Mexico City, ...) are not in stadium_coordinates; add them with the weather pull.


def ensure_data_dirs() -> None:
    """Create the (git-ignored) data folders if they don't exist yet."""
    for d in (RAW_DIR, SNAPSHOT_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

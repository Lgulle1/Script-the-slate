"""4c.0 -- component, market and search definitions for the comparables engine. CONSTANTS ONLY: no logic. Every other 4c task imports from here.

Nothing below is to be added, dropped or renamed without asking. Where a requested feature has no source in the loaded tables it is listed in
DROPPED (with the reason), not silently left out. Every feature carries a data-quality tag (observed / derived / estimated, QUALITY_RULE) that
the fingerprint vectors store and the similarity search weights with config.QUALITY_WEIGHTS.

Source tables (data/raw/raw.duckdb): pbp (2016-), ftn_charting (2022-), pfr_advstats_pass / _rush / _rec (2018-), snap_counts (2016-),
rosters_weekly (2016-), participation (2016-; man/zone and coverage type only partly filled before 2023). 2025+ is the locked holdout and is
never part of a pool used for evaluation (config.cap_season).
"""
from __future__ import annotations

from typing import NamedTuple


class Feature(NamedTuple):
    name: str
    source: str            # table(s) the columns come from
    columns: tuple         # the raw columns the feature is computed from
    space: str             # "base" or "extended"
    first_season: int      # first season with the source columns populated (>= 50% of rows)
    quality: str           # "observed" | "derived" | "estimated" (see QUALITY_RULE)
    note: str              # one-line definition


# ---------------------------------------------------------------------------------------------------------------------------------- 3. SPACES
BASE_START = 2016          # BASE = play-by-play features (plus snap_counts / rosters_weekly, which cover the same seasons: see PENDING_APPROVAL 1)
PFR_START = 2018           # EXTENDED adds PFR (2018 on)
FTN_START = 2022           # EXTENDED adds FTN (2022 on)
SPACES = ("base", "extended")      # two separate pools: never mixed in one nearest-neighbour search

# ---------------------------------------------------------------------------------------------------------------------------------- 1. UNITS
OFFENSE_UNITS = ("run_offense", "pass_offense", "rb_rotation", "ol_protection", "receiver_usage")
DEFENSE_UNITS = ("run_defense", "pass_rush", "pass_coverage", "coverage_mix")
ARCHETYPE_UNITS = ("rb_archetype", "receiver_archetype", "qb_archetype")      # one vector per player per window
UNITS = OFFENSE_UNITS + DEFENSE_UNITS + ARCHETYPE_UNITS

# thresholds the feature definitions use
EXPLOSIVE_RUN_YARDS = 10           # given in the request
EXPLOSIVE_PASS_YARDS = 20          # PROPOSED (not in the request): see PENDING_APPROVAL 2
GOAL_LINE_YARDLINE = 5             # PROPOSED (not in the request): carries with yardline_100 <= this count as goal-line: see PENDING_APPROVAL 2
RB_POSITIONS = ("RB", "FB")        # rosters_weekly position values counted as the RB group
MIN_PLAYS_RUN_DIRECTION = 30       # 4c S5: a run-direction interaction needs at least this many plays

_P = "pbp"


def _f(name, source, columns, space, first_season, note, quality="observed"):
    return Feature(name, source, tuple(columns), space, first_season, quality, note)


# ---------------------------------------------------------------------------------------------------------------------------------- 2. FEATURES
FEATURES = {
    "run_offense": (
        _f("rush_epa_per_play", _P, ["epa", "rush", "qb_kneel"], "base", 2016, "mean epa on designed team rush plays (kneels out)"),
        _f("rush_success_rate", _P, ["success", "rush", "qb_kneel"], "base", 2016, "mean success on the same plays"),
        _f("yards_per_carry", _P, ["yards_gained", "rush", "qb_kneel"], "base", 2016, "mean yards_gained on the same plays"),
        _f("explosive_run_rate", _P, ["yards_gained", "rush"], "base", 2016, f"share of those plays with yards_gained >= {EXPLOSIVE_RUN_YARDS}"),
        _f("rush_rate_over_expected", _P, ["pass_oe"], "base", 2016, "minus the mean of pass_oe over all team run/pass plays (derived; pass_oe is the pbp pass-over-expected)", quality="derived"),
        _f("run_location_share_left", _P, ["run_location", "rush"], "base", 2016, "share of rush plays with a recorded run_location that went left"),
        _f("run_location_share_middle", _P, ["run_location", "rush"], "base", 2016, "same, middle"),
        _f("run_location_share_right", _P, ["run_location", "rush"], "base", 2016, "same, right"),
        _f("run_gap_share_end", _P, ["run_gap", "rush"], "base", 2016, "share of rush plays with a recorded run_gap at the end"),
        _f("run_gap_share_tackle", _P, ["run_gap", "rush"], "base", 2016, "same, tackle"),
        _f("run_gap_share_guard", _P, ["run_gap", "rush"], "base", 2016, "same, guard"),
        _f("shotgun_run_share", _P, ["shotgun", "rush"], "base", 2016, "share of rush plays run from shotgun"),
    ),
    "pass_offense": (
        _f("epa_per_dropback", _P, ["epa", "qb_dropback"], "base", 2016, "mean epa over team dropbacks (sacks and scrambles included)"),
        _f("cpoe", _P, ["cpoe", "pass"], "base", 2016, "mean cpoe over team pass attempts that have one"),
        _f("yards_per_attempt", _P, ["yards_gained", "pass", "sack"], "base", 2016, "mean yards_gained over team pass attempts (sacks out)"),
        _f("explosive_pass_rate", _P, ["yards_gained", "pass", "sack"], "base", 2016, f"share of pass attempts with yards_gained >= {EXPLOSIVE_PASS_YARDS}"),
        _f("average_depth_of_target", _P, ["air_yards", "pass"], "base", 2016, "mean air_yards over pass attempts that have it"),
        _f("sack_rate", _P, ["sack", "qb_dropback"], "base", 2016, "sacks / dropbacks"),
        _f("play_action_rate", "ftn_charting", ["is_play_action"], "extended", 2022, "share of pass plays flagged play-action"),
        _f("screen_rate", "ftn_charting", ["is_screen_pass"], "extended", 2022, "share of pass plays flagged screen"),
        _f("motion_rate", "ftn_charting", ["is_motion"], "extended", 2022, "share of team plays with pre-snap motion"),
        _f("no_huddle_rate", _P, ["no_huddle"], "base", 2016, "share of team run/pass plays in no-huddle (pbp has it for every season, so it is BASE)"),
    ),
    "rb_rotation": (
        _f("rb1_carry_share", f"{_P}, rosters_weekly", ["rusher_player_id", "rush", "position"], "base", 2016, "RB1 = the team's RB with the most carries that game; his share of team RB carries", quality="derived"),
        _f("rb2_carry_share", f"{_P}, rosters_weekly", ["rusher_player_id", "rush", "position"], "base", 2016, "same, the second RB", quality="derived"),
        _f("rb1_snap_share", "snap_counts", ["offense_pct", "position"], "base", 2016, "RB1's offense_pct that game", quality="derived"),
        _f("rb_target_share", f"{_P}, rosters_weekly", ["receiver_player_id", "pass", "position"], "base", 2016, "share of team targets that went to RBs", quality="derived"),
        _f("goal_line_carry_share", f"{_P}, rosters_weekly", ["rusher_player_id", "yardline_100", "rush"], "base", 2016, f"RB1's share of team carries with yardline_100 <= {GOAL_LINE_YARDLINE}", quality="derived"),
    ),
    "ol_protection": (
        _f("sack_rate_allowed", _P, ["sack", "qb_dropback"], "base", 2016, "sacks taken / dropbacks"),
        _f("pressure_rate_allowed", "pfr_advstats_pass", ["times_pressured", "team"], "extended", 2018, "sum over the team's QB rows of times_pressured / team dropbacks (pbp)", quality="derived"),
        _f("rush_epa_left", _P, ["epa", "run_location", "rush"], "base", 2016, "mean epa on rushes with run_location = left (run-block proxy)"),
        _f("rush_epa_middle", _P, ["epa", "run_location", "rush"], "base", 2016, "same, middle"),
        _f("rush_epa_right", _P, ["epa", "run_location", "rush"], "base", 2016, "same, right"),
    ),
    "receiver_usage": (
        _f("top3_target_share", _P, ["receiver_player_id", "pass"], "base", 2016, "sum of the three largest player target shares of the team", quality="derived"),
        _f("target_split_wr", f"{_P}, rosters_weekly", ["receiver_player_id", "position"], "base", 2016, "share of team targets to WRs", quality="derived"),
        _f("target_split_te", f"{_P}, rosters_weekly", ["receiver_player_id", "position"], "base", 2016, "same, TEs", quality="derived"),
        _f("target_split_rb", f"{_P}, rosters_weekly", ["receiver_player_id", "position"], "base", 2016, "same, RBs", quality="derived"),
        _f("adot_wr", f"{_P}, rosters_weekly", ["air_yards", "receiver_player_id", "position"], "base", 2016, "mean air_yards on targets to WRs", quality="derived"),
        _f("adot_te", f"{_P}, rosters_weekly", ["air_yards", "receiver_player_id", "position"], "base", 2016, "same, TEs", quality="derived"),
        _f("adot_rb", f"{_P}, rosters_weekly", ["air_yards", "receiver_player_id", "position"], "base", 2016, "same, RBs", quality="derived"),
    ),
    "run_defense": (
        _f("rush_epa_allowed", _P, ["epa", "rush", "qb_kneel", "defteam"], "base", 2016, "mean epa on designed rushes against the defense"),
        _f("rush_success_allowed", _P, ["success", "rush", "defteam"], "base", 2016, "mean success on those plays"),
        _f("yards_per_carry_allowed", _P, ["yards_gained", "rush", "defteam"], "base", 2016, "mean yards_gained on those plays"),
        _f("explosive_runs_allowed", _P, ["yards_gained", "rush", "defteam"], "base", 2016, f"share of those plays with yards_gained >= {EXPLOSIVE_RUN_YARDS}"),
        _f("box_count_faced", "ftn_charting", ["n_defense_box"], "extended", 2022, "mean defenders in the box on rush plays (the box count the defense showed)"),
        _f("rush_epa_allowed_left", _P, ["epa", "run_location", "defteam"], "base", 2016, "mean epa allowed on rushes with run_location = left"),
        _f("rush_epa_allowed_middle", _P, ["epa", "run_location", "defteam"], "base", 2016, "same, middle"),
        _f("rush_epa_allowed_right", _P, ["epa", "run_location", "defteam"], "base", 2016, "same, right"),
    ),
    "pass_rush": (
        _f("sack_rate", _P, ["sack", "qb_dropback", "defteam"], "base", 2016, "sacks / opposing dropbacks"),
        _f("pressure_rate", "pfr_advstats_pass", ["times_pressured", "opponent"], "extended", 2018,
           "times_pressured by the opposing QB / dropbacks faced (PFR def_times_* columns are empty in the table, so this comes from the offense-side rows grouped by opponent)", quality="derived"),
        _f("blitz_rate_pfr", "pfr_advstats_pass", ["times_blitzed", "opponent"], "extended", 2018, "opposing QB times_blitzed / dropbacks faced", quality="derived"),
        _f("blitz_rate_ftn", "ftn_charting", ["n_blitzers"], "extended", 2022, "share of pass plays with n_blitzers > 0 (kept apart from blitz_rate_pfr: different definitions)"),
        _f("qb_hit_rate", _P, ["qb_hit", "qb_dropback", "defteam"], "base", 2016, "qb_hit plays / opposing dropbacks"),
    ),
    "pass_coverage": (
        _f("epa_per_dropback_allowed", _P, ["epa", "qb_dropback", "defteam"], "base", 2016, "mean epa on opposing dropbacks"),
        _f("cpoe_allowed", _P, ["cpoe", "pass", "defteam"], "base", 2016, "mean cpoe on opposing attempts"),
        _f("yards_per_attempt_allowed", _P, ["yards_gained", "pass", "sack", "defteam"], "base", 2016, "mean yards_gained on opposing attempts"),
        _f("explosive_passes_allowed", _P, ["yards_gained", "pass", "sack", "defteam"], "base", 2016, f"share of opposing attempts with yards_gained >= {EXPLOSIVE_PASS_YARDS}"),
        _f("completion_rate_allowed_wr", f"{_P}, rosters_weekly", ["complete_pass", "receiver_player_id", "position"], "base", 2016, "completion rate on targets to WRs", quality="derived"),
        _f("completion_rate_allowed_te", f"{_P}, rosters_weekly", ["complete_pass", "receiver_player_id", "position"], "base", 2016, "same, TEs", quality="derived"),
        _f("completion_rate_allowed_rb", f"{_P}, rosters_weekly", ["complete_pass", "receiver_player_id", "position"], "base", 2016, "same, RBs", quality="derived"),
    ),
    "coverage_mix": (      # EXTENDED space only; the participation table (approved exception), quality `estimated`; first_season = first season the table classifies pass attempts (2018)
        _f("man_rate", "participation", ["defense_man_zone_type"], "extended", 2018, "share of classified pass attempts in man coverage (participation classifies ~100% of pass attempts 2018-2024; nothing before 2018)", quality="estimated"),
        _f("zone_rate", "participation", ["defense_man_zone_type"], "extended", 2018, "same, zone", quality="estimated"),
        _f("middle_closed_rate", "participation", ["defense_coverage_type"], "extended", 2018,
           "share of classified pass attempts in a single-high shell (COVER_1, COVER_3); two-high (COVER_2, 2_MAN, COVER_4, COVER_6) is middle open; 89-96% of attempts carry a shell 2018-2024", quality="estimated"),
    ),
    "rb_archetype": (
        _f("carry_share", f"{_P}, rosters_weekly", ["rusher_player_id", "rush"], "base", 2016, "player carries / team carries", quality="derived"),
        _f("yards_per_carry", _P, ["yards_gained", "rusher_player_id"], "base", 2016, "yards per carry"),
        _f("explosive_rate", _P, ["yards_gained", "rusher_player_id"], "base", 2016, f"share of carries with yards_gained >= {EXPLOSIVE_RUN_YARDS}"),
        _f("rush_epa", _P, ["epa", "rusher_player_id"], "base", 2016, "mean epa per carry"),
        _f("target_share", _P, ["receiver_player_id", "pass"], "base", 2016, "targets / team targets", quality="derived"),
        _f("yards_after_contact", "pfr_advstats_rush", ["rushing_yards_after_contact_avg"], "extended", 2018, "PFR yards after contact per carry"),
        _f("goal_line_share", _P, ["rusher_player_id", "yardline_100"], "base", 2016, f"share of the team's carries at yardline_100 <= {GOAL_LINE_YARDLINE} taken by the player", quality="derived"),
        _f("height", "rosters_weekly", ["height"], "base", 2016, "inches"),
        _f("weight", "rosters_weekly", ["weight"], "base", 2016, "pounds"),
    ),
    "receiver_archetype": (
        _f("target_share", _P, ["receiver_player_id", "pass"], "base", 2016, "targets / team targets", quality="derived"),
        _f("average_depth_of_target", _P, ["air_yards", "receiver_player_id"], "base", 2016, "mean air_yards per target"),
        _f("air_yards_share", _P, ["air_yards", "receiver_player_id"], "base", 2016, "player air_yards / team air_yards", quality="derived"),
        _f("yards_after_catch", _P, ["yards_after_catch", "receiver_player_id", "complete_pass"], "base", 2016, "mean yards_after_catch per reception"),
        _f("catch_rate_over_expected", _P, ["complete_pass", "cp", "receiver_player_id"], "base", 2016, "mean of (complete_pass - cp) over targets with a cp (derived: pbp has cp but no receiver-level CROE)", quality="derived"),
        _f("height", "rosters_weekly", ["height"], "base", 2016, "inches"),
        _f("weight", "rosters_weekly", ["weight"], "base", 2016, "pounds"),
    ),
    "qb_archetype": (
        _f("epa_per_dropback", _P, ["epa", "passer_player_id", "qb_dropback"], "base", 2016, "mean epa over the QB's dropbacks"),
        _f("cpoe", _P, ["cpoe", "passer_player_id"], "base", 2016, "mean cpoe"),
        _f("average_depth_of_target", _P, ["air_yards", "passer_player_id"], "base", 2016, "mean air_yards per attempt"),
        _f("sack_rate", _P, ["sack", "passer_player_id", "qb_dropback"], "base", 2016, "sacks / dropbacks"),
        _f("scramble_rate", _P, ["qb_scramble", "passer_player_id", "qb_dropback"], "base", 2016, "scrambles / dropbacks"),
        _f("rush_attempts_per_game", _P, ["rusher_player_id", "rush", "qb_kneel"], "base", 2016, "designed rushes by the QB per game played (kneels out, scrambles out)"),
        _f("play_action_rate", "ftn_charting", ["is_play_action"], "extended", 2022, "share of the QB's pass plays flagged play-action"),
    ),
}

# Requested features that have no source in the loaded tables. They are NOT in FEATURES.
DROPPED = (
    ("receiver_usage", "slot vs outside rate", "no alignment / slot flag in pbp, FTN, PFR, snap_counts or rosters (participation lists positions on the field, not alignments)"),
    ("receiver_archetype", "slot rate", "same: no slot / alignment source"),
)

# Decisions (all approved): coverage_mix reads the participation table (an approved exception to the BASE / EXTENDED source list, scoped to
# coverage_mix only) and is tagged `estimated` (the kind of data: tracking-inferred labels, not charted facts; this holds whatever the completeness, and
# stays after the measured completeness turned out higher than first quoted) with its per-season completeness logged (comp_completeness.parquet); BASE includes snap_counts and rosters_weekly (same 2016+
# coverage as pbp); EXPLOSIVE_PASS_YARDS = 20 and GOAL_LINE_YARDLINE = 5 stand as defined above.
QUALITY_RULE = ("observed = a direct aggregate of a provided column of one table; derived = needs another table, a rank, a share of the team "
                "total or a computed construct; estimated = the source is only partly populated (participation man/zone and coverage type)")
# Seasons in which participation classifies a play, as a share of plays (rounded; 4c.1 logs the exact value per season in comp_completeness)
PARTICIPATION_COMPLETENESS_NOTE = ("share of PASS ATTEMPTS classified (measured in 4c.1, comp_completeness.parquet): man/zone ~100% 2018-2024, coverage shell 89-96% 2018-2024; "
                                   "nothing before 2018. (The ~38% / ~49% quoted at 4c.0 were shares of ALL plays, where the table leaves run plays blank.)")

# ---------------------------------------------------------------------------------------------------------------------------------- 4. WINDOWS
WINDOWS = ("last_3", "last_6", "season_to_date", "recency_weighted", "continuity_weighted")      # 4c.6 selects one per unit and market

# ---------------------------------------------------------------------------------------------------------------------------------- 5. MARKET -> UNITS
MARKET_UNITS = {
    "pass_att": ("pass_offense", "ol_protection", "qb_archetype", "pass_rush", "pass_coverage", "coverage_mix"),
    "pass_cmp": ("pass_offense", "ol_protection", "qb_archetype", "pass_rush", "pass_coverage", "coverage_mix"),
    "pass_yds": ("pass_offense", "ol_protection", "qb_archetype", "pass_rush", "pass_coverage", "coverage_mix"),
    "rush_att": ("run_offense", "rb_rotation", "rb_archetype", "ol_protection", "run_defense"),
    "rush_yds": ("run_offense", "rb_rotation", "rb_archetype", "ol_protection", "run_defense"),
    "qb_rush_att": ("run_offense", "qb_archetype", "ol_protection", "run_defense"),
    "qb_rush_yds": ("run_offense", "qb_archetype", "ol_protection", "run_defense"),
    "targets": ("receiver_usage", "receiver_archetype", "pass_offense", "pass_coverage", "coverage_mix", "pass_rush"),
    "rec": ("receiver_usage", "receiver_archetype", "pass_offense", "pass_coverage", "coverage_mix", "pass_rush"),
    "rec_yds": ("receiver_usage", "receiver_archetype", "pass_offense", "pass_coverage", "coverage_mix", "pass_rush"),
    "total": ("run_offense", "pass_offense", "run_defense", "pass_rush", "pass_coverage"),          # team level only: no player archetypes
    "spread": ("run_offense", "pass_offense", "run_defense", "pass_rush", "pass_coverage"),
    "moneyline": ("run_offense", "pass_offense", "run_defense", "pass_rush", "pass_coverage"),
}

# ---------------------------------------------------------------------------------------------------------------------------------- 6. THE FIVE SEARCHES
SEARCHES = {
    "S1": ("own_history_vs_similar_defenses", "past games of the target offense or player (same team, same player) against defenses whose lineup-adjusted defensive unit vectors resemble tonight's opponent"),
    "S2": ("opponent_history_vs_similar_offenses", "past games of tonight's defense against offenses (and players) whose unit vectors resemble tonight's offense and the target player"),
    "S3": ("similar_players_vs_similar_defenses", "past games of players whose archetype vector resembles the target player, on any team, against defenses resembling tonight's defense"),
    "S4": ("similar_offenses_vs_similar_defenses", "past games in which an offense resembling tonight's (relevant units) faced a defense resembling tonight's (relevant units); weight uses BOTH similarities multiplied"),
    "S5": ("similar_matchup_structure", "past games matching tonight's interaction profile (EXTENDED space only); a missing interaction feature gets zero weight plus a completeness penalty; often returns no reliable comparable: expected, not a bug"),
}
S5_SPACE = "extended"
S5_INTERACTIONS = (
    "box_count_faced_vs_run_rate",
    "motion_usage_vs_defense_results_against_motion",
    "offense_explosive_play_rate_vs_defense_explosive_play_rate_allowed",
    "run_direction_share_vs_defense_run_direction_results",          # only where the sample is at least MIN_PLAYS_RUN_DIRECTION plays
    "ol_protection_vs_pass_rush",
    "play_action_usage_vs_defense_play_action_results",
    "screen_usage_vs_defense_screen_results",
    "blitz_rate_faced_vs_qb_results_against_the_blitz",
)

# ---------------------------------------------------------------------------------------------------------------------------------- 7. CONSTANTS
# starting values; tunable ONLY on 2020-2024 walk-forward results
SIM_THRESHOLD = 0.70
MIN_NEFF = 2
QUALITY = {"observed": 1.0, "derived": 0.9, "estimated": 0.65}
SHRINK_K = 5
TEAM_CAP_PER_SEARCH = 0.25
TEAM_CAP_AVG_ACROSS_SEARCHES = 0.15

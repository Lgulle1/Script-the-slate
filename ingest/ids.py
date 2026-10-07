"""Canonical id join helper: attach a single gsis_id from the master player table.

Join player data through here (and so through the `players` table from 1.1),
never around it.
"""
import logging

import duckdb
import polars as pl

import config
from ingest.pull_nflverse import PLAYERS_CURRENT, ensure_players_view

log = logging.getLogger(__name__)

# Fail loudly if more than this share of a batch can't be matched to a gsis_id.
MAX_UNMATCHED_FRACTION = config.MAX_UNMATCHED_FRACTION

# Input column -> master-table column. pfr_player_id is the name some nflverse
# tables (snap counts, PFR advstats) use for what the master table calls pfr_id.
ID_COLUMNS = {"gsis_id": "gsis_id", "pfr_id": "pfr_id", "espn_id": "espn_id", "pfr_player_id": "pfr_id"}


class IdMatchError(ValueError):
    """Too many rows could not be matched to a canonical gsis_id."""


def _load_master(db_path) -> pl.DataFrame:
    """The master id table = the players_current view (latest pull per player). Created on first use."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        has_view = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [PLAYERS_CURRENT]).fetchone()[0]
    finally:
        con.close()
    if not has_view:
        ensure_players_view(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f"SELECT gsis_id, pfr_id, espn_id FROM {PLAYERS_CURRENT} WHERE gsis_id IS NOT NULL").pl()
    finally:
        con.close()


def _lookup(master: pl.DataFrame, col: str) -> pl.DataFrame:
    """id -> gsis_id, dropping ids that map to more than one gsis_id (ambiguous)."""
    m = master.filter(pl.col(col).is_not_null()).select(pl.col(col).alias("_id"), pl.col("gsis_id").alias("_gsis")).unique()
    return m.filter(pl.col("_id").is_unique())


def add_canonical_gsis_id(df: pl.DataFrame, db_path=config.DUCKDB_PATH, master: pl.DataFrame | None = None,
                          max_unmatched: float = MAX_UNMATCHED_FRACTION) -> pl.DataFrame:
    """Return df with a canonical `gsis_id` column resolved via the master table.

    Tries each id column present, in order gsis_id, pfr_id, espn_id, pfr_player_id; a row's first
    id that resolves wins. A supplied gsis_id is itself checked against the master table. Rows
    that don't resolve keep a null gsis_id, but if their share exceeds `max_unmatched`
    an IdMatchError is raised instead of returning the frame.
    """
    present = [c for c in ID_COLUMNS if c in df.columns]
    if not present:
        raise ValueError(f"dataframe needs at least one of {list(ID_COLUMNS)}; has {df.columns}")
    if df.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("gsis_id")) if "gsis_id" not in df.columns else df
    master = master if master is not None else _load_master(db_path)

    work = df.with_row_index("_row")
    resolved = None
    for col in present:
        key = f"_key_{col}"
        mcol = ID_COLUMNS[col]
        look = _lookup(master, mcol).rename({"_id": key, "_gsis": f"_gsis_{col}"})
        work = work.with_columns(pl.col(col).cast(pl.String).alias(key)).join(look, on=key, how="left").drop(key)
        cand = pl.col(f"_gsis_{col}")
        resolved = cand if resolved is None else resolved.fill_null(cand)
    work = work.sort("_row")
    out = work.with_columns(resolved.alias("_canonical"))
    out = out.drop([c for c in out.columns if c.startswith("_gsis_")])
    if "gsis_id" in out.columns:
        out = out.drop("gsis_id")
    out = out.rename({"_canonical": "gsis_id"}).drop("_row")

    n_bad = out["gsis_id"].null_count()
    frac = n_bad / out.height
    if frac > max_unmatched:
        sample = df.filter(out["gsis_id"].is_null()).select(present).head(5).to_dicts()
        raise IdMatchError(
            f"{n_bad}/{out.height} rows ({frac:.1%}) did not match a gsis_id in the master player table "
            f"(limit {max_unmatched:.1%}); id columns tried: {present}; sample: {sample}"
        )
    if n_bad:
        log.warning("ids: %d/%d rows (%.2f%%) unmatched; gsis_id left null", n_bad, out.height, 100 * frac)
    return out

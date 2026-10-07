"""Load the hand-built coaching seed CSV into a DuckDB `coaches` table.

Reloading never clobbers hand corrections: rows already flagged
hand_corrected = TRUE in the table win over the CSV row with the same
(team, role, name). Use mark_hand_corrected() to fix a row after import.
"""
import duckdb
import polars as pl

import config

SEED_CSV = config.DATA_DIR / "coaches_seed.csv"
COACHES_TABLE = "coaches"
SEED_COLUMNS = ["team", "role", "name", "start_year_with_team", "prior_stops", "source_url", "confidence"]
KEY = ["team", "role", "name"]


def load_coaches(csv_path=SEED_CSV, db_path=config.DUCKDB_PATH) -> int:
    """Read the seed CSV into the coaches table; returns the table's row count."""
    if not csv_path.exists():
        raise FileNotFoundError(f"coaches seed not found: {csv_path}")
    seed = pl.read_csv(csv_path, infer_schema_length=0)  # all strings; no guessing
    missing = [c for c in SEED_COLUMNS if c not in seed.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    seed = seed.select(SEED_COLUMNS).with_columns(
        pl.col("start_year_with_team").cast(pl.Int32, strict=False),
        pl.lit(False).alias("hand_corrected"),
    )
    con = duckdb.connect(str(db_path))
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [COACHES_TABLE]
        ).fetchone()[0]
        if exists:
            kept = con.execute(f"SELECT * FROM {COACHES_TABLE} WHERE hand_corrected").pl()
            if kept.height:
                seed = pl.concat([seed.join(kept.select(KEY), on=KEY, how="anti"), kept.select(seed.columns)])
        con.register("seed_df", seed.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {COACHES_TABLE} AS SELECT * FROM seed_df")
        return con.execute(f"SELECT count(*) FROM {COACHES_TABLE}").fetchone()[0]
    finally:
        con.close()


def mark_hand_corrected(team: str, role: str, name: str, db_path=config.DUCKDB_PATH, **fields) -> None:
    """Apply a manual fix to one coaches row and flag it hand_corrected."""
    allowed = set(SEED_COLUMNS) - set(KEY)
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"cannot set {sorted(bad)}; allowed: {sorted(allowed)}")
    sets = ", ".join([f"{k} = ?" for k in fields] + ["hand_corrected = TRUE"])
    con = duckdb.connect(str(db_path))
    try:
        n = con.execute(
            f"UPDATE {COACHES_TABLE} SET {sets} WHERE team = ? AND role = ? AND name = ? RETURNING 1",
            [*fields.values(), team, role, name],
        ).fetchall()
        if not n:
            raise KeyError(f"no coaches row for ({team}, {role}, {name})")
    finally:
        con.close()


if __name__ == "__main__":
    print(f"coaches table now has {load_coaches()} rows")

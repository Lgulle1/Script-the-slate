"""The prediction store of Phase 5 (docs/overnight_plan.md "Phase 5 gates", P5.1 and P5.3).

One parquet file per write (<run_id>.parquet), never rewritten, in a directory of the caller's choice (records/predictions/ in production), plus an
append-only refusals.csv. A record is stored only before its game kicks off; a rerun writes version = previous max + 1 and leaves earlier versions
byte-identical; an identical rerun writes nothing; grading reads the newest version stored before kickoff.

Pure track: the record schema has no column derived from a sportsbook line, and an input with any other column is rejected. Unknown values (median, quantiles,
ladder, confidence, comp shifts) are null, never 0.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

RECORD_COLUMNS = [
    "prediction_id", "run_id", "run_timestamp", "stored_at", "kickoff_time", "version", "game_id", "player_id", "market", "model_version",
    "stack", "mean", "median", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "ladder", "confidence", "comp_shift", "no_match", "injury_status", "input_hash"]
INPUT_REQUIRED = ["game_id", "market", "kickoff_time", "model_version", "stack", "mean", "input_hash"]
INPUT_OPTIONAL = [c for c in RECORD_COLUMNS if c not in ("prediction_id", "run_id", "run_timestamp", "stored_at", "version") and c not in INPUT_REQUIRED]
KEY = ["game_id", "player_id", "market"]
SCHEMA = {"prediction_id": pl.String, "run_id": pl.String, "run_timestamp": pl.Datetime("us", "UTC"), "stored_at": pl.Datetime("us", "UTC"),
          "kickoff_time": pl.Datetime("us", "UTC"), "version": pl.Int32, "game_id": pl.String, "player_id": pl.String, "market": pl.String, "model_version": pl.String,
          "stack": pl.String, "mean": pl.Float64, "median": pl.Float64, **{q: pl.Float64 for q in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")},
          "ladder": pl.String, "confidence": pl.String, "comp_shift": pl.String, "no_match": pl.Boolean, "injury_status": pl.String, "input_hash": pl.String}
REFUSAL_COLUMNS = ["clock", "run_id", "game_id", "player_id", "market", "kickoff_time", "reason"]


@dataclass
class WriteResult:
    written: int = 0
    refused: int = 0
    unchanged: int = 0


def _utc(x) -> datetime:
    if x.tzinfo is None:
        raise ValueError("naive datetime: every time in the store is UTC with a timezone")
    return x.astimezone(timezone.utc)


def _prediction_id(game_id, player_id, market, version) -> str:
    return hashlib.sha256(f"{game_id}|{player_id}|{market}|{version}".encode()).hexdigest()[:20]


class PredictionStore:
    def __init__(self, root):
        self.root = Path(root)

    def load(self) -> pl.DataFrame:
        files = sorted(self.root.glob("*.parquet"))
        if not files:
            return pl.DataFrame(schema=SCHEMA)
        return pl.concat([pl.read_parquet(f) for f in files]).select(RECORD_COLUMNS)

    def _refuse(self, rows: pl.DataFrame, now: datetime, run_id: str, reason_of) -> None:
        if not rows.height:
            return
        path = self.root / "refusals.csv"
        self.root.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(REFUSAL_COLUMNS)
            for r in rows.iter_rows(named=True):
                w.writerow([now.isoformat(), run_id, r["game_id"], r.get("player_id"), r["market"], r["kickoff_time"].isoformat() if r["kickoff_time"] else "", reason_of(r)])

    def write(self, preds: pl.DataFrame, now: datetime, run_id: str) -> WriteResult:
        """Store `preds` as of the clock `now` (UTC). Refuses (and logs) a started game and a record without a mean; skips a record identical to its latest version."""
        now = _utc(now)
        extra = [c for c in preds.columns if c not in INPUT_REQUIRED + INPUT_OPTIONAL + ["player_id"]]
        if extra:
            raise ValueError(f"columns {extra} are not in the record schema (no sportsbook-derived or unknown column can enter a prediction)")
        missing = [c for c in INPUT_REQUIRED if c not in preds.columns]
        if missing:
            raise ValueError(f"missing columns {missing}")
        p = preds
        for c in ["player_id"] + INPUT_OPTIONAL:
            if c not in p.columns:
                p = p.with_columns(pl.lit(None, dtype=SCHEMA[c]).alias(c))
        p = p.with_columns(pl.col("kickoff_time").cast(pl.Datetime("us", "UTC")), pl.col("mean").cast(pl.Float64))
        result = WriteResult()
        started = p.filter(pl.col("kickoff_time") <= now)
        self._refuse(started, now, run_id, lambda r: f"kickoff {r['kickoff_time'].isoformat()} is not after the clock {now.isoformat()}")
        p = p.filter(pl.col("kickoff_time") > now)
        nomean = p.filter(pl.col("mean").is_null() | ~pl.col("mean").is_finite().fill_null(False))
        self._refuse(nomean, now, run_id, lambda r: "no mean: a market with a missing component is not a prediction" if r["mean"] is None else f"mean {r['mean']} is not finite")
        p = p.filter(pl.col("mean").is_not_null() & pl.col("mean").is_finite())
        result.refused = started.height + nomean.height
        if p.is_empty():
            return result
        existing = self.load()
        latest = (existing.sort("version").group_by(KEY, maintain_order=True).last()
                  .select(*KEY, prev_version="version", prev_model=pl.col("model_version"), prev_stack=pl.col("stack"), prev_hash=pl.col("input_hash"), prev_mean="mean")) if existing.height else None
        if latest is not None:
            p = p.join(latest, on=KEY, how="left", nulls_equal=True)
            same = (pl.col("prev_version").is_not_null() & (pl.col("prev_model") == pl.col("model_version")) & (pl.col("prev_stack") == pl.col("stack")) & (pl.col("prev_hash") == pl.col("input_hash"))
                    & (pl.col("prev_mean") == pl.col("mean")))
            result.unchanged = p.filter(same).height
            p = p.filter(~same).with_columns(version=(pl.col("prev_version").fill_null(0) + 1).cast(pl.Int32)).drop("prev_version", "prev_model", "prev_stack", "prev_hash", "prev_mean")
        else:
            p = p.with_columns(version=pl.lit(1, dtype=pl.Int32))
        if p.is_empty():
            return result
        p = p.with_columns(run_id=pl.lit(run_id), run_timestamp=pl.lit(now).cast(pl.Datetime("us", "UTC")), stored_at=pl.lit(now).cast(pl.Datetime("us", "UTC")))
        p = p.with_columns(prediction_id=pl.struct("game_id", "player_id", "market", "version").map_elements(
            lambda r: _prediction_id(r["game_id"], r["player_id"], r["market"], r["version"]), return_dtype=pl.String))
        out = p.select([pl.col(c).cast(SCHEMA[c]) for c in RECORD_COLUMNS]).sort(KEY)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{run_id}.parquet"
        if path.exists():
            raise FileExistsError(f"{path.name}: a run id is written once; earlier files are never touched")
        out.write_parquet(path, compression="zstd")
        result.written = out.height
        return result

    def gradable(self) -> pl.DataFrame:
        """The newest version of each record that was stored before its game's kickoff."""
        return latest_before_kickoff(self.load())


def latest_before_kickoff(t: pl.DataFrame) -> pl.DataFrame:
    return (t.filter(pl.col("stored_at") < pl.col("kickoff_time")).sort("version").group_by(KEY, maintain_order=True).last()
            .select(RECORD_COLUMNS).sort(KEY))

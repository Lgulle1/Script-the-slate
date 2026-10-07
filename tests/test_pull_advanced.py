import duckdb
import polars as pl

from ingest import pull_nflverse as pn


def _fake(seen):
    def loader(seasons, **kw):
        seen.append(list(seasons))
        return pl.DataFrame({"season": seasons})

    return loader


def test_min_season_clipping(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(pn.nflreadpy, "load_ftn_charting", _fake(seen))
    monkeypatch.setattr(pn.nflreadpy, "load_pfr_advstats", _fake(seen))
    db = tmp_path / "r.duckdb"
    assert pn.pull_ftn_charting([2020, 2021, 2022, 2023], db) == 2
    assert pn.pull_pfr_advstats_rec([2016, 2018, 2019], db) == 2
    assert seen == [[2022, 2023], [2018, 2019]]
    # nothing in range -> no call, no table
    assert pn.pull_ftn_charting([2020], db) == 0


def test_participation_skips_out_of_range_and_errors_and_is_tagged(tmp_path, monkeypatch):
    def loader(seasons):
        if seasons == [2018]:
            raise ValueError("boom")
        return pl.DataFrame({"season": seasons})

    monkeypatch.setattr(pn.nflreadpy, "load_participation", loader)
    db = tmp_path / "r.duckdb"
    n = pn.pull_participation([2015, 2017, 2018, 2025, 2026], db)
    assert n == 2  # only 2017 and 2025 loaded; 2015/2026 out of range, 2018 errored
    con = duckdb.connect(str(db))
    assert con.execute("SELECT count(*) FROM participation WHERE training_labels_only").fetchone()[0] == 2


def test_snapshot_pulls_keep_every_pull(tmp_path, monkeypatch):
    monkeypatch.setattr(pn.nflreadpy, "load_injuries", _fake([]))
    db = tmp_path / "r.duckdb"
    pn.pull_injuries([2024], db)
    pn.pull_injuries([2024], db)
    con = duckdb.connect(str(db))
    assert con.execute("SELECT count(*) FROM injuries").fetchone()[0] == 2

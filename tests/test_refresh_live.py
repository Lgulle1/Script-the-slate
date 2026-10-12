"""ingest/refresh_live.py pulls everything a live run reads, including the player-id master table (a fresh container's dry run failed without it)."""
from ingest import refresh_live as RL


def test_a_pull_includes_the_player_master_and_every_live_table(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(RL, "TABLES", {n: (lambda seasons, db, n=n: calls.append((n, tuple(seasons))) or 1) for n in ("schedules", "player_stats", "pbp")})
    monkeypatch.setattr(RL.pn, "pull_players", lambda *a, **k: calls.append(("players", ())) or 5)
    monkeypatch.setattr(RL, "current_season", lambda: 2026)
    out = RL.refresh(tmp_path / "raw.duckdb")
    assert ("players", ()) in calls and out["players"] == 5
    assert {c[0] for c in calls} == {"players", "schedules", "player_stats", "pbp"} and all(c[1] == (2026,) for c in calls if c[0] != "players")
    calls.clear()
    RL.bootstrap(tmp_path / "raw.duckdb")
    assert min(c[1][0] for c in calls if c[0] != "players") == 2018 and ("players", ()) in calls

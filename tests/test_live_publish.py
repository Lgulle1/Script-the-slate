"""Publishing run records to the data-snapshots branch (live/publish.py) without touching the working branch or its checkout."""
import subprocess
from pathlib import Path

import pytest

from live import publish as PB


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    _git(work, "config", "user.email", "t@t"); _git(work, "config", "user.name", "t")
    (work / "README").write_text("x"); _git(work, "add", "."); _git(work, "commit", "-qm", "init"); _git(work, "push", "-q", "origin", "main")
    _git(work, "branch", "data-snapshots"); _git(work, "push", "-q", "origin", "data-snapshots")
    return origin, work


def test_records_land_on_data_snapshots_and_the_working_branch_is_untouched(repos):
    origin, work = repos
    (work / "records" / "predictions").mkdir(parents=True)
    f = work / "records" / "predictions" / "run_a.parquet"; f.write_bytes(b"abc")
    head_before = _git(work, "rev-parse", "HEAD")
    sha = PB.publish(work, [f], "predictions run_a", branch="data-snapshots", remote="origin")
    assert sha and _git(origin, "rev-parse", "data-snapshots") == sha
    assert _git(work, "rev-parse", "HEAD") == head_before and _git(work, "rev-parse", "--abbrev-ref", "HEAD") == "main"             # no commit on the working branch
    assert _git(origin, "show", "data-snapshots:records/predictions/run_a.parquet") == "abc"
    assert not (work / ".git" / "worktrees").exists() or not list((work / ".git" / "worktrees").iterdir())                       # the temporary worktree is gone


def test_a_second_publish_adds_a_commit_and_never_rewrites_earlier_files(repos):
    origin, work = repos
    (work / "records").mkdir()
    a = work / "records" / "a.csv"; a.write_text("1")
    PB.publish(work, [a], "one", branch="data-snapshots", remote="origin")
    b = work / "records" / "b.csv"; b.write_text("2")
    PB.publish(work, [b], "two", branch="data-snapshots", remote="origin")
    assert _git(origin, "show", "data-snapshots:records/a.csv") == "1" and _git(origin, "show", "data-snapshots:records/b.csv") == "2"
    assert len(_git(origin, "log", "--format=%H", "data-snapshots").splitlines()) == 3                                              # init + two


def test_publishing_only_files_under_records_and_never_to_another_branch(repos):
    _, work = repos
    (work / "secret.env").write_text("k=1")
    with pytest.raises(ValueError, match="records"):
        PB.publish(work, [work / "secret.env"], "no", branch="data-snapshots", remote="origin")
    with pytest.raises(ValueError, match="data-snapshots"):
        PB.publish(work, [], "no", branch="main", remote="origin")


def test_an_unchanged_file_publishes_nothing(repos):
    _, work = repos
    (work / "records").mkdir()
    a = work / "records" / "a.csv"; a.write_text("1")
    first = PB.publish(work, [a], "one", branch="data-snapshots", remote="origin")
    assert PB.publish(work, [a], "again", branch="data-snapshots", remote="origin") is None and first


def test_sync_brings_the_published_records_into_a_fresh_checkout_so_versions_continue(repos, tmp_path):
    origin, work = repos
    (work / "records" / "predictions").mkdir(parents=True)
    f = work / "records" / "predictions" / "run_a.parquet"; f.write_bytes(b"abc")
    PB.publish(work, [f], "one", branch="data-snapshots", remote="origin")
    fresh = tmp_path / "fresh"
    _git(tmp_path, "clone", "-q", str(origin), str(fresh))
    assert not (fresh / "records").exists()
    n = PB.sync_records(fresh, ["records/predictions", "records/lines"])
    assert n == 1 and (fresh / "records" / "predictions" / "run_a.parquet").read_bytes() == b"abc"
    assert PB.sync_records(fresh, ["records/predictions", "records/lines"]) == 1                    # idempotent; a path that does not exist yet (records/lines) is not an error


def test_new_files_under_an_ignored_records_directory_are_still_published(repos):
    origin, work = repos
    (work / ".gitignore").write_text("records/predictions/\n"); _git(work, "add", ".gitignore"); _git(work, "commit", "-qm", "ignore"); _git(work, "push", "-q", "origin", "main")
    (work / "records" / "predictions").mkdir(parents=True)
    f = work / "records" / "predictions" / "run_b.parquet"; f.write_bytes(b"zz")
    assert PB.publish(work, [f], "ignored dir", branch="data-snapshots", remote="origin")

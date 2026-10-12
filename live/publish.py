"""Publish run records to the data-snapshots branch (plan P5.3: the commit time is an outside timestamp for every stored prediction).

Uses a temporary git worktree of the remote branch, so the working branch and its checkout are never touched. Only files under records/ are published, only to
the data-snapshots branch, and never with force.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

BRANCH = "data-snapshots"


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def publish(repo, files, message: str, branch: str = BRANCH, remote: str = "origin", attempts: int = 4):
    """Copy `files` (paths under <repo>/records) onto the remote `branch`, commit, push; returns the new commit sha, or None when nothing changed."""
    if branch != BRANCH:
        raise ValueError(f"run records go only to {BRANCH}, not {branch}")
    repo = Path(repo).resolve()
    rels = []
    for f in files:
        rel = Path(f).resolve().relative_to(repo)
        if rel.parts[0] != "records":
            raise ValueError(f"{rel}: only files under records/ are published")
        rels.append(rel)
    for attempt in range(attempts):
        _git(repo, "fetch", "-q", remote, branch)
        tmp = Path(tempfile.mkdtemp(prefix="publish_"))
        wt = tmp / "wt"
        try:
            _git(repo, "worktree", "add", "-q", "--detach", str(wt), f"{remote}/{branch}")
            _git(wt, "config", "user.email", _git(repo, "config", "user.email", check=False).stdout.strip() or "records@script-the-slate")
            _git(wt, "config", "user.name", _git(repo, "config", "user.name", check=False).stdout.strip() or "script-the-slate records")
            for rel in rels:
                dst = wt / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / rel, dst)
            _git(wt, "add", "-f", "--", *[str(r) for r in rels])
            if not _git(wt, "status", "--porcelain").stdout.strip():
                return None
            _git(wt, "commit", "-q", "-m", message)
            sha = _git(wt, "rev-parse", "HEAD").stdout.strip()
            pushed = _git(wt, "push", "-q", remote, f"HEAD:refs/heads/{branch}", check=False)
            if pushed.returncode == 0:
                return sha
            if attempt == attempts - 1:
                raise RuntimeError(f"push to {remote}/{branch} failed after {attempts} attempts: {pushed.stderr.strip()}")
        finally:
            _git(repo, "worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(tmp, ignore_errors=True)
    return None


def sync_records(repo, paths, branch: str = BRANCH, remote: str = "origin") -> int:
    """Copy the given record paths (files or directories under records/) from the remote data-snapshots branch into the working tree, so a run in a fresh
    checkout continues the published version numbers. Returns the number of files brought in; a path the branch does not have yet is skipped."""
    repo = Path(repo).resolve()
    _git(repo, "fetch", "-q", remote, branch)
    n = 0
    for p in paths:
        if p.split("/")[0] != "records":
            raise ValueError(f"{p}: only records/ paths are synced")
        listed = _git(repo, "ls-tree", "-r", "--name-only", f"{remote}/{branch}", "--", p, check=False).stdout.split()
        if not listed:
            continue
        _git(repo, "checkout", f"{remote}/{branch}", "--", *listed)
        _git(repo, "reset", "-q", "--", *listed, check=False)                        # leave the index as it was: these files belong to the other branch
        n += len(listed)
    return n

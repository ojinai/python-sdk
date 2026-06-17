#!/usr/bin/env python3
"""Decide whether a PyPI release is needed, based on the version in pyproject.toml.

Used by ``.github/workflows/release.yml``. On a push to ``main`` this compares the
``version`` in pyproject.toml against the previous commit (``HEAD~1``). Because we
squash-merge, ``HEAD~1`` is the previous release state, so a changed version means a
new release should be built and published; an unchanged version means the run stops.

Subcommands (each writes to ``$GITHUB_OUTPUT``, or to stdout when run locally):

    extract_current_version   ->  VERSION=<x.y.z>
    check_build [--force]      ->  SHOULD_BUILD=<true|false>
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")


def extract_version(text: str) -> str:
    """Return the project version from pyproject.toml text, or ``not_found``."""
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "not_found"


def current_version() -> str:
    """Read the version from the working-tree pyproject.toml."""
    try:
        with open(PYPROJECT_PATH, encoding="utf-8") as handle:
            return extract_version(handle.read())
    except FileNotFoundError:
        return "not_found"


def previous_version() -> str:
    """Read the version from pyproject.toml as of the previous commit (``HEAD~1``)."""
    result = subprocess.run(
        ["git", "show", "HEAD~1:pyproject.toml"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "not_found"
    return extract_version(result.stdout)


def should_build(force: bool = False) -> bool:
    """Return True when a release should be built for the current version."""
    current = current_version()
    if current == "not_found":
        print("Could not read current version; nothing to publish.", file=sys.stderr)
        return False
    if force:
        print("Force flag set — build needed.", file=sys.stderr)
        return True
    previous = previous_version()
    print(f"current={current} previous={previous}", file=sys.stderr)
    # A missing previous version (first commit / new file) is treated as a new release.
    return previous == "not_found" or current != previous


def set_output(name: str, value: str) -> None:
    """Write a workflow output, or print it when run outside GitHub Actions."""
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main() -> None:
    """Dispatch the requested subcommand."""
    args = sys.argv[1:]
    force = "--force" in args
    if args and args[0] == "extract_current_version":
        set_output("VERSION", current_version())
    elif args and args[0] == "check_build":
        set_output("SHOULD_BUILD", str(should_build(force)).lower())
    else:
        print("usage: release_check.py [extract_current_version|check_build] [--force]")
        sys.exit(1)


if __name__ == "__main__":
    main()

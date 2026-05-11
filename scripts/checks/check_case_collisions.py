#!/usr/bin/env python3
"""
Fail if the repo contains two tracked paths that collide on case-insensitive filesystems.

The check uses `git ls-files` so it sees tracked sources only and ignores build outputs such as
`.lake/`.
"""

from __future__ import annotations

import collections
import subprocess
import sys


def main() -> int:
    """Scan tracked git paths and report names that collide after case-folding."""
    out = subprocess.check_output(["git", "ls-files", "-z"], text=True)
    paths = [p for p in out.split("\0") if p]

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for p in paths:
        # Guardrails in case someone ever adds these to the index.
        if p.startswith(".lake/") or "/.lake/" in p:
            continue
        if p.startswith(".git/") or "/.git/" in p:
            continue
        groups[p.casefold()].append(p)

    dups = {k: v for k, v in groups.items() if len(v) > 1}
    if not dups:
        print("ok: no tracked case-insensitive path collisions")
        return 0

    print("error: tracked case-insensitive path collisions found:", file=sys.stderr)
    for k, v in sorted(dups.items()):
        print(f"  {k}", file=sys.stderr)
        for p in sorted(v):
            print(f"    {p}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

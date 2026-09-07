#!/usr/bin/env python3
"""Move the training run registry between a runner and the Hub."""
from __future__ import annotations

import argparse

from bsdraft.registry_sync import SyncError, pull_registry, push_registry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["pull", "push"])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--path", default="runs/registry.db")
    p.add_argument("--allow-missing", action="store_true",
                   help="Exit 0 when no registry exists yet — the first run")
    args = p.parse_args()

    try:
        if args.command == "pull":
            found = pull_registry(args.repo_id, args.path)
            if not found and not args.allow_missing:
                raise SystemExit("error: no registry stored yet")
            print("restored" if found else "no registry yet; starting one")
        else:
            print(push_registry(args.repo_id, args.path))
    except SyncError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect and fetch training data."""
from __future__ import annotations

import argparse
import json

from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import (
    DEFAULT_REPO_ID,
    DatasetError,
    available_seasons,
    load_matches,
    resolve_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("seasons", help="Seasons available in the published dataset")

    res = sub.add_parser("resolve", help="Pin a season to an exact dataset commit")
    res.add_argument("season")
    res.add_argument("--revision", default=None)

    sm = sub.add_parser("summary", help="Load a season and describe what came back")
    sm.add_argument("season")
    sm.add_argument("--revision", default=None)
    sm.add_argument("--elo-min", type=float, default=10.0)
    sm.add_argument("--elo-max", type=float, default=23.0)
    sm.add_argument("--games", action="store_true",
                    help="Expand sets into per-game rows, as training does")

    args = p.parse_args()
    try:
        if args.command == "seasons":
            for s in available_seasons(repo_id=args.repo_id):
                print(s)
            return

        ref = resolve_dataset(args.season, repo_id=args.repo_id, revision=args.revision)
        if args.command == "resolve":
            print(json.dumps(
                {"repo_id": ref.repo_id, "season": ref.season, "revision": ref.revision},
                indent=2))
            return

        if args.games:
            df, vocab = build_game_dataset(ref, elo_min=args.elo_min, elo_max=args.elo_max)
            extra = {"games": len(df), "brawler_vocab": len(vocab)}
        else:
            df = load_matches(ref, elo_min=args.elo_min, elo_max=args.elo_max)
            extra = {"sets": len(df)}
        print(json.dumps({
            "dataset": str(ref),
            **extra,
            "modes": int(df["mode"].nunique()),
            "maps": int(df["map"].nunique()),
            "first_match": str(df["battle_time"].min()),
            "last_match": str(df["battle_time"].max()),
            "avg_elo_mean": round(float(df["avg_elo"].mean()), 3),
        }, indent=2))
    except DatasetError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()

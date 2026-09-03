#!/usr/bin/env python3
"""Look at what training runs have happened, and compare them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bsdraft.config import REPO_ROOT
from bsdraft.tracking import RunStore

DEFAULT_STORE = REPO_ROOT / "runs" / "registry.db"


def _fmt(v, width=10):
    if v is None:
        return "—".rjust(width)
    if isinstance(v, float):
        return f"{v:>{width}.4f}" if abs(v) < 1e4 else f"{v:>{width}.3g}"
    return f"{v:>{width}}"


def cmd_list(store: RunStore, args) -> int:
    rows = store.metric_table(stage=args.stage, limit=args.limit)
    if not rows:
        print("No runs recorded yet.")
        return 0
    print(f"{'run_id':<34} {'status':<8} {'seed':>8} {'logloss':>10} {'auc':>10} {'elapsed':>9}")
    print("-" * 84)
    for r in rows:
        print(f"{r['run_id']:<34} {r['status']:<8} {_fmt(r.get('seed'), 8)}"
              f" {_fmt(r.get('val_logloss'))} {_fmt(r.get('val_auc'))}"
              f" {_fmt(r.get('elapsed_seconds'), 9)}")
    return 0


def cmd_show(store: RunStore, args) -> int:
    run = store.get_run(args.run_id)
    if run is None:
        raise SystemExit(f"error: no run {args.run_id!r}")
    run.pop("history", None) if not args.history else None
    print(json.dumps(run, indent=2, default=str))
    return 0


def cmd_best(store: RunStore, args) -> int:
    run = store.best(args.metric, stage=args.stage, mode=args.mode)
    if run is None:
        raise SystemExit(f"error: no completed run has a {args.metric!r} metric")
    print(f"{run['run_id']}  {args.metric}={run['metrics'].get(args.metric)}")
    for a in run["artifacts"]:
        print(f"  {a['name']:<24} {a['path']}")
    return 0


def cmd_compare(store: RunStore, args) -> int:
    runs = [store.get_run(r) for r in args.run_ids]
    missing = [r for r, got in zip(args.run_ids, runs, strict=True) if got is None]
    if missing:
        raise SystemExit(f"error: no run(s) {missing}")

    keys = sorted({k for r in runs for k in r["metrics"]})
    width = max((len(k) for k in keys), default=6) + 2
    print(f"{'metric':<{width}}" + "".join(f"{r['run_id'][-13:]:>16}" for r in runs))
    print("-" * (width + 16 * len(runs)))
    for k in keys:
        print(f"{k:<{width}}" + "".join(_fmt(r["metrics"].get(k), 16) for r in runs))

    # Differences in config are usually the reason the metrics differ.
    print("\nconfig differences")
    flat = [dict(_flatten(r["config"])) for r in runs]
    diff = [k for k in sorted({k for f in flat for k in f})
            if len({str(f.get(k)) for f in flat}) > 1]
    if not diff:
        print("  none — same configuration")
    for k in diff:
        print(f"  {k:<{width}}" + "".join(f"{str(f.get(k)):>16}" for f in flat))
    return 0


def _flatten(d, prefix=""):
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, f"{key}.")
        else:
            yield key, v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=str(DEFAULT_STORE), help="Path to registry.db")
    sub = p.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="Recent runs with their headline metrics")
    ls.add_argument("--stage", default=None, choices=["fm", "selfplay"])
    ls.add_argument("--limit", type=int, default=25)
    ls.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="Everything recorded about one run")
    sh.add_argument("run_id")
    sh.add_argument("--history", action="store_true", help="Include per-step metrics")
    sh.set_defaults(func=cmd_show)

    bt = sub.add_parser("best", help="The run that scored best on a metric")
    bt.add_argument("--metric", default="val_logloss")
    bt.add_argument("--mode", default="min", choices=["min", "max"])
    bt.add_argument("--stage", default=None)
    bt.set_defaults(func=cmd_best)

    cp = sub.add_parser("compare", help="Two or more runs side by side")
    cp.add_argument("run_ids", nargs="+")
    cp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    store_path = Path(args.store)
    if not store_path.exists() and args.command != "list":
        raise SystemExit(f"error: no run store at {store_path}")
    raise SystemExit(args.func(RunStore(store_path), args))


if __name__ == "__main__":
    main()

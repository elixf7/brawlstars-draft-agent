#!/usr/bin/env python3
"""Run a training stage from a config file.

Each stage is one command. The config names the data, the hyperparameters and
the seed; the run directory afterwards holds the resolved config, so a run can
be repeated from its own output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from bsdraft.config import ConfigError, RunConfig, load_config
from bsdraft.data.sources import DatasetError, DatasetRef, resolve_dataset
from bsdraft.seeding import seed_everything


def _git_commit() -> str | None:
    """The code that produced a run, so results can be traced back to it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def resolve_source(cfg: RunConfig) -> DatasetRef | Path:
    """A local database when one is configured, otherwise a pinned dataset commit."""
    if cfg.data.db_path:
        return Path(cfg.data.db_path)
    return resolve_dataset(
        cfg.data.season, repo_id=cfg.data.repo_id, revision=cfg.data.revision
    )


def write_manifest(cfg: RunConfig, stage: str, source, extra: dict) -> Path:
    """Record what this run was, beside what it produced."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": cfg.name,
        "stage": stage,
        "started_utc": datetime.now(tz=UTC).isoformat(),
        "seed": cfg.seed,
        "git_commit": _git_commit(),
        "source": str(source),
        "config": cfg.to_dict(),
        **extra,
    }
    path = cfg.run_dir / f"{stage}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def cmd_fm(cfg: RunConfig) -> int:
    from bsdraft.fm.model import train_fm

    source = resolve_source(cfg)
    seed_everything(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.name}] fm  seed={cfg.seed}  source={source}")

    started = time.monotonic()
    train_fm(
        k=cfg.fm.k, lr=cfg.fm.lr, weight_decay=cfg.fm.weight_decay,
        batch_size=cfg.fm.batch_size, max_epochs=cfg.fm.max_epochs,
        patience=cfg.fm.patience,
        model_path=cfg.run_dir / "fm_model.pkl",
        schema_path=cfg.run_dir / "feature_schema.pkl",
        source=source, elo_min=cfg.data.elo_min, elo_max=cfg.data.elo_max,
    )
    m = write_manifest(cfg, "fm", source,
                       {"elapsed_seconds": round(time.monotonic() - started, 1)})
    print(f"wrote {m}")
    return 0


def cmd_selfplay(cfg: RunConfig) -> int:
    from bsdraft.data.matchup_db import MatchupDB
    from bsdraft.fm.model import FMInference
    from bsdraft.mcts.evaluator import FMEvaluator
    from bsdraft.selfplay.generate import load_map_mode_pairs
    from bsdraft.selfplay.train import IterationConfig, run_iteration_loop

    source = resolve_source(cfg)
    seed_everything(cfg.seed)
    model_path = cfg.run_dir / "fm_model.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"error: no evaluator at {model_path}. Run `bsdraft-train fm` first."
        )
    print(f"[{cfg.name}] selfplay  seed={cfg.seed}  source={source}")

    evaluator = FMEvaluator(FMInference.load(model_path))
    db = MatchupDB.build(source, elo_min=cfg.data.elo_min, elo_max=cfg.data.elo_max)
    iter_cfg = IterationConfig(
        n_games_per_iter=cfg.selfplay.n_games_per_iter,
        n_sims_later=cfg.selfplay.n_sims_later,
        n_eval_games=cfg.selfplay.n_eval_games,
        n_sims_eval=cfg.selfplay.n_sims_eval,
        promotion_threshold=cfg.selfplay.promotion_threshold,
        n_workers=cfg.selfplay.n_workers,
        seed=cfg.seed,
    )
    started = time.monotonic()
    results = run_iteration_loop(
        n_iterations=cfg.selfplay.n_iterations,
        data_dir=cfg.run_dir,
        map_mode_pairs=load_map_mode_pairs(source),
        config=iter_cfg,
        evaluator=evaluator,
        db=db,
        resume=cfg.selfplay.resume,
    )
    m = write_manifest(cfg, "selfplay", source, {
        "iterations": len(results),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    })
    print(f"wrote {m}")
    return 0


def cmd_show(cfg: RunConfig) -> int:
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fm", "selfplay", "show"])
    p.add_argument("-c", "--config", default=None, help="TOML config (defaults apply without one)")
    p.add_argument("--name", default=None, help="Override the run name")
    p.add_argument("--seed", type=int, default=None, help="Override the seed")
    p.add_argument("--output-dir", default=None, help="Override where runs are written")
    args = p.parse_args()

    try:
        cfg = load_config(args.config, name=args.name, seed=args.seed,
                          output_dir=args.output_dir)
    except ConfigError as e:
        raise SystemExit(f"error: {e}") from None

    try:
        raise SystemExit(
            {"fm": cmd_fm, "selfplay": cmd_selfplay, "show": cmd_show}[args.stage](cfg)
        )
    except (DatasetError, ConfigError) as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()

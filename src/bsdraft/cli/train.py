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
from bsdraft.tracking import RunStore


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


def store_for(cfg: RunConfig) -> RunStore:
    """One store per output directory, shared by every run in it."""
    base = cfg.run_dir.parent
    return RunStore(base / "registry.db")


def cmd_fm(cfg: RunConfig) -> int:
    source = resolve_source(cfg)
    seed_everything(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    store = store_for(cfg)
    run_id = store.start(
        name=cfg.name, stage="fm", seed=cfg.seed, git_commit=_git_commit(),
        dataset=str(source), config=cfg.to_dict(),
    )
    print(f"[{run_id}] fm({cfg.fm.model})  seed={cfg.seed}  source={source}")

    started = time.monotonic()
    try:
        if cfg.fm.model == "ffm":
            from bsdraft.data.prep import build_game_dataset
            from bsdraft.features.engineering import chronological_split
            from bsdraft.fm.train_ffm import train_ffm

            df, _ = build_game_dataset(source, elo_min=cfg.data.elo_min,
                                       elo_max=cfg.data.elo_max)
            train_df, val_df = chronological_split(df, val_fraction=0.20)
            inference = train_ffm(
                train_df, val_df,
                k=cfg.fm.k, lr=cfg.fm.lr, weight_decay=cfg.fm.weight_decay,
                batch_size=cfg.fm.batch_size, max_epochs=cfg.fm.max_epochs,
                patience=cfg.fm.patience,
                model_path=cfg.run_dir / "fm_model.pkl",
            )
        else:
            from bsdraft.fm.model import train_fm

            inference = train_fm(
                k=cfg.fm.k, lr=cfg.fm.lr, weight_decay=cfg.fm.weight_decay,
                batch_size=cfg.fm.batch_size, max_epochs=cfg.fm.max_epochs,
                patience=cfg.fm.patience,
                model_path=cfg.run_dir / "fm_model.pkl",
                schema_path=cfg.run_dir / "feature_schema.pkl",
                source=source, elo_min=cfg.data.elo_min, elo_max=cfg.data.elo_max,
            )
    except BaseException as e:
        store.finish(run_id, status="failed", error=f"{type(e).__name__}: {e}",
                     elapsed_seconds=round(time.monotonic() - started, 1))
        raise

    elapsed = round(time.monotonic() - started, 1)
    store.log_metrics(run_id, {
        "val_logloss": getattr(inference, "val_logloss", None),
        "val_auc": getattr(inference, "val_auc", None),
        "val_brier": getattr(inference, "val_brier", None),
        "n_train": getattr(inference, "n_train", None),
        "n_val": getattr(inference, "n_val", None),
    })
    # Per-epoch history, so a run's training curve survives the run.
    for epoch, entry in enumerate(getattr(inference, "train_history", []) or [], start=1):
        values = entry if isinstance(entry, dict) else {}
        if values:
            store.log_metrics(run_id, {
                k: v for k, v in values.items() if isinstance(v, int | float)
            }, step=epoch)
    for name in ("fm_model.pkl", "feature_schema.pkl"):
        store.log_artifact(run_id, name, cfg.run_dir / name)
    store.finish(run_id, elapsed_seconds=elapsed)

    m = write_manifest(cfg, "fm", source, {"run_id": run_id, "elapsed_seconds": elapsed})
    print(f"wrote {m}")
    print(f"logged run {run_id} to {store.path}")
    return 0


def cmd_selfplay(cfg: RunConfig) -> int:
    import pickle

    from bsdraft.data.matchup_db import MatchupDB
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

    # Either model can drive search; the evaluator wrapper differs.
    with open(model_path, "rb") as f:
        trained = pickle.load(f)
    from bsdraft.fm.ffm import FFMInference
    if isinstance(trained, FFMInference):
        from bsdraft.mcts.ffm_evaluator import FFMEvaluator
        evaluator = FFMEvaluator(trained)
        known_maps = set(trained.maps)
    else:
        from bsdraft.mcts.evaluator import FMEvaluator
        evaluator = FMEvaluator(trained)
        known_maps = set(trained.schema.maps)
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
    store = store_for(cfg)
    run_id = store.start(
        name=cfg.name, stage="selfplay", seed=cfg.seed, git_commit=_git_commit(),
        dataset=str(source), config=cfg.to_dict(),
    )
    started = time.monotonic()
    try:
        results = run_iteration_loop(
            n_iterations=cfg.selfplay.n_iterations,
            data_dir=cfg.run_dir,
            map_mode_pairs=load_map_mode_pairs(source, known_maps=known_maps),
            config=iter_cfg,
            evaluator=evaluator,
            db=db,
            resume=cfg.selfplay.resume,
        )
    except BaseException as e:
        store.finish(run_id, status="failed", error=f"{type(e).__name__}: {e}",
                     elapsed_seconds=round(time.monotonic() - started, 1))
        raise

    elapsed = round(time.monotonic() - started, 1)
    # One row per iteration, so promotion decisions can be read back later.
    for i, r in enumerate(results, start=1):
        values = {k: v for k, v in vars(r).items() if isinstance(v, int | float)}
        if values:
            store.log_metrics(run_id, values, step=i)
    final = {k: v for k, v in vars(results[-1]).items()
             if isinstance(v, int | float)} if results else {}
    store.log_metrics(run_id, {f"final_{k}": v for k, v in final.items()})
    store.log_metrics(run_id, {"iterations": len(results)})
    for artifact in sorted(cfg.run_dir.glob("*.pkl")):
        store.log_artifact(run_id, artifact.name, artifact)
    store.finish(run_id, elapsed_seconds=elapsed)

    m = write_manifest(cfg, "selfplay", source, {
        "run_id": run_id, "iterations": len(results), "elapsed_seconds": elapsed,
    })
    print(f"wrote {m}")
    print(f"logged run {run_id} to {store.path}")
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

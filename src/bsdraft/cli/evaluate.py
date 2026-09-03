#!/usr/bin/env python3
"""Score a trained model against baselines on held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bsdraft.cli.train import resolve_source, store_for
from bsdraft.config import ConfigError, load_config
from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import DatasetError
from bsdraft.eval import compare_predictors
from bsdraft.eval.baselines import default_baselines
from bsdraft.features.engineering import chronological_split
from bsdraft.seeding import seed_everything


def _model_probs(model_path: Path, val_df, schema_path: Path) -> np.ndarray | None:
    """Predict on the validation rows with the trained FM, if one exists."""
    if not model_path.exists():
        return None
    import pickle

    from bsdraft.features.engineering import build_feature_matrix
    from bsdraft.fm.evaluate import batch_predict
    from bsdraft.fm.model import _csr_to_compact

    with open(model_path, "rb") as f:
        fm = pickle.load(f)
    schema = getattr(fm, "schema", None)
    if schema is None:
        with open(schema_path, "rb") as f:
            schema = pickle.load(f)

    # build_feature_matrix emits each game twice — as played and with teams
    # flipped, for symmetry. Only the as-played rows line up with val_df.
    X, _ = build_feature_matrix(val_df, schema)
    idx, vals = _csr_to_compact(X)
    return batch_predict(fm, idx, vals)[::2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=None)
    p.add_argument("--name", default=None, help="Run whose model to evaluate")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--shrinkage", type=float, default=50.0,
                   help="Pseudo-counts pulling sparse rates toward the base rate")
    p.add_argument("--baselines-only", action="store_true",
                   help="Skip the model; useful before one is trained")
    p.add_argument("--json", default=None, help="Also write the comparison here")
    args = p.parse_args()

    try:
        cfg = load_config(args.config, name=args.name, output_dir=args.output_dir)
    except ConfigError as e:
        raise SystemExit(f"error: {e}") from None

    seed_everything(cfg.seed)
    try:
        source = resolve_source(cfg)
        df, _ = build_game_dataset(source, elo_min=cfg.data.elo_min,
                                   elo_max=cfg.data.elo_max)
    except DatasetError as e:
        raise SystemExit(f"error: {e}") from None

    # Chronological, so the model is judged on matches that happened after
    # everything it learned from. A random split would leak the meta forward.
    train_df, val_df = chronological_split(df, val_fraction=args.val_fraction)

    model_probs = None
    if not args.baselines_only:
        model_probs = _model_probs(
            cfg.run_dir / "fm_model.pkl", val_df, cfg.run_dir / "feature_schema.pkl"
        )
        if model_probs is None:
            print(f"note: no model at {cfg.run_dir / 'fm_model.pkl'}; baselines only\n")

    comparison = compare_predictors(
        train_df, val_df,
        baselines=default_baselines(args.shrinkage),
        model_probs=model_probs,
    )
    print(comparison.render())

    payload = {
        "run": cfg.name,
        "dataset": str(source),
        "val_fraction": args.val_fraction,
        "shrinkage": args.shrinkage,
        "n_train": comparison.n_train,
        "n_val": comparison.n_val,
        "results": comparison.rows,
    }
    out = Path(args.json) if args.json else cfg.run_dir / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    # Record against the run, so a model's standing versus baselines is part of
    # its history rather than something re-derived later.
    store = store_for(cfg)
    run_id = store.start(name=cfg.name, stage="eval", seed=cfg.seed,
                         dataset=str(source), config=cfg.to_dict())
    for row in comparison.rows:
        key = row["name"].replace(" ", "_")
        store.log_metrics(run_id, {f"{key}.{m}": v
                                   for m, v in row.items() if m != "name"})
    store.finish(run_id)
    print(f"logged run {run_id}")


if __name__ == "__main__":
    main()

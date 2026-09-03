#!/usr/bin/env python3
"""Build the static dashboard: season analytics plus a playable mock draft."""
from __future__ import annotations

import argparse
import pickle
from datetime import UTC, datetime

from bsdraft.cli.train import resolve_source
from bsdraft.config import ConfigError, load_config
from bsdraft.dashboard.export import build_payload
from bsdraft.dashboard.page import write_dashboard
from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import DatasetError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=None)
    p.add_argument("--name", default=None, help="Run whose model to publish")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--out", default="site/index.html")
    args = p.parse_args()

    try:
        cfg = load_config(args.config, name=args.name, output_dir=args.output_dir)
        source = resolve_source(cfg)
        model_path = cfg.run_dir / "fm_model.pkl"
        if not model_path.exists():
            raise SystemExit(f"error: no model at {model_path}. Train one first.")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        df, _ = build_game_dataset(source, elo_min=cfg.data.elo_min,
                                   elo_max=cfg.data.elo_max)
    except (ConfigError, DatasetError) as e:
        raise SystemExit(f"error: {e}") from None

    payload = build_payload(
        model, df, season=cfg.data.season, dataset=str(source),
        generated_utc=datetime.now(tz=UTC).isoformat(),
    )
    out = write_dashboard(payload, args.out)
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

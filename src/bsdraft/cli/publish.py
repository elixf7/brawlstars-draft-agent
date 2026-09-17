#!/usr/bin/env python3
"""Publish the trained model and its season statistics as one JSON payload.

This used to render an interactive dashboard. Brawl Stars Atlas now does that
job properly — https://brawlstars-atlas.pages.dev/ — so what ships from here is
the data behind it and nothing else.

The payload is served as `data.json` beside a landing page that points visitors
at the atlas. Atlas reads that file directly; it used to scrape the payload back
out of a `window.__DATA__` assignment in the dashboard's HTML, which worked but
made a rendering detail into an API.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import UTC, datetime
from pathlib import Path

from bsdraft.cli.train import resolve_source
from bsdraft.config import ConfigError, load_config
from bsdraft.data.prep import build_game_dataset
from bsdraft.data.sources import DatasetError
from bsdraft.payload.export import build_payload, write_payload

ATLAS_URL = "https://brawlstars-atlas.pages.dev/"

LANDING = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brawl Stars Draft Agent</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    margin: 0; min-height: 100vh; display: grid; place-items: center;
    font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 24px;
  }}
  main {{ max-width: 34rem; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .75rem; }}
  p {{ margin: 0 0 1rem; }}
  .go {{
    display: inline-block; padding: .7rem 1.2rem; border-radius: .5rem;
    background: #4f46e5; color: #fff; text-decoration: none; font-weight: 600;
  }}
  .meta {{ font-size: .85rem; opacity: .65; margin-top: 2rem; }}
  code {{ font-size: .85em; }}
</style>
<main>
  <h1>Brawl Stars Draft Agent</h1>
  <p>
    The interactive dashboard that used to live here has moved to
    <strong>Brawl Stars Atlas</strong>, which runs this model in your browser
    alongside the ranked statistics it was trained on.
  </p>
  <p><a class="go" href="{atlas}">Open Brawl Stars Atlas &rarr;</a></p>
  <p>
    This page still serves the model itself, as
    <a href="data.json"><code>data.json</code></a> &mdash; the weights, the
    character embedding, and the season's counted matches. The code that
    produces it is on
    <a href="https://github.com/elixf7/brawlstars-draft-agent">GitHub</a>.
  </p>
  <p class="meta">
    {season} &middot; {games:,} games &middot; generated {generated}
  </p>
</main>
"""


def write_landing(payload: dict, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    season = payload.get("season", {})
    p.write_text(
        LANDING.format(
            atlas=ATLAS_URL,
            season=season.get("season", "unknown season"),
            games=int(season.get("games", 0)),
            generated=payload.get("generated_utc", "")[:16].replace("T", " ") + " UTC",
        ),
        encoding="utf-8",
    )
    return p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default=None)
    p.add_argument("--name", default=None, help="Run whose model to publish")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--out-dir", default="site",
                   help="Directory to write data.json and index.html into")
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
        registry_path=cfg.run_dir.parent / "registry.db",
    )
    out_dir = Path(args.out_dir)
    data = write_payload(payload, out_dir / "data.json")
    landing = write_landing(payload, out_dir / "index.html")
    print(f"{data}  ({data.stat().st_size / 1024:.0f} KB)")
    print(f"{landing}  ({landing.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

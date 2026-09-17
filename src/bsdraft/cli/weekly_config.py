"""Select a season worth training and pin one dataset revision for the whole run.

Newest is the usual answer, but not always the right one. A season's first crawl
lands hours after the reset, when every battle log still holds mostly pre-reset
matches, so the newest partition can be a few thousand rows — season 54 opened
with 4,082. Training that fails the baseline gate, which leaves the payload
unpublished, which strands the atlas on whatever it last deployed and turns the
weekly run red until the season fills out.

So the newest season has to clear a floor before it is chosen. Below it, the
season before is trained again: the same data produces the same weights, the
payload republishes, and the chain keeps moving until the new season is worth
switching to.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from collections.abc import Callable

#: Sets a season needs before it is worth training. The gate downstream is the
#: real authority — 100,000 training and 10,000 held-out *games* after filtering
#: on elo and on skill_ns being trustworthy. This is the cheap proxy for it,
#: measured in sets, and deliberately clear of the gate: choosing a season that
#: then fails the gate costs a whole week, while waiting one extra run costs a
#: few days of freshness on a site that is already showing last season.
DEFAULT_MIN_ROWS = 150_000

_SEASON_PARQUET = re.compile(r"data/season=season(\d+)/data\.parquet")


def published_seasons(files: list[str]) -> list[str]:
    """Published seasons, newest first."""
    numbers = sorted(
        (int(match.group(1)) for name in files if (match := _SEASON_PARQUET.fullmatch(name))),
        reverse=True,
    )
    return [f"season{n}" for n in numbers]


def choose_season(
    files: list[str],
    count_rows: Callable[[str], int] | None = None,
    min_rows: int = 0,
) -> str:
    """The newest published season holding enough data to train on."""
    seasons = published_seasons(files)
    if not seasons:
        raise ValueError("No published season parquet found; nothing to train.")
    if count_rows is None or min_rows <= 0:
        return seasons[0]

    for season in seasons:
        rows = count_rows(season)
        if rows >= min_rows:
            print(f"{season}: {rows:,} sets; training it.", file=sys.stderr)
            return season
        print(
            f"{season}: {rows:,} sets, below the {min_rows:,} floor "
            f"— too thin to beat the baselines. Falling back.",
            file=sys.stderr,
        )
    raise ValueError(
        f"No published season reaches {min_rows:,} sets; nothing worth training."
    )


def render_config(
    template: str,
    files: list[str],
    revision: str,
    repo: str,
    count_rows: Callable[[str], int] | None = None,
    min_rows: int = 0,
) -> str:
    season = choose_season(files, count_rows, min_rows)
    config = tomllib.loads(template)
    config["name"] = season
    config["data"].update(season=season, revision=revision, repo_id=repo)
    # The template contains only scalar top-level values and scalar sections.
    lines = [f"{key} = {json.dumps(value)}" for key, value in config.items()
             if not isinstance(value, dict)]
    for section, values in config.items():
        if isinstance(values, dict):
            lines.extend(["", f"[{section}]"])
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def _row_counter(repo: str, revision: str) -> Callable[[str], int]:
    """Count a season's sets from the parquet footer, without fetching the data.

    A season is tens of megabytes and only the row count is wanted, so this
    range-reads the footer over HTTP — under a second against a file the
    workflow would otherwise spend a minute downloading to reject.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def count(season: str) -> int:
        path = f"datasets/{repo}/data/season={season}/data.parquet"
        with fs.open(path, "rb", revision=revision) as handle:
            return pq.ParquetFile(handle).metadata.num_rows

    return count


def main() -> None:
    import argparse

    from huggingface_hub import HfApi

    from bsdraft.config import REPO_ROOT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="Report which season would be trained, as GitHub Actions outputs, "
             "and write nothing",
    )
    args = ap.parse_args()

    repo = "EliF77/brawlstars-ranked"
    # A workflow_dispatch input left blank arrives as an empty string rather
    # than as an unset variable, so `os.environ.get`'s default is not enough.
    min_rows = int(os.environ.get("BSDRAFT_MIN_SEASON_ROWS") or DEFAULT_MIN_ROWS)
    api = HfApi()
    revision = api.dataset_info(repo).sha
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)

    if args.check:
        # "Ready" means the newest season is worth training in its own right,
        # which is exactly what an extra, off-schedule run exists to find out.
        # Falling back to the season before is the right answer for the weekly
        # run, and a reason for a daily one not to bother.
        try:
            newest = published_seasons(files)[0]
            season = choose_season(files, _row_counter(repo, revision), min_rows)
        except (ValueError, IndexError) as exc:
            print(exc, file=sys.stderr)
            print("ready=false")
            return
        print(f"season={season}")
        print(f"newest={newest}")
        print(f"ready={str(season == newest).lower()}")
        return

    template = (REPO_ROOT / "configs/season53.toml").read_text()
    rendered = render_config(
        template, files, revision, repo, _row_counter(repo, revision), min_rows
    )
    output = REPO_ROOT / "runs/weekly.toml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(f"Weekly training: {tomllib.loads(rendered)['data']}")


if __name__ == "__main__":
    main()

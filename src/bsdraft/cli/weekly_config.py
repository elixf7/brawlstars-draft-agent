"""Select the newest published season and pin one dataset revision for the whole run."""

from __future__ import annotations

import json
import re
import tomllib


def render_config(template: str, files: list[str], revision: str, repo: str) -> str:
    seasons = [
        int(match.group(1))
        for name in files
        if (match := re.fullmatch(r"data/season=season(\d+)/data\.parquet", name))
    ]
    if not seasons:
        raise ValueError("No published season parquet found; nothing to train.")
    season = f"season{max(seasons)}"
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


def main() -> None:
    from huggingface_hub import HfApi

    from bsdraft.config import REPO_ROOT

    repo = "EliF77/brawlstars-ranked"
    api = HfApi()
    revision = api.dataset_info(repo).sha
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    template = (REPO_ROOT / "configs/season53.toml").read_text()
    rendered = render_config(template, files, revision, repo)
    output = REPO_ROOT / "runs/weekly.toml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(f"Weekly training: {tomllib.loads(rendered)['data']}")


if __name__ == "__main__":
    main()

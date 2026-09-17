import tomllib

import pytest

from bsdraft.cli.weekly_config import render_config
from bsdraft.config import REPO_ROOT, load_config


def test_latest_season_and_one_revision_for_all_stages(tmp_path):
    template = (REPO_ROOT / "configs/season53.toml").read_text()
    rendered = render_config(template, [
        "data/season=season99/data.parquet", "data/season=season100/data.parquet",
        "state/season101.db",
    ], "abc123", "owner/data")
    path = tmp_path / "weekly.toml"
    path.write_text(rendered)
    cfg = load_config(path)
    assert cfg.name == cfg.data.season == "season100"
    assert cfg.data.revision == "abc123"
    assert cfg.data.repo_id == "owner/data"
    assert tomllib.loads(rendered)["fm"] == tomllib.loads(template)["fm"]
    assert cfg.selfplay.resume is True


def test_missing_published_season_fails():
    with pytest.raises(ValueError, match="No published season"):
        render_config("", ["state/season54.db"], "abc123", "owner/data")

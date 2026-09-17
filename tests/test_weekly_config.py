import tomllib

import pytest

from bsdraft.cli.weekly_config import choose_season, render_config
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


def _files(*seasons):
    return [f"data/season=season{n}/data.parquet" for n in seasons] + ["state/season99.db"]


def test_thin_newest_season_falls_back_to_the_one_before():
    """A season's first crawl is a few thousand sets; training it fails the gate."""
    rows = {"season54": 4_082, "season53": 1_369_934}
    assert choose_season(_files(53, 54), rows.__getitem__, min_rows=150_000) == "season53"


def test_newest_season_is_chosen_once_it_clears_the_floor():
    rows = {"season54": 620_000, "season53": 1_369_934}
    assert choose_season(_files(53, 54), rows.__getitem__, min_rows=150_000) == "season54"


def test_no_floor_means_newest_wins_without_counting():
    def explode(_season):
        raise AssertionError("must not count rows when no floor is set")

    assert choose_season(_files(53, 54), explode, min_rows=0) == "season54"


def test_every_season_too_thin_is_an_error():
    with pytest.raises(ValueError, match="nothing worth training"):
        choose_season(_files(53, 54), lambda _s: 100, min_rows=150_000)


def test_fallback_reaches_the_config_that_gets_written(tmp_path):
    template = (REPO_ROOT / "configs/season53.toml").read_text()
    rendered = render_config(
        template, _files(53, 54), "abc123", "owner/data",
        {"season54": 4_082, "season53": 1_369_934}.__getitem__, 150_000,
    )
    path = tmp_path / "weekly.toml"
    path.write_text(rendered)
    cfg = load_config(path)
    assert cfg.name == cfg.data.season == "season53"
    assert cfg.data.revision == "abc123"


def test_blank_dispatch_input_falls_back_to_the_default_floor(monkeypatch):
    """An unset workflow_dispatch input arrives as "", not as a missing variable."""
    import bsdraft.cli.weekly_config as wc

    monkeypatch.setenv("BSDRAFT_MIN_SEASON_ROWS", "")
    import os
    assert int(os.environ.get("BSDRAFT_MIN_SEASON_ROWS") or wc.DEFAULT_MIN_ROWS) == 150_000
    monkeypatch.setenv("BSDRAFT_MIN_SEASON_ROWS", "5000")
    assert int(os.environ.get("BSDRAFT_MIN_SEASON_ROWS") or wc.DEFAULT_MIN_ROWS) == 5_000

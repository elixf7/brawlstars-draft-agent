"""The training CLI turns a config into a run that can be traced afterwards."""
import json
from pathlib import Path

from bsdraft.cli.train import resolve_source, write_manifest
from bsdraft.config import load_config
from bsdraft.data.sources import DatasetRef


def test_a_configured_local_database_wins_over_the_hub(tmp_path):
    """So a run can be reproduced offline, or against data not yet published."""
    cfg = load_config(None, output_dir=str(tmp_path))
    cfg = type(cfg)(**{**cfg.to_dict(), "data": type(cfg.data)(db_path="season49/v1.db")})
    assert resolve_source(cfg) == Path("season49/v1.db")


def test_the_hub_is_used_when_no_database_is_configured(tmp_path, monkeypatch):
    seen = {}

    def fake_resolve(season, **kw):
        seen.update(season=season, **kw)
        return DatasetRef(kw["repo_id"], season, "deadbeef")

    monkeypatch.setattr("bsdraft.cli.train.resolve_dataset", fake_resolve)
    cfg = load_config(None, output_dir=str(tmp_path))
    ref = resolve_source(cfg)
    assert isinstance(ref, DatasetRef)
    assert seen["season"] == cfg.data.season


def test_manifest_records_what_the_run_can_be_traced_by(tmp_path):
    cfg = load_config(None, name="exp", output_dir=str(tmp_path))
    ref = DatasetRef("me/ds", "season53", "4c45efee11")
    path = write_manifest(cfg, "fm", ref, {"elapsed_seconds": 3.2})

    m = json.loads(path.read_text())
    assert m["run"] == "exp" and m["stage"] == "fm"
    assert m["seed"] == cfg.seed
    assert m["source"] == str(ref)          # the exact data commit
    assert m["config"]["fm"]["k"] == cfg.fm.k
    assert "started_utc" in m and "git_commit" in m


def test_manifest_lands_in_the_run_directory(tmp_path):
    cfg = load_config(None, name="exp", output_dir=str(tmp_path))
    path = write_manifest(cfg, "selfplay", "local.db", {})
    assert path == Path(tmp_path) / "exp" / "selfplay_manifest.json"

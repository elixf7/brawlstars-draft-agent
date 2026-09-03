"""A run is described by its config. If the config can be wrong quietly, the
run cannot be explained afterwards."""
from pathlib import Path

import pytest

from bsdraft.config import ConfigError, RunConfig, load_config


def write(tmp_path, text):
    p = tmp_path / "run.toml"
    p.write_text(text)
    return p


def test_defaults_apply_without_a_file():
    cfg = load_config(None)
    assert cfg.name == "default"
    assert cfg.data.season.startswith("season")
    assert cfg.fm.k > 0 and cfg.selfplay.n_iterations > 0


def test_file_values_override_defaults(tmp_path):
    cfg = load_config(write(tmp_path, """
name = "wide-k"
seed = 11
[fm]
k = 64
[data]
season = "season52"
elo_min = 12.5
"""))
    assert (cfg.name, cfg.seed, cfg.fm.k) == ("wide-k", 11, 64)
    assert cfg.data.season == "season52"
    assert cfg.data.elo_min == 12.5
    assert cfg.fm.lr == RunConfig().fm.lr      # untouched keys keep defaults


def test_a_mistyped_key_is_rejected_not_ignored(tmp_path):
    """Silently keeping the default for a misspelled hyperparameter produces a
    run whose config does not describe it."""
    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(write(tmp_path, "[fm]\nlearning_rate = 0.01\n"))


def test_unknown_section_key_names_the_valid_ones(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path, "[data]\nseasons = 'season53'\n"))
    assert "season" in str(e.value)


def test_command_line_overrides_beat_the_file(tmp_path):
    p = write(tmp_path, 'name = "from-file"\nseed = 1\n')
    cfg = load_config(p, name="from-cli", seed=99)
    assert (cfg.name, cfg.seed) == ("from-cli", 99)


def test_omitted_overrides_do_not_clobber(tmp_path):
    p = write(tmp_path, 'name = "from-file"\nseed = 1\n')
    cfg = load_config(p, name=None, seed=None)
    assert (cfg.name, cfg.seed) == ("from-file", 1)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="No such config"):
        load_config(tmp_path / "absent.toml")


def test_negative_seed_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="seed"):
        load_config(write(tmp_path, "seed = -1\n"))


def test_run_dir_is_named_after_the_run(tmp_path):
    cfg = load_config(None, name="exp-7", output_dir=str(tmp_path))
    assert cfg.run_dir == Path(tmp_path) / "exp-7"


def test_config_round_trips_to_a_dict():
    """The manifest records this verbatim."""
    d = load_config(None).to_dict()
    assert d["fm"]["k"] == RunConfig().fm.k
    assert set(d) >= {"name", "seed", "data", "fm", "selfplay"}

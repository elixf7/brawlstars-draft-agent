"""Structural guards.

The restructure moved every module two directories deeper, which silently
invalidated the `REPO_ROOT` anchors that locate model artifacts — the paths
still resolved, just to the wrong place. These tests make that class of
breakage loud.
"""
import importlib

import pytest

MODULES = [
    "bsdraft",
    "bsdraft.data.prep",
    "bsdraft.data.matchup_db",
    "bsdraft.features.engineering",
    "bsdraft.fm.model",
    "bsdraft.fm.evaluate",
    "bsdraft.fm.interpret",
    "bsdraft.mcts.state",
    "bsdraft.mcts.node",
    "bsdraft.mcts.evaluator",
    "bsdraft.mcts.rollout",
    "bsdraft.mcts.confidence",
    "bsdraft.mcts.recommend",
    "bsdraft.selfplay.generate",
    "bsdraft.selfplay.policy_net",
    "bsdraft.selfplay.train",
    "bsdraft.selfplay.joint_net",
    "bsdraft.pipeline.training",
    "bsdraft.pipeline.workflow",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    """No module may depend on the caller's working directory to be importable."""
    importlib.import_module(name)


@pytest.mark.parametrize(
    "name",
    ["bsdraft.fm.model", "bsdraft.fm.evaluate", "bsdraft.selfplay.joint_net",
     "bsdraft.features.engineering", "bsdraft.selfplay.policy_net",
     "bsdraft.data.prep", "bsdraft.data.matchup_db"],
)
def test_repo_root_points_at_the_repository(name):
    """A wrong REPO_ROOT does not raise — it silently reads and writes model
    artifacts in the wrong place."""
    root = importlib.import_module(name).REPO_ROOT
    assert (root / "pyproject.toml").is_file(), f"{name}.REPO_ROOT resolved to {root}"
    assert (root / "src" / "bsdraft").is_dir()


def test_no_module_manipulates_sys_path():
    """sys.path juggling is what the package layout replaces."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "bsdraft"
    offenders = [p.name for p in src.rglob("*.py") if "sys.path" in p.read_text()]
    assert not offenders, f"sys.path manipulation remains in: {offenders}"

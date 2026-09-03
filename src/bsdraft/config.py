"""Run configuration.

A training run is described by a file, not by remembered arguments. The config
is the unit of reproducibility: it names the data, the hyperparameters, and the
seed, and it is recorded alongside whatever the run produces.

TOML rather than YAML because Python reads it without a dependency, and because
it will not silently interpret `no` as False.
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """The configuration file is missing something, or says something impossible."""


@dataclass(frozen=True)
class DataConfig:
    """Which rows a run trains on."""

    season: str = "season53"
    repo_id: str = "EliF77/brawlstars-ranked"
    #: None resolves the current commit and records it. A value pins to it.
    revision: str | None = None
    #: A local season database, used instead of the Hub when set.
    db_path: str | None = None
    elo_min: float = 10.0
    elo_max: float = 23.0


@dataclass(frozen=True)
class FMConfig:
    """Factorization machine hyperparameters."""

    k: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 50
    patience: int = 5


@dataclass(frozen=True)
class SelfPlayConfig:
    """Iterative self-play settings."""

    n_iterations: int = 5
    n_games_per_iter: int = 500
    n_sims_later: int = 5_000
    n_eval_games: int = 200
    n_sims_eval: int = 2_000
    promotion_threshold: float = 0.525
    n_workers: int = 1
    resume: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Everything one run needs to be repeatable."""

    name: str = "default"
    #: Seeds Python, NumPy and Torch. Without it a run cannot be repeated.
    seed: int = 20260903
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        base = Path(self.output_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / self.name


def _build(cls, raw: dict[str, Any], where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"[{where}] does not accept {sorted(unknown)}. Valid keys: {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path | None = None, **overrides: Any) -> RunConfig:
    """Read a run configuration, with optional top-level overrides.

    Unknown keys are rejected rather than ignored: a typo in a hyperparameter
    that silently keeps the default is a run you cannot explain later.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"No such config file: {p}")
        raw = tomllib.loads(p.read_text())

    sections = {
        "data": _build(DataConfig, raw.pop("data", {}), "data"),
        "fm": _build(FMConfig, raw.pop("fm", {}), "fm"),
        "selfplay": _build(SelfPlayConfig, raw.pop("selfplay", {}), "selfplay"),
    }
    raw.update({k: v for k, v in overrides.items() if v is not None})
    top = _build(RunConfig, {**raw, **sections}, "run")
    if top.seed < 0:
        raise ConfigError("seed must be non-negative")
    return top

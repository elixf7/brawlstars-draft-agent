"""Where training data comes from.

Two sources, one contract. A local SQLite file is what the seasons collected so
far live in; the published Hugging Face dataset is what everything should read
from now — it is versioned, fetchable by anyone, and pinnable to an exact commit.

That pinning is the point. A model trained against "the dataset" is not
reproducible, because the dataset grows every time the ETL pipeline runs. A
model trained against a named revision is.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TEAM1_BRAWLER_COLS = ["t1_b0_name", "t1_b1_name", "t1_b2_name"]
TEAM2_BRAWLER_COLS = ["t2_b0_name", "t2_b1_name", "t2_b2_name"]

ALL_BRAWLER_COLS = TEAM1_BRAWLER_COLS + TEAM2_BRAWLER_COLS

#: The columns modelling needs. Everything else is dropped at read time.
KEEP_COLS = [
    "id", "battle_time", "mode", "map", "record", "avg_elo", "skill_ns",
    *TEAM1_BRAWLER_COLS, *TEAM2_BRAWLER_COLS,
]

DEFAULT_REPO_ID = "EliF77/brawlstars-ranked"


class DatasetError(RuntimeError):
    """The requested data could not be located or read."""


@dataclass(frozen=True)
class DatasetRef:
    """A specific season at a specific commit of a specific dataset.

    Carried into run metadata so a trained model can name the data it saw.
    """

    repo_id: str
    season: str
    revision: str

    def __str__(self) -> str:
        return f"{self.repo_id}@{self.revision[:8]}:{self.season}"


def _require_hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:  # pragma: no cover - import guard
        raise DatasetError(
            "huggingface-hub is not installed. Install it with: uv add huggingface-hub"
        ) from e
    return HfApi, snapshot_download


def resolve_dataset(
    season: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> DatasetRef:
    """Pin a season to an exact dataset commit.

    Passing `revision=None` resolves whatever the default branch points at right
    now and records that commit, so the run is reproducible afterwards even
    though it was not pinned beforehand.
    """
    HfApi, _ = _require_hub()
    try:
        info = HfApi(token=token).dataset_info(repo_id, revision=revision)
    except Exception as e:
        raise DatasetError(f"Could not reach dataset {repo_id!r}: {e}") from e
    return DatasetRef(repo_id=repo_id, season=season, revision=info.sha)


def available_seasons(
    *, repo_id: str = DEFAULT_REPO_ID, revision: str | None = None,
    token: str | None = None,
) -> list[str]:
    """Season labels present in the dataset, read from its partition layout."""
    HfApi, _ = _require_hub()
    try:
        files = HfApi(token=token).list_repo_files(
            repo_id, repo_type="dataset", revision=revision
        )
    except Exception as e:
        raise DatasetError(f"Could not list {repo_id!r}: {e}") from e
    seasons = {
        part.split("=", 1)[1]
        for f in files
        for part in f.split("/")
        if part.startswith("season=")
    }
    return sorted(seasons)


def load_matches_from_hub(
    ref: DatasetRef,
    *,
    elo_min: float | None = None,
    elo_max: float | None = None,
    require_skill: bool = True,
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch one season from the published dataset and apply quality filters.

    Only that season's partition is downloaded, and filters are pushed into the
    Parquet scan, so a season is read without materialising the whole dataset.
    """
    _, snapshot_download = _require_hub()
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as pads
    except ImportError as e:  # pragma: no cover - import guard
        raise DatasetError("pyarrow is not installed. Install it with: uv add pyarrow") from e

    pattern = f"data/season={ref.season}/**"
    try:
        local = snapshot_download(
            repo_id=ref.repo_id, repo_type="dataset", revision=ref.revision,
            allow_patterns=[pattern], token=token,
        )
    except Exception as e:
        raise DatasetError(f"Could not download {ref}: {e}") from e

    season_dir = Path(local) / "data" / f"season={ref.season}"
    if not season_dir.is_dir():
        raise DatasetError(
            f"{ref.season!r} is not in {ref.repo_id} at {ref.revision[:8]}. "
            f"Available: {', '.join(available_seasons(repo_id=ref.repo_id)) or 'none'}"
        )

    dataset = pads.dataset(str(season_dir), format="parquet", partitioning="hive")
    present = set(dataset.schema.names)

    conditions = []
    if require_skill and "skill_ns_ok" in present:
        conditions.append(pc.field("skill_ns_ok") == 1)
    if elo_min is not None:
        conditions.append(pc.field("avg_elo") >= elo_min)
    if elo_max is not None:
        conditions.append(pc.field("avg_elo") <= elo_max)
    expr = conditions[0] if conditions else None
    for c in conditions[1:]:
        expr = expr & c

    columns = [c for c in KEEP_COLS if c in present]
    missing = set(KEEP_COLS) - present
    if missing:
        raise DatasetError(
            f"{ref} is missing columns needed for modelling: {sorted(missing)}"
        )
    return dataset.to_table(columns=columns, filter=expr).to_pandas()


def load_matches_from_sqlite(
    db_path: str | Path,
    *,
    elo_min: float | None = None,
    elo_max: float | None = None,
    require_skill: bool = True,
) -> pd.DataFrame:
    """Read a season from a local working database."""
    path = Path(db_path)
    if not path.exists():
        raise DatasetError(f"No such database: {path}")

    where = []
    if require_skill:
        where.append("skill_ns_ok = 1")
    if elo_min is not None:
        where.append(f"avg_elo >= {float(elo_min)}")
    if elo_max is not None:
        where.append(f"avg_elo <= {float(elo_max)}")
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(
            f"SELECT {', '.join(KEEP_COLS)} FROM matches {clause}", conn
        )


def load_matches(
    source: DatasetRef | str | Path,
    *,
    elo_min: float | None = None,
    elo_max: float | None = None,
    require_skill: bool = True,
    token: str | None = None,
) -> pd.DataFrame:
    """Load a season from either source, applying the same filters to both."""
    if isinstance(source, DatasetRef):
        return load_matches_from_hub(
            source, elo_min=elo_min, elo_max=elo_max,
            require_skill=require_skill, token=token,
        )
    return load_matches_from_sqlite(
        source, elo_min=elo_min, elo_max=elo_max, require_skill=require_skill
    )

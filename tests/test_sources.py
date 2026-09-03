"""Both sources must produce the same thing, and a run must be able to name
the exact data it saw."""
import sqlite3

import pytest

from bsdraft.data.sources import (
    KEEP_COLS,
    DatasetError,
    DatasetRef,
    load_matches,
    load_matches_from_sqlite,
)


def make_db(path, rows):
    cols = ", ".join(f"{c} TEXT" if "name" in c or c in
                     ("battle_time", "mode", "map", "record") else f"{c} REAL"
                     for c in KEEP_COLS)
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE matches ({cols}, skill_ns_ok INTEGER)")
    conn.executemany(
        f"INSERT INTO matches VALUES ({', '.join('?' * (len(KEEP_COLS) + 1))})", rows
    )
    conn.commit()
    conn.close()
    return str(path)


def a_row(avg_elo=15.0, skill_ok=1, tag="A"):
    return (
        1.0, "20260901T000000.000Z", "brawlBall", "Hot Potato", "T1-T1",
        avg_elo, 0.5, f"{tag}1", f"{tag}2", f"{tag}3", f"{tag}4", f"{tag}5", f"{tag}6",
        skill_ok,
    )


# ------------------------------------------------------------------ the ref
def test_ref_names_the_exact_data_a_run_saw():
    """Recorded in run metadata, so a model can say what it trained on."""
    ref = DatasetRef("me/ds", "season53", "4c45efee11223344")
    assert str(ref) == "me/ds@4c45efee:season53"


# ---------------------------------------------------------------- filtering
def test_sqlite_applies_elo_bounds(tmp_path):
    db = make_db(tmp_path / "s.db", [a_row(avg_elo=9.0), a_row(avg_elo=15.0), a_row(avg_elo=30.0)])
    df = load_matches_from_sqlite(db, elo_min=10, elo_max=23)
    assert len(df) == 1
    assert df["avg_elo"].iloc[0] == 15.0


def test_sqlite_drops_rows_without_a_trustworthy_skill_score(tmp_path):
    """skill_ns from a thin time bin is a confident-looking guess; the ETL
    flags those and modelling must not silently use them."""
    db = make_db(tmp_path / "s.db", [a_row(skill_ok=1), a_row(skill_ok=0)])
    assert len(load_matches_from_sqlite(db)) == 1
    assert len(load_matches_from_sqlite(db, require_skill=False)) == 2


def test_sqlite_returns_exactly_the_modelling_columns(tmp_path):
    db = make_db(tmp_path / "s.db", [a_row()])
    assert list(load_matches_from_sqlite(db).columns) == KEEP_COLS


def test_missing_database_is_a_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="No such database"):
        load_matches_from_sqlite(tmp_path / "absent.db")


# ---------------------------------------------------------------- dispatch
def test_load_matches_dispatches_on_the_source_type(tmp_path, monkeypatch):
    db = make_db(tmp_path / "s.db", [a_row()])
    assert len(load_matches(db)) == 1          # path -> sqlite

    called = {}

    def fake_hub(ref, **kw):
        called["ref"] = ref
        called["kw"] = kw
        return "from-hub"

    monkeypatch.setattr("bsdraft.data.sources.load_matches_from_hub", fake_hub)
    ref = DatasetRef("me/ds", "season53", "abc123")
    assert load_matches(ref, elo_min=10, elo_max=23) == "from-hub"
    assert called["ref"] is ref
    assert called["kw"]["elo_min"] == 10


def test_both_sources_take_the_same_filters(tmp_path):
    """Whichever source a run uses, the filtering must be identical — otherwise
    a model trained locally and one trained from the Hub see different data."""
    import inspect

    from bsdraft.data.sources import load_matches_from_hub
    hub = set(inspect.signature(load_matches_from_hub).parameters)
    lite = set(inspect.signature(load_matches_from_sqlite).parameters)
    assert {"elo_min", "elo_max", "require_skill"} <= hub & lite


# ------------------------------------------------------------------- errors
def test_absent_hub_dependency_says_how_to_fix_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_hub(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    from bsdraft.data import sources
    with pytest.raises(DatasetError, match="uv add huggingface-hub"):
        sources._require_hub()


def test_season_labels_are_read_from_the_partition_layout(monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            return [
                "README.md",
                "data/season=season52/battle_date=2026-07-20/data.parquet",
                "data/season=season53/battle_date=2026-08-21/data.parquet",
                "data/season=season53/battle_date=2026-08-22/data.parquet",
                "state/season53.db",
            ]

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    from bsdraft.data.sources import available_seasons
    assert available_seasons(repo_id="me/ds") == ["season52", "season53"]

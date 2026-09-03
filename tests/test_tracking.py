"""A run that cannot be compared to another run teaches nothing.

These cover the recording contract: a run is visible while it is happening,
survives failing, and carries enough to be compared afterwards."""
import pytest

from bsdraft.tracking import RunStore, sha256_of


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "registry.db")


def a_run(store, name="exp", stage="fm", **kw):
    return store.start(name=name, stage=stage, seed=7, git_commit="abc123",
                       dataset="me/ds@deadbeef:season53",
                       config={"fm": {"k": 32}}, **kw)


# ------------------------------------------------------------------ lifecycle
def test_a_run_is_visible_before_it_finishes(store):
    """A process that dies must leave evidence, not nothing."""
    run_id = a_run(store)
    run = store.get_run(run_id)
    assert run["status"] == "running"
    assert run["finished_utc"] is None


def test_finishing_records_the_outcome(store):
    run_id = a_run(store)
    store.finish(run_id, elapsed_seconds=12.5)
    run = store.get_run(run_id)
    assert run["status"] == "ok"
    assert run["elapsed_seconds"] == 12.5
    assert run["finished_utc"] is not None


def test_a_failed_run_keeps_its_error(store):
    run_id = a_run(store)
    store.finish(run_id, status="failed", error="ValueError: bad shape")
    run = store.get_run(run_id)
    assert run["status"] == "failed"
    assert "bad shape" in run["error"]


def test_unknown_run_returns_nothing(store):
    assert store.get_run("does-not-exist") is None


# -------------------------------------------------------------------- context
def test_a_run_records_what_produced_it(store):
    """Seed, code commit and dataset revision — enough to reproduce it."""
    run = store.get_run(a_run(store))
    assert run["seed"] == 7
    assert run["git_commit"] == "abc123"
    assert run["dataset"] == "me/ds@deadbeef:season53"
    assert run["config"]["fm"]["k"] == 32


# -------------------------------------------------------------------- metrics
def test_final_metrics_and_per_step_history_are_kept_apart(store):
    run_id = a_run(store)
    store.log_metrics(run_id, {"val_logloss": 0.66, "val_auc": 0.61})
    for epoch, loss in enumerate([0.69, 0.67, 0.66], start=1):
        store.log_metrics(run_id, {"train_loss": loss}, step=epoch)

    run = store.get_run(run_id)
    assert run["metrics"] == {"val_logloss": 0.66, "val_auc": 0.61}
    assert [h["value"] for h in run["history"]] == [0.69, 0.67, 0.66]


def test_none_valued_metrics_are_skipped(store):
    run_id = a_run(store)
    store.log_metrics(run_id, {"val_auc": 0.6, "val_brier": None})
    assert store.get_run(run_id)["metrics"] == {"val_auc": 0.6}


def test_relogging_a_metric_replaces_it(store):
    run_id = a_run(store)
    store.log_metrics(run_id, {"val_auc": 0.5})
    store.log_metrics(run_id, {"val_auc": 0.7})
    assert store.get_run(run_id)["metrics"]["val_auc"] == 0.7


# ------------------------------------------------------------------ artifacts
def test_artifacts_are_recorded_with_a_content_hash(store, tmp_path):
    f = tmp_path / "fm_model.pkl"
    f.write_bytes(b"weights")
    run_id = a_run(store)
    digest = store.log_artifact(run_id, "fm_model.pkl", f)

    (artifact,) = store.get_run(run_id)["artifacts"]
    assert artifact["sha256"] == digest == sha256_of(f)
    assert artifact["bytes"] == 7


def test_identical_models_share_a_hash(store, tmp_path):
    """Two runs whose configs differ but whose models match means the change
    did nothing — worth being able to see."""
    a, b = tmp_path / "a.pkl", tmp_path / "b.pkl"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    r1, r2 = a_run(store, name="one"), a_run(store, name="two")
    assert store.log_artifact(r1, "m", a) == store.log_artifact(r2, "m", b)


def test_a_missing_artifact_is_not_recorded(store, tmp_path):
    run_id = a_run(store)
    assert store.log_artifact(run_id, "absent", tmp_path / "nope.pkl") is None
    assert store.get_run(run_id)["artifacts"] == []


# ---------------------------------------------------------------- comparison
def test_best_finds_the_lowest_loss(store):
    for name, loss in [("a", 0.70), ("b", 0.66), ("c", 0.68)]:
        rid = a_run(store, name=name)
        store.log_metrics(rid, {"val_logloss": loss})
        store.finish(rid)
    assert store.best("val_logloss", mode="min")["name"] == "b"
    assert store.best("val_logloss", mode="max")["name"] == "a"


def test_best_ignores_runs_that_did_not_finish(store):
    """A crashed run's partial metrics must not win."""
    good = a_run(store, name="good")
    store.log_metrics(good, {"val_logloss": 0.66})
    store.finish(good)
    bad = a_run(store, name="crashed")
    store.log_metrics(bad, {"val_logloss": 0.01})
    store.finish(bad, status="failed", error="died")
    assert store.best("val_logloss", mode="min")["name"] == "good"


def test_best_of_nothing_is_none(store):
    assert store.best("val_logloss") is None


def test_best_rejects_a_nonsense_mode(store):
    with pytest.raises(ValueError, match="min.*max"):
        store.best("val_logloss", mode="sideways")


def test_runs_can_be_filtered_by_stage(store):
    a_run(store, stage="fm")
    a_run(store, stage="selfplay")
    assert len(store.list_runs(stage="fm")) == 1
    assert len(store.list_runs()) == 2


def test_metric_table_flattens_metrics_onto_runs(store):
    rid = a_run(store)
    store.log_metrics(rid, {"val_logloss": 0.66})
    store.finish(rid)
    (row,) = store.metric_table()
    assert row["val_logloss"] == 0.66 and row["status"] == "ok"


def test_runs_are_listed_newest_first(store):
    import time
    first = a_run(store, name="first")
    time.sleep(1.01)          # run ids and timestamps are second-resolution
    second = a_run(store, name="second")
    assert [r["run_id"] for r in store.list_runs()][:2] == [second, first]

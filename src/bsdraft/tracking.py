"""A record of every training run.

Seventeen self-play iterations were run and kept as `joint_net_iter_*.pkl` with
no record of what changed between them or which was best. That is the failure
this closes: a run that cannot be compared to another run teaches nothing.

Runs, metrics and artifacts live in one SQLite file so they can be queried
together. A run is written as `running` before any work happens and updated on
the way out, so a process that dies leaves evidence rather than nothing.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    stage           TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_utc     TEXT NOT NULL,
    finished_utc    TEXT,
    elapsed_seconds REAL,
    seed            INTEGER,
    git_commit      TEXT,
    dataset         TEXT,
    config_json     TEXT,
    error           TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL, key TEXT NOT NULL, value REAL, step INTEGER,
    PRIMARY KEY (run_id, key, step)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL, name TEXT NOT NULL, path TEXT NOT NULL,
    sha256 TEXT, bytes INTEGER, created_utc TEXT,
    PRIMARY KEY (run_id, name)
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_utc);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key);
"""


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of an artifact — identity independent of its filename."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class RunStore:
    """Runs, their metrics, and the artifacts they produced."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------- writing
    def start(
        self, *, name: str, stage: str, seed: int | None = None,
        git_commit: str | None = None, dataset: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        run_id = f"{stage}-{datetime.now(tz=UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (run_id, name, stage, status, started_utc, seed,"
                " git_commit, dataset, config_json) VALUES (?,?,?,'running',?,?,?,?,?)",
                (run_id, name, stage, datetime.now(tz=UTC).isoformat(), seed,
                 git_commit, dataset, json.dumps(config or {}, default=str, sort_keys=True)),
            )
        return run_id

    def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Record metrics. `step` distinguishes per-iteration values from finals."""
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO metrics (run_id, key, value, step) VALUES (?,?,?,?)",
                [(run_id, k, float(v), -1 if step is None else step)
                 for k, v in metrics.items() if v is not None],
            )

    def log_artifact(self, run_id: str, name: str, path: str | Path) -> str | None:
        """Register a file this run produced, with a content hash.

        The hash is what makes two runs comparable at the artifact level: an
        identical model from a changed config means the change did nothing.
        """
        p = Path(path)
        if not p.exists():
            return None
        digest = sha256_of(p)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, name, path, sha256, bytes,"
                " created_utc) VALUES (?,?,?,?,?,?)",
                (run_id, name, str(p), digest, p.stat().st_size,
                 datetime.now(tz=UTC).isoformat()),
            )
        return digest

    def finish(
        self, run_id: str, *, status: str = "ok", error: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status=?, finished_utc=?, elapsed_seconds=?, error=?"
                " WHERE run_id=?",
                (status, datetime.now(tz=UTC).isoformat(), elapsed_seconds,
                 error, run_id),
            )

    # ------------------------------------------------------------- reading
    def list_runs(
        self, *, stage: str | None = None, name: str | None = None, limit: int = 25
    ) -> list[dict]:
        where, params = [], []
        if stage:
            where.append("stage = ?")
            params.append(stage)
        if name:
            where.append("name = ?")
            params.append(name)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM runs {clause} ORDER BY started_utc DESC LIMIT ?", params
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            run = dict(row)
            run["config"] = json.loads(run.pop("config_json") or "{}")
            run["metrics"] = {
                m["key"]: m["value"] for m in c.execute(
                    "SELECT key, value FROM metrics WHERE run_id=? AND step=-1", (run_id,)
                )
            }
            run["history"] = [
                dict(m) for m in c.execute(
                    "SELECT key, value, step FROM metrics WHERE run_id=? AND step>=0"
                    " ORDER BY step", (run_id,)
                )
            ]
            run["artifacts"] = [dict(a) for a in c.execute(
                "SELECT name, path, sha256, bytes FROM artifacts WHERE run_id=?", (run_id,)
            )]
        return run

    def best(
        self, metric: str, *, stage: str | None = None, mode: str = "min",
    ) -> dict | None:
        """The run that scored best on a metric. `mode` is 'min' or 'max'."""
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        clause = "AND r.stage = ?" if stage else ""
        params: list[Any] = [metric] + ([stage] if stage else [])
        with self._conn() as c:
            row = c.execute(
                f"SELECT r.run_id FROM runs r JOIN metrics m ON m.run_id = r.run_id"
                f" WHERE m.key = ? AND m.step = -1 AND r.status = 'ok' {clause}"
                f" ORDER BY m.value {'ASC' if mode == 'min' else 'DESC'} LIMIT 1",
                params,
            ).fetchone()
        return self.get_run(row["run_id"]) if row else None

    def metric_table(self, *, stage: str | None = None, limit: int = 25) -> list[dict]:
        """Runs with their final metrics flattened in — one row per run."""
        out = []
        for run in self.list_runs(stage=stage, limit=limit):
            full = self.get_run(run["run_id"])
            out.append({**run, **(full["metrics"] if full else {})})
        return out

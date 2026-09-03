"""Scoring predictors against each other on the same held-out data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from bsdraft.eval.baselines import Baseline, default_baselines

#: log(2). What a predictor scores knowing nothing at all.
RANDOM_LOGLOSS = float(np.log(2))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Mean gap between predicted probability and observed frequency.

    A model used inside tree search must be calibrated, not merely ranked well:
    search multiplies probabilities together, so a confident-but-wrong evaluator
    compounds its error at every ply.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    total, error = len(probs), 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.sum() / total * abs(probs[mask].mean() - labels[mask].mean())
    return float(error)


def score(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    y = labels.astype(np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(p, y),
    }


@dataclass
class Comparison:
    """What every predictor scored on the same held-out rows."""

    n_train: int
    n_val: int
    rows: list[dict] = field(default_factory=list)

    @property
    def best(self) -> dict | None:
        return min(self.rows, key=lambda r: r["logloss"], default=None)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def render(self) -> str:
        if not self.rows:
            return "No predictors evaluated."
        width = max(len(r["name"]) for r in self.rows) + 2
        out = [
            f"Held-out comparison — {self.n_train:,} train / {self.n_val:,} val rows",
            "",
            f"{'predictor':<{width}}{'logloss':>10}{'vs random':>11}"
            f"{'auc':>9}{'brier':>9}{'ece':>9}",
            "-" * (width + 48),
        ]
        for r in self.rows:
            lift = RANDOM_LOGLOSS - r["logloss"]
            out.append(
                f"{r['name']:<{width}}{r['logloss']:>10.4f}{lift:>+11.4f}"
                f"{r['auc']:>9.4f}{r['brier']:>9.4f}{r['ece']:>9.4f}"
            )
        out += ["", f"random baseline logloss = {RANDOM_LOGLOSS:.4f}"]
        best = self.best
        if best:
            out.append(f"best: {best['name']} ({best['logloss']:.4f})")
        return "\n".join(out)


def compare_predictors(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    baselines: list[Baseline] | None = None,
    model_probs: np.ndarray | None = None,
    model_name: str = "factorization machine",
) -> Comparison:
    """Fit each baseline on train, score everything on the same val rows.

    `model_probs` is predicted separately because the model has its own feature
    pipeline; it must be aligned to `val` row for row.
    """
    labels = val["team1_wins"].to_numpy()
    result = Comparison(n_train=len(train), n_val=len(val))

    for baseline in (baselines if baselines is not None else default_baselines()):
        probs = baseline.fit(train).predict(val)
        result.rows.append({"name": baseline.name, **score(probs, labels)})

    if model_probs is not None:
        if len(model_probs) != len(val):
            raise ValueError(
                f"model produced {len(model_probs)} predictions for {len(val)} rows"
            )
        result.rows.append({"name": model_name, **score(model_probs, labels)})

    return result

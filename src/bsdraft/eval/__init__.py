"""Measuring what the model is worth relative to knowing less."""

from bsdraft.eval.baselines import Baseline, default_baselines
from bsdraft.eval.report import (
    RANDOM_LOGLOSS,
    Comparison,
    compare_predictors,
    expected_calibration_error,
    score,
)

__all__ = [
    "RANDOM_LOGLOSS",
    "Baseline",
    "Comparison",
    "compare_predictors",
    "default_baselines",
    "expected_calibration_error",
    "score",
]

"""Making a run repeatable.

Self-play draws every pick from an RNG, and network initialisation and batch
shuffling are random too. Without a single seed reaching all of them, two runs
of the same config produce different models and the config explains nothing.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and Torch. Returns the seed, for recording.

    `deterministic` also constrains cuDNN, which costs some speed and is worth
    it: a training run that cannot be reproduced cannot be debugged.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Seeding alone leaves reduction order free, which drifts weights by
            # ~1e-8 between runs of the same config. Close enough to look right,
            # different enough that two runs cannot be compared exactly.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Some ops have no deterministic implementation; seeding still
                # applies, and results stay reproducible to floating-point noise.
                pass
    except ImportError:  # pragma: no cover
        pass

    return seed

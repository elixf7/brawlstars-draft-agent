"""Two runs of one config must produce one model."""
import random

from bsdraft.seeding import seed_everything


def test_seeding_makes_python_random_repeatable():
    seed_everything(123)
    first = [random.random() for _ in range(5)]
    seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_seeding_reaches_numpy():
    import numpy as np
    seed_everything(7)
    first = np.random.rand(5).tolist()
    seed_everything(7)
    assert np.random.rand(5).tolist() == first


def test_seeding_reaches_torch():
    import torch
    seed_everything(7)
    first = torch.randn(4).tolist()
    seed_everything(7)
    assert torch.randn(4).tolist() == first


def test_different_seeds_diverge():
    seed_everything(1)
    a = [random.random() for _ in range(5)]
    seed_everything(2)
    assert [random.random() for _ in range(5)] != a


def test_seed_is_returned_for_recording():
    assert seed_everything(42) == 42


def test_strict_determinism_is_requested():
    """Seeding alone leaves reduction order free; without this two runs of the
    same config differ by ~1e-8."""
    import torch
    seed_everything(1, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()

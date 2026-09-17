"""The train/held-out split is what makes the ladder comparison meaningful, so
its invariants are worth pinning down: every rung must get the *same* split, and
no held-out episode may ever appear in training.
"""
import numpy as np

from vla.data.dataset import split_episodes


def test_split_is_a_partition():
    train, val = split_episodes(80, 0.1, seed=42)
    assert set(train) & set(val) == set()
    assert sorted(train + val) == list(range(80))


def test_split_is_deterministic_for_a_seed():
    # ACT/SmolVLA compute this in scripts/train.py and BC computes it inside its
    # own loop; if the two ever disagreed, models would train on each other's
    # eval episodes and every reported number would be contaminated.
    a = split_episodes(80, 0.1, seed=42)
    b = split_episodes(80, 0.1, seed=42)
    assert a == b


def test_different_seeds_give_different_splits():
    _, val_a = split_episodes(80, 0.1, seed=42)
    _, val_b = split_episodes(80, 0.1, seed=7)
    assert val_a != val_b


def test_val_fraction_is_respected():
    train, val = split_episodes(80, 0.25, seed=0)
    assert len(val) == 20
    assert len(train) == 60


def test_at_least_one_val_episode_even_for_tiny_fractions():
    # Guards against an empty held-out set silently producing a NaN metric.
    _, val = split_episodes(10, 0.01, seed=0)
    assert len(val) >= 1


def test_expected_holdout_for_the_lane_a_dataset():
    # Pinned so a library change to the RNG cannot silently move the split
    # underneath already-published numbers.
    _, val = split_episodes(80, 0.1, seed=42)
    assert val == [0, 25, 28, 33, 40, 51, 60, 61]

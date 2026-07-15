import numpy as np
import pytest

from vla.data.normalization import NormStats, DatasetStats


def test_normstats_standardizes():
    x = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    stats = NormStats.from_array(x)
    z = stats.normalize(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)


def test_normalize_denormalize_roundtrip():
    x = np.random.default_rng(0).normal(5, 3, size=(50, 6))
    stats = NormStats.from_array(x)
    assert np.allclose(stats.denormalize(stats.normalize(x)), x, atol=1e-6)


def test_constant_dimension_does_not_divide_by_zero():
    # A joint that never moves has std 0; the eps floor must keep it finite.
    x = np.column_stack([np.linspace(0, 1, 20), np.full(20, 7.0)])
    stats = NormStats.from_array(x)
    z = stats.normalize(x)
    assert np.all(np.isfinite(z))


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        NormStats.from_array(np.zeros((5,)))


def test_dataset_stats_save_load(tmp_path):
    rng = np.random.default_rng(1)
    ds = DatasetStats(
        state=NormStats.from_array(rng.normal(size=(30, 6))),
        action=NormStats.from_array(rng.normal(size=(30, 6))),
    )
    p = tmp_path / "stats.json"
    ds.save(p)
    loaded = DatasetStats.load(p)
    assert np.allclose(loaded.state.mean, ds.state.mean)
    assert np.allclose(loaded.action.std, ds.action.std)

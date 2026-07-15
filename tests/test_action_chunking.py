import numpy as np
import pytest

from vla.inference.action_chunking import TemporalEnsembler


def test_single_chunk_open_loop_equivalent():
    # With only one chunk and no overlap, ensembling returns that chunk's actions.
    e = TemporalEnsembler(action_dim=2, weight_m=0.01)
    chunk = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    e.add_chunk(chunk)
    out = [e.step() for _ in range(3)]
    assert np.allclose(out, chunk)


def test_overlapping_chunks_are_averaged():
    e = TemporalEnsembler(action_dim=1, weight_m=0.0)  # m=0 -> uniform average
    # Two chunks both predict timestep t=1.
    e.add_chunk(np.array([[0.0], [10.0]]))  # covers t=0,1
    _ = e.step()                            # consume t=0
    e.add_chunk(np.array([[20.0], [30.0]]))  # covers t=1,2
    # t=1 now has predictions 10 (age1 from chunk1) and 20 (age0 from chunk2).
    val = e.step()
    assert val[0] == pytest.approx(15.0)  # uniform mean of 10 and 20


def test_weighting_favors_newer_predictions():
    e = TemporalEnsembler(action_dim=1, weight_m=1.0)  # strong decay
    e.add_chunk(np.array([[0.0], [10.0]]))
    _ = e.step()
    e.add_chunk(np.array([[20.0], [30.0]]))
    val = e.step()[0]
    # Newer prediction (age 0, value 20) should dominate the older (age 1, value 10),
    # so the ensemble sits closer to 20 than the uniform 15.
    assert val > 15.0


def test_step_without_prediction_raises():
    e = TemporalEnsembler(action_dim=1)
    with pytest.raises(RuntimeError):
        e.step()


def test_bad_chunk_shape_raises():
    e = TemporalEnsembler(action_dim=3)
    with pytest.raises(ValueError):
        e.add_chunk(np.zeros((4, 2)))


def test_reset_clears_state():
    e = TemporalEnsembler(action_dim=1)
    e.add_chunk(np.array([[1.0]]))
    e.step()
    e.reset()
    assert e.t == 0
    assert e.pending_steps() == 0

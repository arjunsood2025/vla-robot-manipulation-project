import pytest

from vla.data.randomization import (
    RandomizationConfig,
    all_cells,
    cell_to_xy_m,
    generate_sheet,
    DEFAULT_HELDOUT_CELLS,
)


def test_all_cells_count():
    cells = all_cells()
    assert len(cells) == 24  # 6 columns x 4 rows
    assert "A1" in cells and "F4" in cells


def test_generate_sheet_is_deterministic():
    cfg = RandomizationConfig(num_episodes=30, objects=["red block", "green bowl"], seed=7)
    a = generate_sheet(cfg)
    b = generate_sheet(cfg)
    assert a == b


def test_generate_sheet_changes_with_seed():
    o = ["red block", "green bowl"]
    a = generate_sheet(RandomizationConfig(num_episodes=30, objects=o, seed=1))
    b = generate_sheet(RandomizationConfig(num_episodes=30, objects=o, seed=2))
    assert a != b


def test_heldout_cells_never_used_in_training_sheet():
    cfg = RandomizationConfig(num_episodes=200, objects=["red block", "green bowl"], seed=3)
    rows = generate_sheet(cfg)
    heldout = set(DEFAULT_HELDOUT_CELLS)
    for row in rows:
        for obj, cell in row.items():
            if obj == "episode":
                continue
            assert cell not in heldout


def test_objects_never_share_a_cell():
    cfg = RandomizationConfig(num_episodes=100, objects=["a", "b", "c"], seed=5)
    for row in generate_sheet(cfg):
        cells = [v for k, v in row.items() if k != "episode"]
        assert len(cells) == len(set(cells))


def test_min_separation_respected_when_feasible():
    cfg = RandomizationConfig(num_episodes=50, objects=["a", "b"], seed=9,
                              min_separation_cells=2)
    from vla.data.randomization import _cell_distance
    for row in generate_sheet(cfg):
        a, b = row["a"], row["b"]
        assert _cell_distance(a, b) >= 2


def test_too_many_objects_raises():
    cfg = RandomizationConfig(num_episodes=1, objects=[f"o{i}" for i in range(25)], seed=0)
    with pytest.raises(ValueError):
        generate_sheet(cfg)


def test_cell_to_xy_monotonic():
    x_a, y_a = cell_to_xy_m("A1")
    x_f, _ = cell_to_xy_m("F1")
    _, y_4 = cell_to_xy_m("A4")
    assert x_f > x_a   # column F is further in x than column A
    assert y_4 > y_a   # row 4 is further in y than row 1

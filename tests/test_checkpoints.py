import pytest

from vla.utils.checkpoints import resolve_last_checkpoint


def test_prefers_last_directory(tmp_path):
    ckpts = tmp_path / "checkpoints"
    (ckpts / "000100").mkdir(parents=True)
    (ckpts / "last").mkdir()
    assert resolve_last_checkpoint(ckpts).name == "last"


def test_falls_back_to_marker_file(tmp_path):
    # The Windows path: no symlink privilege, so a text marker names the newest.
    ckpts = tmp_path / "checkpoints"
    (ckpts / "000100").mkdir(parents=True)
    (ckpts / "000200").mkdir()
    (ckpts / "last.txt").write_text("000200", encoding="utf-8")
    assert resolve_last_checkpoint(ckpts).name == "000200"


def test_falls_back_to_highest_step_when_no_marker(tmp_path):
    # A run killed before any marker was written must still be recoverable.
    ckpts = tmp_path / "checkpoints"
    for step in ("000100", "000900", "001000"):
        (ckpts / step).mkdir(parents=True)
    assert resolve_last_checkpoint(ckpts).name == "001000"


def test_step_ordering_is_numeric_not_lexicographic(tmp_path):
    ckpts = tmp_path / "checkpoints"
    for step in ("9000", "10000"):
        (ckpts / step).mkdir(parents=True)
    assert resolve_last_checkpoint(ckpts).name == "10000"


def test_stale_marker_pointing_nowhere_falls_through(tmp_path):
    ckpts = tmp_path / "checkpoints"
    (ckpts / "000300").mkdir(parents=True)
    (ckpts / "last.txt").write_text("000999", encoding="utf-8")
    assert resolve_last_checkpoint(ckpts).name == "000300"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_last_checkpoint(tmp_path / "nope")


def test_empty_directory_raises(tmp_path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_last_checkpoint(ckpts)

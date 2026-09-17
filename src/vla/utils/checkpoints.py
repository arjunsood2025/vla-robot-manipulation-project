"""Locate the newest LeRobot checkpoint, whichever way it was marked.

LeRobot points at its most recent checkpoint with a ``checkpoints/last``
symlink. On Windows without Developer Mode that symlink cannot be created, so
``vla.training.lerobot_launch`` writes a ``checkpoints/last.txt`` marker file
instead. Consumers (the policy server, evaluation) should not have to know which
mechanism was used, so they go through this helper.
"""
from __future__ import annotations

from pathlib import Path

LAST_CHECKPOINT_TXT = "last.txt"
LAST_CHECKPOINT_LINK = "last"


def resolve_last_checkpoint(checkpoints_dir: str | Path) -> Path:
    """Return the newest checkpoint directory under ``checkpoints_dir``.

    Resolution order:
      1. the ``last`` symlink/directory LeRobot normally creates
      2. the ``last.txt`` marker written by the Windows fallback
      3. the highest-numbered checkpoint directory (LeRobot names them by
         zero-padded step, e.g. ``000060``), so a run interrupted before any
         marker was written is still recoverable

    Raises FileNotFoundError if no checkpoint can be found.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints directory at {checkpoints_dir}")

    link = checkpoints_dir / LAST_CHECKPOINT_LINK
    if link.is_dir():
        return link.resolve()

    marker = checkpoints_dir / LAST_CHECKPOINT_TXT
    if marker.is_file():
        target = checkpoints_dir / marker.read_text(encoding="utf-8").strip()
        if target.is_dir():
            return target

    # Fall back to the highest step number present.
    candidates = sorted(
        (p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"No checkpoints found under {checkpoints_dir}")

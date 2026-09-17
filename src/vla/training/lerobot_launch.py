"""Launch LeRobot's trainer in-process, with the Windows compatibility shims
this project needs.

Why this module exists
----------------------
``scripts/train.py`` used to shell out to the ``lerobot-train`` console script.
That works on Linux but dies on Windows at the first checkpoint:

    OSError: [WinError 1314] A required privilege is not held by the client

LeRobot marks the newest checkpoint by creating a ``checkpoints/last`` symlink.
Creating a symlink on Windows requires either Developer Mode or an elevated
process; on a stock Windows 11 user account it raises, and because the call is
not guarded the whole training run dies *after* the checkpoint was already
written safely to disk. Hours of GPU time are lost to a convenience alias.

Rather than patch ``site-packages`` (which any ``pip install`` would silently
undo, and which would not survive a clone of this repo onto another machine) we
launch the trainer in-process and replace that one function with a version that
falls back to a plain marker file when symlinks are unavailable. The checkpoint
itself is untouched — only how "which one is newest" gets recorded.

``vla.utils.checkpoints.resolve_last_checkpoint`` reads either form, so the rest
of the project does not care which mechanism was used.

Run via ``scripts/train.py --policy act|smolvla``; not intended to be called
directly.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Written next to the checkpoints when a real symlink cannot be created.
LAST_CHECKPOINT_TXT = "last.txt"


def _install_symlink_fallback() -> None:
    """Make LeRobot's ``update_last_checkpoint`` degrade gracefully.

    Order of preference:
      1. a real symlink (what LeRobot does; works with Developer Mode on)
      2. a ``last.txt`` file naming the newest checkpoint directory

    We deliberately do *not* copy the checkpoint directory as a fallback: ACT
    and SmolVLA checkpoints are hundreds of MB and doing so every ``save_freq``
    steps would quietly fill the disk.
    """
    try:
        from lerobot.utils.constants import LAST_CHECKPOINT_LINK
    except ImportError:  # pragma: no cover - older LeRobot layout
        from lerobot.constants import LAST_CHECKPOINT_LINK
    from lerobot.utils import train_utils

    def update_last_checkpoint(checkpoint_dir: Path) -> Path:
        parent = checkpoint_dir.parent
        link = parent / LAST_CHECKPOINT_LINK
        relative_target = checkpoint_dir.relative_to(parent)

        if link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(relative_target)
        except OSError:
            # Windows without Developer Mode / admin. Record the pointer as
            # text instead; see resolve_last_checkpoint().
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            (parent / LAST_CHECKPOINT_TXT).write_text(str(relative_target), encoding="utf-8")
        return checkpoint_dir

    train_utils.update_last_checkpoint = update_last_checkpoint
    # lerobot_train.py does `from lerobot.utils.train_utils import ...`, binding
    # the original function into its own namespace, so patch that too.
    from lerobot.scripts import lerobot_train

    if hasattr(lerobot_train, "update_last_checkpoint"):
        lerobot_train.update_last_checkpoint = update_last_checkpoint


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    _install_symlink_fallback()

    from lerobot.scripts.lerobot_train import main as lerobot_main

    # LeRobot parses its config from sys.argv via draccus.
    sys.argv = [sys.argv[0], *argv]
    lerobot_main()


if __name__ == "__main__":
    main()

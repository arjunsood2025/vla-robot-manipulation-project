"""The one definition of the train/held-out episode split.

Every rung of the model ladder must be trained on the *same* episodes and
scored on the *same* held-out episodes, or the comparison between them measures
the split as much as the model.

The from-scratch BC loop computes its split internally from
``vla.data.dataset.split_episodes``. ACT and SmolVLA are trained by LeRobot's
own trainer, which by default consumes **every** episode in the dataset — so
without passing it an explicit episode list they would train on the episodes BC
holds out, and any "held-out action error" reported for them would be measured
on data they had already seen. This module exists so both paths derive the
split from one function with one seed.

Held out for ``youliangtan/so101-table-cleanup`` (80 episodes, val_fraction
0.1, seed 42): episodes [0, 25, 28, 33, 40, 51, 60, 61].
"""
from __future__ import annotations

from vla.data.dataset import split_episodes


def dataset_num_episodes(repo_id: str, root: str | None = None) -> int:
    """Episode count from the dataset metadata, without loading any video."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ModuleNotFoundError:  # pragma: no cover - legacy LeRobot layout
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id, root=root).total_episodes


def train_val_episodes(
    repo_id: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    root: str | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(train_episode_ids, val_episode_ids)`` for a dataset.

    Identical in behaviour to what ``train_bc`` does internally, so the BC
    baseline and the LeRobot-trained rungs agree on which episodes are held out.
    """
    n = dataset_num_episodes(repo_id, root=root)
    return split_episodes(n, val_fraction, seed)

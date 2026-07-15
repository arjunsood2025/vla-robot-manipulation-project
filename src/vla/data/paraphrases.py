"""Loader for the instruction paraphrase bank (configs/paraphrases.json).

The bank cleanly separates 'train' phrasings (rotated through during recording)
from 'heldout' phrasings (reserved for evaluation). This separation is the
mechanism behind the paraphrase-generalization metric: success on heldout
phrasings minus success on train phrasings tells us whether the policy learned
the *task* or just memorised a sentence.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskParaphrases:
    task_id: str
    curriculum_id: int
    objects: dict[str, str]
    train: list[str]
    heldout: list[str]

    def sample(self, source: str, rng: random.Random | None = None) -> str:
        """Return one instruction from either the 'train' or 'heldout' pool."""
        pool = self.train if source == "train" else self.heldout
        if not pool:
            raise ValueError(f"Task '{self.task_id}' has no '{source}' phrasings.")
        chooser = rng or random
        return chooser.choice(pool)


class ParaphraseBank:
    def __init__(self, tasks: dict[str, TaskParaphrases], heldout_axes: dict):
        self._tasks = tasks
        self.heldout_axes = heldout_axes

    @classmethod
    def load(cls, path: str | Path) -> "ParaphraseBank":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tasks = {
            tid: TaskParaphrases(
                task_id=tid,
                curriculum_id=spec.get("curriculum_id", -1),
                objects=spec.get("objects", {}),
                train=spec["train"],
                heldout=spec["heldout"],
            )
            for tid, spec in raw["tasks"].items()
        }
        return cls(tasks, raw.get("heldout_axes", {}))

    def __getitem__(self, task_id: str) -> TaskParaphrases:
        return self._tasks[task_id]

    def task_ids(self) -> list[str]:
        return list(self._tasks.keys())

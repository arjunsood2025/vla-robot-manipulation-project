"""Schema + loader for the evaluation trial specification.

A trial spec pins down everything that must be held constant across the models
being compared: the task, where each object starts (grid cells), which distractors
are present, whether the instruction is drawn from the train or held-out phrasing
pool, and the RNG seed. Making this an explicit, versioned file is what lets the
BC-vs-ACT-vs-SmolVLA-vs-OpenVLA table be a fair comparison rather than four runs
under subtly different conditions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Condition:
    condition_id: str
    task_id: str
    phrasing_source: str            # "train" | "heldout"
    layout: dict[str, str]          # object name -> grid cell
    distractors: list[str] = field(default_factory=list)  # "name:cell" strings
    seed: int = 0


@dataclass
class TrialSuite:
    suite_name: str
    trials_per_condition: int
    conditions: list[Condition]

    @classmethod
    def load(cls, path: str | Path) -> "TrialSuite":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        conditions = [
            Condition(
                condition_id=c["condition_id"],
                task_id=c["task_id"],
                phrasing_source=c["phrasing_source"],
                layout=c["layout"],
                distractors=c.get("distractors", []),
                seed=c.get("seed", 0),
            )
            for c in raw["conditions"]
        ]
        return cls(
            suite_name=raw["suite_name"],
            trials_per_condition=raw["trials_per_condition"],
            conditions=conditions,
        )

    def validate(self, valid_task_ids: set[str]) -> None:
        """Fail fast on a malformed suite before an operator wastes an hour."""
        for c in self.conditions:
            if c.task_id not in valid_task_ids:
                raise ValueError(f"Condition '{c.condition_id}' references unknown task '{c.task_id}'")
            if c.phrasing_source not in ("train", "heldout"):
                raise ValueError(
                    f"Condition '{c.condition_id}' phrasing_source must be train|heldout"
                )

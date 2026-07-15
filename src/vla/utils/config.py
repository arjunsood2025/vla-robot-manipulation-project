"""YAML config loading with dotted-key overrides.

Keeping config in YAML (not argparse) is deliberate: every experiment's exact
settings are a file you can commit next to its results, which is what makes the
ablation table auditable months later.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge ``updates`` into a deep copy of ``base``.

    Used to apply CLI ``--override key.subkey=value`` on top of a YAML config
    without mutating the loaded file.
    """
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ["train.lr=1e-3", "seed=7"] into a nested dict for ``deep_update``.

    Values are parsed as YAML scalars so ``1e-3`` -> float, ``true`` -> bool,
    ``[1,2]`` -> list, and everything else stays a string.
    """
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must be of the form key.subkey=value")
        dotted, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = result
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return result

"""Seeded object-placement randomization over the 6x4 workspace grid.

The workspace mat is a 6-column (A-F) by 4-row (1-4) grid of 10 cm cells. Before
every recorded episode the operator places objects into the cells named on a
pre-generated sheet. Generating the sheet ONCE from a fixed seed is what makes
data collection reproducible and, crucially, lets us reserve a handful of cells
that are never used in training so we can measure unseen-position generalization
at evaluation time.

Pure-python + numpy only, so it is unit-tested on any machine (no GPU/torch).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = ["A", "B", "C", "D", "E", "F"]
ROWS = [1, 2, 3, 4]

# Cells reserved for held-out evaluation ONLY. Training layouts never touch these,
# which is what makes "unseen object position success" an honest metric.
DEFAULT_HELDOUT_CELLS = ("A1", "F4", "A4", "F1")


def all_cells() -> list[str]:
    """Every grid cell label, e.g. ['A1', 'A2', ..., 'F4']."""
    return [f"{c}{r}" for c in COLUMNS for r in ROWS]


def cell_to_xy_m(cell: str, cell_size_m: float = 0.10, origin: str = "A1") -> tuple[float, float]:
    """Map a grid label to the (x, y) centre of the cell in metres.

    Column letter -> x, row number -> y. The origin cell's centre is at
    (cell_size/2, cell_size/2). Handy for programmatically checking that a
    randomized layout stays inside the reachable workspace.
    """
    col = COLUMNS.index(cell[0])
    row = ROWS.index(int(cell[1:]))
    x = (col + 0.5) * cell_size_m
    y = (row + 0.5) * cell_size_m
    return x, y


@dataclass
class RandomizationConfig:
    num_episodes: int
    objects: list[str]          # e.g. ["red block", "green bowl"]
    seed: int = 42
    heldout_cells: tuple[str, ...] = DEFAULT_HELDOUT_CELLS
    min_separation_cells: int = 1   # objects can't share a cell; >1 spaces them out


def generate_sheet(cfg: RandomizationConfig) -> list[dict[str, str]]:
    """Produce one placement row per episode.

    Each row is {"episode": i, "<object>": "<cell>", ...}. Placements are drawn
    only from the TRAINING cell pool (all cells minus the reserved held-out set)
    and no two objects in an episode occupy the same cell.
    """
    rng = np.random.default_rng(cfg.seed)
    train_pool = [c for c in all_cells() if c not in set(cfg.heldout_cells)]
    if len(train_pool) < len(cfg.objects):
        raise ValueError(
            f"{len(cfg.objects)} objects but only {len(train_pool)} training cells available."
        )

    rows: list[dict[str, str]] = []
    for ep in range(cfg.num_episodes):
        chosen: dict[str, str] = {}
        used: set[str] = set()
        for obj in cfg.objects:
            candidates = [c for c in train_pool if c not in used]
            if cfg.min_separation_cells > 1 and used:
                candidates = [
                    c for c in candidates
                    if all(_cell_distance(c, u) >= cfg.min_separation_cells for u in used)
                ]
                # Fall back to any free cell if separation is unsatisfiable.
                if not candidates:
                    candidates = [c for c in train_pool if c not in used]
            cell = str(rng.choice(candidates))
            chosen[obj] = cell
            used.add(cell)
        rows.append({"episode": str(ep), **chosen})
    return rows


def _cell_distance(a: str, b: str) -> int:
    """Chebyshev distance between two cells in grid units."""
    ax, ay = COLUMNS.index(a[0]), int(a[1:])
    bx, by = COLUMNS.index(b[0]), int(b[1:])
    return max(abs(ax - bx), abs(ay - by))


def write_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a seeded object-placement sheet.")
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--objects", nargs="+", required=True,
                        help='Object names, e.g. --objects "red block" "green bowl"')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--min-separation-cells", type=int, default=1)
    args = parser.parse_args()

    cfg = RandomizationConfig(
        num_episodes=args.num_episodes,
        objects=args.objects,
        seed=args.seed,
        min_separation_cells=args.min_separation_cells,
    )
    rows = generate_sheet(cfg)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} episode layouts to {args.out}")
    print(f"Held-out cells reserved for eval (never used here): {list(cfg.heldout_cells)}")


if __name__ == "__main__":
    main()

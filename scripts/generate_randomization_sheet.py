#!/usr/bin/env python
"""Entrypoint for the seeded placement-sheet generator (see vla.data.randomization).

    python scripts/generate_randomization_sheet.py \
        --num-episodes 50 --objects "red block" "green bowl" \
        --seed 42 --out data/layouts/block_in_bowl.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.data.randomization import main  # noqa: E402

if __name__ == "__main__":
    main()

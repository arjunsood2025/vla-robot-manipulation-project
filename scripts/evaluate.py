#!/usr/bin/env python
"""Thin entrypoint for the evaluation harness (see vla.eval.evaluate).

    # hardware-free smoke test of the harness itself:
    python scripts/evaluate.py --suite configs/eval_trials_example.json --dry-run

    # real evaluation against a running policy server:
    python scripts/evaluate.py --suite configs/eval_trials_example.json \
        --server-uri ws://gpu-box:8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.eval.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()

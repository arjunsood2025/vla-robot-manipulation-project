#!/usr/bin/env python
"""Entrypoint for the policy inference server (see vla.inference.policy_server).

    python scripts/run_policy_server.py \
        --policy-kind bc --checkpoint outputs/bc_baseline_v1/bc_best.pt --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.inference.policy_server import main  # noqa: E402

if __name__ == "__main__":
    main()

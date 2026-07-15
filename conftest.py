"""Make ``src`` importable and pin the working directory to the repo root so that
tests can load configs via relative paths (e.g. "configs/robot_so101.yaml")
without an editable install. Run pytest from the repository root.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

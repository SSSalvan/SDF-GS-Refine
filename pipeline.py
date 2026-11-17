"""Command-line entry point for SDF-GS Refine pipeline.

Usage:
    python pipeline.py --scan 24 --stage1_iters 3500 --stage2_iters 1000 --grid_res 480
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'src' is importable as a package
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.pipeline import main_cli

if __name__ == "__main__":
    main_cli()

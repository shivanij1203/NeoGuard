"""Test configuration. Adds the backend dir to sys.path so tests can import
the same module names the running service uses (`ml`, `config`, etc.) without
needing an installable package layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

"""The crawler and dataset commands must stay light: no scikit-learn, no discord.py, no pandas unless needed."""

from __future__ import annotations

import subprocess
import sys

CODE = """
import os, sys
from oracle.cli import main
code = main(["status"])
heavy = sorted(m for m in ("sklearn", "joblib", "discord", "aiohttp", "pandas") if m in sys.modules)
print(code, heavy)
"""


def test_status_loads_none_of_the_bot_libraries(tmp_path):
    env = {"PATH": "", "RIOT_PLATFORM": "eun1", "ORACLE_DATA_DIR": str(tmp_path / "data")}
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert out.stdout.strip().splitlines()[-1] == "0 []", out.stderr

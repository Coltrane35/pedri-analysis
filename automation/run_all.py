# automation/run_all.py
# -*- coding: utf-8 -*-
"""
Run the Pedri pipeline strictly within this repo.
- Prefers venv python
- Searches only in <ROOT>/core and <ROOT>
- Refuses to run scripts outside ROOT (guards typos like 'pedrii-analysis')
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prefer venv python if available
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

SCRIPTS = [
    "pedri_inspect_lineups.py",  # optional
    "pedri_profile.py",
    "find_pedri_matches_in_events.py",  # optional
    "pedri_analysis.py",
]

CANDIDATE_DIRS = [
    ROOT / "core",  # prefer core/
    ROOT,  # fallback (root)
]


def is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def find_script(name: str) -> Path | None:
    for base in CANDIDATE_DIRS:
        p = (base / name).resolve()
        if p.exists() and is_within(p, ROOT):
            return p
    return None


def run_script(script_name: str):
    script_path = find_script(script_name)
    if not script_path:
        print(f"ℹ️  Pomijam: {script_name} (nie znaleziono w {', '.join(str(d) for d in CANDIDATE_DIRS)})")
        return

    env = os.environ.copy()
    # PYTHONPATH: tylko ten projekt
    pp = os.pathsep.join([str(ROOT), str(ROOT / "core")])
    env["PYTHONPATH"] = pp

    print(f"\n▶️  Running: {PY} {script_path}  (cwd={ROOT})")
    print(f"   Using PYTHONPATH={env['PYTHONPATH']}")
    proc = subprocess.run([PY, str(script_path)], cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        print(f"❌ Step failed: {script_name}")
        sys.exit(proc.returncode)
    print("✅ OK")


def main():
    print(f"🔧 Project root: {ROOT}")
    print(f"🐍 Interpreter:  {PY}")
    for s in SCRIPTS:
        run_script(s)
    print("\n🎯 Pipeline finished.")


if __name__ == "__main__":
    main()

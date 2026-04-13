"""
Demonstration entry point for the EATM subway comfort stack.

Delegates to the scripted scenario under ``scripts/`` so the repository root
remains a convenient launch location for batch jobs.
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    scenario = Path(__file__).resolve().parent / "scripts" / "run_platform_to_carriage_scenario.py"
    runpy.run_path(str(scenario), run_name="__main__")

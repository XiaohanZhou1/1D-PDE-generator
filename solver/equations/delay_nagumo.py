"""Bistable equation with a delayed term.

Only the equations and the parameter values: the solver searches for the waves.
The equations are those of problems/delay_nagumo_front.py.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).resolve().parent.parent / "problems" / "delay_nagumo_front.py")).items()
                  if not key.startswith("__") and key != "wave"})

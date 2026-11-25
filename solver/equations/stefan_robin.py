"""Free boundary with a Neumann condition (r = 0).

Only the equations and the parameter values: the solver searches for the waves.
The equations are those of problems/stefan_robin.py.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).resolve().parent.parent / "problems" / "stefan_robin.py")).items()
                  if not key.startswith("__") and key != "wave"})

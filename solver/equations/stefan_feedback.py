"""Free boundary with a component without diffusion.

Only the equations and the parameter values: the solver searches for the waves.
The equations are those of problems/stefan_feedback.py.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).resolve().parent.parent / "problems" / "stefan_feedback.py")).items()
                  if not key.startswith("__") and key != "wave"})

"""Extended Fisher-Kolmogorov equation (fourth order).

Only the equations and the parameter values: the solver searches for the waves.
The equations are those of problems/efk_kink.py.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).resolve().parent.parent / "problems" / "efk_kink.py")).items()
                  if not key.startswith("__") and key != "wave"})

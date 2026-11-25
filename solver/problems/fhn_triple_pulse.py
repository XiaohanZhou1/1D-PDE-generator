"""FitzHugh-Nagumo equations: triple pulses (same equations and parameters as fhn_fast_pulse.py).

Three pulses with two gaps.  Each pulse interacts with the wake of the pulse ahead of it,
so each gap behaves like a double pulse: the triple pulse is stable when both gaps are
stable distances.  D0 = 51, D1 = 90: stable.   D0 = 51, D1 = 70: unstable.
"""
import runpy
from pathlib import Path

import sympy as sp

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("fhn_fast_pulse.py"))).items()
                  if not key.startswith("__")})

D0, D1 = sp.symbols("D0 D1", positive=True)
parameters = {**parameters, D0: 51.0, D1: 90.0}
wave = dict(wave, copies=3, separation=[D0, D1])

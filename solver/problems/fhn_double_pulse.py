"""FitzHugh-Nagumo equations: double pulses (same equations and parameters as fhn_fast_pulse.py).

The rest state is a saddle-focus behind the fast pulse (eigenvalues 0.116 +- 0.161 i):
the wake oscillates.  A second pulse can travel in this wake at a discrete set of
distances D_1 < D_2 < ..., spaced by about pi / 0.161 = 19.5.
Known result (Yanagida and Maginu 1989; Sandstede 1998): the double pulses of a stable
primary pulse are alternately stable and unstable.

The solver takes the admissible distance nearest to `separation`.  D0 is not a parameter of
the equations; it only selects the wave:  --set D0=70  gives the next double pulse.
D0 = 51: distance 51.385, stable.   D0 = 70: 70.630, unstable.   D0 = 90: 90.205, stable.
"""
import runpy
from pathlib import Path

import sympy as sp

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("fhn_fast_pulse.py"))).items()
                  if not key.startswith("__")})

D0 = sp.Symbol("D0", positive=True)
parameters = {**parameters, D0: 51.0}
wave = dict(wave, copies=2, separation=D0)

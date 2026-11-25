"""stefan_passive.py for a retreating boundary (c < 0).

The characteristics of w leave the domain at the boundary: w has no boundary condition.
Wave: U = exp(m z), W = exp(m z) / (1 - c m), m = -1/(1 + kappa), c = (1 - m^2)/m.
For kappa = -1.5: m = 2, c = -1.5, lambda = kappa (1 - m) = 1.5: unstable.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("stefan_passive.py"))).items()
                  if not key.startswith("__")})

free_boundary = dict(conditions=[sp.Eq(u, 1)], speed=kappa * (u.diff(x) - 1))
parameters = {kappa: -1.5}

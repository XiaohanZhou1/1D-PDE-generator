"""stefan_robin_moving.py with a component without diffusion; the boundary advances (c > 0).

    u_t = u_xx - u,   w_t = u - w            on x < s(t),
    u_x = 1,   w = 0                         at x = s(t),
    ds/dt = kappa (u - 1)                    at x = s(t).

The eigenvalues are those of stefan_robin_moving.py (w does not act on u or on the boundary).
For kappa = 1.5: c = 1.5, lambda = 0.209431: unstable.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("stefan_robin_moving.py"))).items()
                  if not key.startswith("__")})

w = sp.Function("w")(x, t)
equations = equations + [sp.Eq(w.diff(t), u - w)]
free_boundary = dict(conditions=[sp.Eq(u.diff(x) + r * u, 1 + r), sp.Eq(w, 0)], speed=kappa * (u - 1))
parameters = {kappa: 1.5, r: 0.0}
wave = dict(wave, far={u: 0, w: 0})

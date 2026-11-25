"""stefan_linear_moving.py with a component without diffusion; the boundary advances (c > 0).

    u_t = u_xx - u,   w_t = u - w            on x < s(t),
    u = 1,   w = 0                           at x = s(t),
    ds/dt = kappa (u_x - 1)                  at x = s(t).

The characteristics of w enter the domain at the boundary if c > 0: w needs a boundary value.
Wave: U = exp(m z), W = (exp(m z) - exp(z / c)) / (1 - c m), m = -1/(1 + kappa), c = (1 - m^2)/m.
w does not act on u or on the boundary, and its own operator has no eigenvalue to the
right of Re(lambda) = -1.  The eigenvalues are those of stefan_linear_moving.py.
For kappa = -3: c = 1.5, no eigenvalue: stable.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("stefan_linear_moving.py"))).items()
                  if not key.startswith("__")})

w = sp.Function("w")(x, t)
equations = equations + [sp.Eq(w.diff(t), u - w)]
free_boundary = dict(conditions=[sp.Eq(u, 1), sp.Eq(w, 0)], speed=kappa * (u.diff(x) - 1))
wave = dict(wave, far={u: 0, w: 0})

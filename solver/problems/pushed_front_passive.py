"""The pushed front of nagumo_pushed.py with a passive component: eigenvalues in the weighted space.

    u_t = u_xx + u (1 - u) (u - a),
    v_t = v_xx - v + sigma u (1 - u) v.

In L2 the wave is unstable through the essential spectrum of u = 0.  In the weighted space
the essential spectrum is stable, and the v-block decides: its eigenvalues are
    (s - n)^2 / 8 - 1 - c^2 / 4,     s (s + 1) = 2 sigma,     c = (1 - 2a) / sqrt(2).
a = -0.25, sigma = 15: 1.84375 and 0.71875 are positive: unstable also in the weighted space.
sigma = 3: all negative: stable in the weighted space.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("nagumo_pushed.py"))).items()
                  if not key.startswith("__")})

sigma = sp.Symbol("sigma", positive=True)
v = sp.Function("v")(x, t)
unknowns = [u, v]
equations = equations + [sp.Eq(v.diff(t), v.diff(x, 2) - v + sigma * u * (1 - u) * v)]
parameters = {a: -0.25, sigma: 15.0}
wave = dict(wave, left={u: 1, v: 0}, right={u: 0, v: 0})

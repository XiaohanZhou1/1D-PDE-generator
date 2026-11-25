"""Haptotaxis front with a passive component: eigenvalues from a scalar reference problem.

    u_t = eps u_xx - chi (u m_x)_x + u (1 - u) (u - a),
    m_t = mu (1 - m) - k u m,
    v_t = d v_xx - v + sigma u (1 - u) v.

On the front of haptotaxis_front.py, v = 0.  The linearisation is block triangular; the v-block is
    lambda r = d r'' + c r' + (sigma U (1 - U) - 1) r,
a scalar operator on the computed profile.  With r = exp(-c z / (2 d)) s it is symmetric; its
eigenvalues come from a tridiagonal matrix (tests/test_core.py).  They do not involve the solver.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
eps, chi, a, mu, k, d, sigma = sp.symbols("eps chi a mu k d sigma", positive=True)
u, m, v = sp.Function("u")(x, t), sp.Function("m")(x, t), sp.Function("v")(x, t)

equations = [sp.Eq(u.diff(t), eps * u.diff(x, 2) - chi * (u * m.diff(x)).diff(x) + u * (1 - u) * (u - a)),
             sp.Eq(m.diff(t), mu * (1 - m) - k * u * m),
             sp.Eq(v.diff(t), d * v.diff(x, 2) - v + sigma * u * (1 - u) * v)]
parameters = {eps: 1.0, chi: 1.0, a: 0.25, mu: 0.5, k: 1.0, d: 1.0, sigma: 30.0}
wave = dict(kind="front", left={u: 1, m: mu / (mu + k), v: 0}, right={u: 0, m: 1, v: 0}, speed_guess=0.4, width=2.0)
options = dict(region=(8.0, 8.0))

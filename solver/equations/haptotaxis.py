"""Haptotaxis with a remodelling matrix, without a wave (see problems/haptotaxis_front.py).

    u_t = eps u_xx - chi (u m_x)_x + u (1 - u) (u - a),      m_t = mu (1 - m) - k u m.

The solver finds the equilibria, the front between the two stable ones, and its label.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
eps, chi, a, mu, k = sp.symbols("eps chi a mu k", positive=True)
u, m = sp.Function("u")(x, t), sp.Function("m")(x, t)

equations = [sp.Eq(u.diff(t), eps * u.diff(x, 2) - chi * (u * m.diff(x)).diff(x) + u * (1 - u) * (u - a)),
             sp.Eq(m.diff(t), mu * (1 - m) - k * u * m)]
parameters = {eps: 1.0, chi: 1.0, a: 0.25, mu: 0.5, k: 1.0}
options = dict(equilibria_range=3.0)

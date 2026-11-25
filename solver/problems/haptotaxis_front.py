"""Haptotaxis with a remodelling matrix: an invasion front (transport towards a component without diffusion).

    u_t = eps u_xx - chi (u m_x)_x + u (1 - u) (u - a),
    m_t = mu (1 - m) - k u m.

u: cell density; m: density of the extracellular matrix.  The cells move up the gradient of m
(haptotaxis), they degrade it, and it is rebuilt at the rate mu.  m does not diffuse, but m_xx appears
in the equation of u: the matrix of the highest derivatives is singular.
End states: (u, m) = (1, mu / (mu + k)) behind the front and (0, 1) ahead of it; both are stable.
The search rectangle is given: no energy estimate covers the taxis term.
There is no closed form.  The label is compared with an independent difference operator of the
original equations (tests/test_core.py).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
eps, chi, a, mu, k = sp.symbols("eps chi a mu k", positive=True)
u, m = sp.Function("u")(x, t), sp.Function("m")(x, t)

equations = [sp.Eq(u.diff(t), eps * u.diff(x, 2) - chi * (u * m.diff(x)).diff(x) + u * (1 - u) * (u - a)),
             sp.Eq(m.diff(t), mu * (1 - m) - k * u * m)]
parameters = {eps: 1.0, chi: 1.0, a: 0.25, mu: 0.5, k: 1.0}
wave = dict(kind="front", left={u: 1, m: mu / (mu + k)}, right={u: 0, m: 1}, speed_guess=0.4, width=2.0)
options = dict(region=(4.0, 8.0))

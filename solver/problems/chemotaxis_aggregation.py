"""Chemotaxis towards a signal that the cells produce and that does not diffuse: aggregation.

    u_t = eps u_xx - chi (u m_x)_x + u (1 - u) (u - a),
    m_t = mu (u - m).

At the state u = m = 1 the dispersion relation is
    lambda^2 + (eps kappa^2 + 1 - a + mu) lambda + mu (kappa^2 (eps - chi) + 1 - a) = 0.
For chi > eps one root is positive at large wavenumbers; it tends to  mu (chi / eps - 1)  as
kappa -> infinity (aggregation at small scales).  Every wave that ends at this state is unstable.
For chi < eps the state is stable, and the front gets a label.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
eps, chi, a, mu = sp.symbols("eps chi a mu", positive=True)
u, m = sp.Function("u")(x, t), sp.Function("m")(x, t)

equations = [sp.Eq(u.diff(t), eps * u.diff(x, 2) - chi * (u * m.diff(x)).diff(x) + u * (1 - u) * (u - a)),
             sp.Eq(m.diff(t), mu * (u - m))]
parameters = {eps: 1.0, chi: 2.0, a: 0.25, mu: 0.5}
wave = dict(kind="front", left={u: 1, m: 1}, right={u: 0, m: 0}, speed_guess=0.3, width=2.0)
options = dict(region=(4.0, 8.0))

"""Fisher-Stefan problem: the semi-wave.

    u_t = u_xx + u (1 - u)       on x < s(t),
    u = 0                        at x = s(t),
    ds/dt = -kappa u_x           at x = s(t).

Known results: for every kappa > 0 there is one semi-wave; spreading solutions
converge to it (Du and Lin 2010), so it is stable.  For small kappa the speed
is c = kappa / sqrt(3) to leading order.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
kappa = sp.Symbol("kappa", positive=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u))]
free_boundary = dict(conditions=[sp.Eq(u, 0)], speed=-kappa * u.diff(x))
parameters = {kappa: 1.0}
wave = dict(kind="semi", far={u: 1}, speed_guess=kappa / sp.sqrt(3), width=1.5)

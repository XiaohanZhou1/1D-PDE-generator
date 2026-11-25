"""Undamped nonlinear wave equation (phi^4 model), without a wave (see problems/phi4_kink.py).

    u_tt = u_xx + u - u^3.

There is no damping: the waves form a family in the speed.  The solver lists the standing waves.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t, 2), u.diff(x, 2) + u - u ** 3)]
parameters = {}
options = dict(equilibria_range=3.0)

"""Undamped nonlinear wave equation with a focusing cubic term, without a wave (see problems/klein_gordon_pulse.py).

    u_tt = u_xx - u + 2 u^3.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t, 2), u.diff(x, 2) - u + 2 * u ** 3)]
parameters = {}
options = dict(equilibria_range=3.0)

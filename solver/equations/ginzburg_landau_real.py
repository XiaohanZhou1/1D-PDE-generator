"""Cubic Ginzburg-Landau equation with real coefficients, as a system for (u, v), without a wave.

    A_t = A_xx - A + |A|^2 A,      A = u + i v.

The equations commute with the rotation of (u, v).  Only travelling waves are in the scope of the
solver: every wave of the search has the frequency 0.  The pulse A = sqrt(2) sech(x) is one.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)

s = u ** 2 + v ** 2
equations = [sp.Eq(u.diff(t), u.diff(x, 2) - u + s * u),
             sp.Eq(v.diff(t), v.diff(x, 2) - v + s * v)]
parameters = {}
options = dict(equilibria_range=3.0)

"""Stationary pulse of u_t = u_xx - u + 2 u^3.

Known result: U = sech(z), c = 0.  The linearisation has the eigenvalues 3 and 0,
so the pulse is unstable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
z = sp.Symbol("z", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) - u + 2 * u ** 3)]
parameters = {}
# The guess is deliberately not the exact profile.
wave = dict(kind="pulse", rest={u: 0}, speed_guess=0.0, guess={u: 1.1 / sp.cosh(0.8 * z)})

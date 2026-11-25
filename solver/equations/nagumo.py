"""Bistable (Nagumo) equation.  Only the equation and its parameter: the solver searches for the waves."""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a = sp.Symbol("a", positive=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a))]
parameters = {a: 0.25}

"""Fisher-KPP equation.  Only the equation: the solver searches for the waves."""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u))]
parameters = {}

"""Bistable (Nagumo) front.

Known result: the front from u = 1 to u = 0 has speed c = (1 - 2a)/sqrt(2) and
is spectrally stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a = sp.Symbol("a", positive=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a))]
parameters = {a: 0.25}
wave = dict(kind="front", left={u: 1}, right={u: 0}, speed_guess=0.3, width=1.5)

"""Fisher-KPP front.

Known result: the state u = 0 ahead of the front is unstable, so the essential
spectrum reaches Re(lambda) = 1 in the unweighted space.  The label in L2 needs no wave.
In the weighted space exp(r z) ahead of the front the supremum is r^2 - c r + 1: fronts with
c > 2 are stable there for r between (c -+ sqrt(c^2 - 4)) / 2 (Sattinger 1976).  The fronts
form a one-parameter family, so the speed is given.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u))]
parameters = {}
wave = dict(kind="front", left={u: 1}, right={u: 0}, speed=2.5)

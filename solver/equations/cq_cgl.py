"""Cubic-quintic complex Ginzburg-Landau equation, without a wave (see problems/cq_cgl_pulse.py).

The equations commute with the rotation of (u, v).  The automatic search does not handle this
symmetry: the solver says so and stops.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
delta, beta, D, eps, mu, nu = sp.symbols("delta beta D eps mu nu", real=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)

s = u ** 2 + v ** 2
equations = [sp.Eq(u.diff(t), delta * u + beta * u.diff(x, 2) - D / 2 * v.diff(x, 2)
                   + (eps * u - v) * s + (mu * u - nu * v) * s ** 2),
             sp.Eq(v.diff(t), delta * v + beta * v.diff(x, 2) + D / 2 * u.diff(x, 2)
                   + (eps * v + u) * s + (mu * v + nu * u) * s ** 2)]
parameters = {delta: -0.1, beta: 0.08, D: 1.0, eps: 0.5, mu: -0.1, nu: -0.1}

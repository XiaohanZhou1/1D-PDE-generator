"""Burgers equation, without a wave (see problems/burgers_shock.py).

    u_t + (u^2 / 2)_x = u_xx.

Every constant is an equilibrium: the equation alone does not determine a wave.  The solver says so;
the end states must be given.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t) + (u ** 2 / 2).diff(x), u.diff(x, 2))]
parameters = {}

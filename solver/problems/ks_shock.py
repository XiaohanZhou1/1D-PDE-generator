"""Kuramoto-Sivashinsky equation: a shock between two constants.

    u_t = -u_xxxx - u_xx - (u^2 / 2)_x.

Conservation form: every constant is an equilibrium.  The constants are unstable at long
wavelengths: lambda(kappa) = kappa^2 - kappa^4 - i (u - c) kappa, with the supremum 1/4 at
kappa = 1 / sqrt(2).  So every wave between constants is unstable through the essential
spectrum; the label needs no wave.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), -u.diff(x, 4) - u.diff(x, 2) - (u ** 2 / 2).diff(x))]
parameters = {}
wave = dict(kind="front", left={u: 1}, right={u: -1}, width=2.0)

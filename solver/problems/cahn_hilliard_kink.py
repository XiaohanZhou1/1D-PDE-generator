"""Cahn-Hilliard equation: the kink.

    u_t = -(u_xx + u - u^3)_xx.

The right-hand side is a second derivative: every constant is an equilibrium, and the mass is
conserved.  The kink U = tanh(x / sqrt(2)) does not move.  Two integrations of the profile
equation give  U'' + U - U^3 = 0.
Linearisation: L = -d^2/dx^2 (d^2/dx^2 + 1 - 3 U^2).  The essential spectrum is
lambda = -kappa^2 (kappa^2 + 2): it touches the origin, and there is no transport, so no
exponential weight moves it away.
Known result: the kink is stable; perturbations decay algebraically (Bricmont, Kupiainen and
Taskinen 1999; Howard 2007).  L is similar to a symmetric non-positive operator: no eigenvalue
with a positive real part.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), -(u.diff(x, 2) + u - u ** 3).diff(x, 2))]
parameters = {}
wave = dict(kind="front", left={u: -1}, right={u: 1}, width=sp.sqrt(2))

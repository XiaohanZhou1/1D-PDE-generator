"""Undamped nonlinear wave equation (phi^4 model): the kink.

    u_tt = u_xx + u - u^3.

The kink U = tanh(x / sqrt(2)) does not move.  Linearisation:  lambda^2 p = A p,  A = d^2/dx^2 + 1 - 3 U^2
with the eigenvalues 0 and -3/2 and the essential spectrum (-inf, -2].  So all of the spectrum lies
on the imaginary axis: lambda = 0 (twice), +-i sqrt(3/2) (the internal mode), and |Im| >= sqrt(2).
Known result: the kink is stable in the neutral sense (no growth; no decay either).
Without damping the speed is a symmetry parameter (Lorentz boost): it is given (speed = 0).
No estimate bounds the search rectangle without damping: it is given in `options`.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t, 2), u.diff(x, 2) + u - u ** 3)]
parameters = {}
wave = dict(kind="front", left={u: -1}, right={u: 1}, speed=0.0, width=sp.sqrt(2))
options = dict(region=(4.0, 6.0))

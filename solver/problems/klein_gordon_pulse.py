"""Undamped nonlinear wave equation: the standing pulse (eigenvalue in closed form).

    u_tt = u_xx - u + 2 u^3.

There is no damping: the essential spectrum is the part |Im(lambda)| >= 1 of the imaginary axis.
The pulse U = sech(x) is the pulse of cubic_pulse.py.  Linearisation:  lambda^2 p = L0 p,  L0 with
the eigenvalues 3 and 0.  So lambda = +-sqrt(3): the pulse is unstable.
Without damping the speed is a symmetry parameter (Lorentz boost): it is given (speed = 0).
No estimate bounds the search rectangle without damping: it is given in `options`.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t, 2), u.diff(x, 2) - u + 2 * u ** 3)]
parameters = {}
wave = dict(kind="pulse", rest={u: 0}, amplitude={u: 1.1}, width=1.2, speed=0.0)
options = dict(region=(4.0, 6.0))

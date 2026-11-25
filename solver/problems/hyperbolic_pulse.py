"""Damped wave equation with a standing pulse: eigenvalue in closed form.

    tau u_tt + (1 + beta u^2) u_t = u_xx - u + 2 u^3.

Second order in time.  The pulse U = sech(x) does not move (c = 0), so it is the pulse of
cubic_pulse.py.  Linearisation:  tau lambda^2 p + lambda (1 + beta U^2) p = L0 p,
L0 = d^2/dx^2 - 1 + 6 sech^2(x) with the eigenvalues 3 and 0 and the essential spectrum (-inf, -1].
For beta = 0:  tau lambda^2 + lambda = e  for every e in the spectrum of L0.
  * e = 3: lambda = (-1 + sqrt(1 + 12 tau)) / (2 tau)   (tau = 1: 1.302776): unstable;
  * e = 0: lambda = 0 (translation) and lambda = -1 / tau;
  * essential spectrum: sup Re = -1 / (2 tau) for tau >= 1/4.
For beta > 0 the damping depends on u; the eigenvalue is smaller and has no closed form.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
tau, beta = sp.symbols("tau beta", nonnegative=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(tau * u.diff(t, 2) + (1 + beta * u ** 2) * u.diff(t), u.diff(x, 2) - u + 2 * u ** 3)]
parameters = {tau: 1.0, beta: 0.0}
wave = dict(kind="pulse", rest={u: 0}, amplitude={u: 1.1}, width=1.2, speed_guess=0.0)

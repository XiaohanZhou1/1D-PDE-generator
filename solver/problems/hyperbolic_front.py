"""Damped wave equation with a bistable nonlinearity (hyperbolic Nagumo equation).

    tau u_tt + u_t = u_xx + u (1 - u) (u - a).

Second order in time.  The profile equation is  (1 - c^2 tau) U'' + c U' + f(U) = 0.  With
z = sqrt(1 - c^2 tau) xi it becomes the profile equation of the Nagumo front with the speed
c0 = c / sqrt(1 - c^2 tau).  So (Hadeler 1988)
    c = c0 / sqrt(1 + c0^2 tau),      c0 = (1 - 2a) / sqrt(2).
The speed stays below the characteristic speed 1 / sqrt(tau).
Known result: the front is stable (Gallay and Raugel 1997).
Essential spectrum at an end state with f'(u) = -m:  tau mu^2 + mu + kappa^2 + m = 0 (mu = lambda - i c kappa);
sup Re = (-1 + sqrt(1 - 4 tau m)) / (2 tau) if 4 tau m < 1, else -1 / (2 tau).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, tau = sp.symbols("a tau", positive=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(tau * u.diff(t, 2) + u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a))]
parameters = {a: 0.25, tau: 2.0}
wave = dict(kind="front", left={u: 1}, right={u: 0}, speed_guess=0.3, width=1.5)

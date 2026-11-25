"""Nonlocal phase-transition equation without diffusion (Bates, Fife, Ren and Wang 1997).

    u_t = J * u - u + k u (1 - u) (u - a),      J(s) = exp(-s^2 / (2 sigma^2)) / (sigma sqrt(2 pi)).

There is no term u_xx.  The profile equation  c U' + J * U - U + f(U) = 0  is of first
order for c != 0 and has no derivative at all for c = 0.
Known results (for 1 - f'(u) > 0, which holds for k < 3.69 at a = 0.25):
  * there is one monotone front between 0 and 1; it is stable (Chen 1997);
  * c = 0 if and only if the integral of f over (0, 1) vanishes: a = 1/2;
  * essential spectrum:  lambda(kappa) = i c kappa - 1 + f'(u) + Jhat(kappa),  sup Re = f'(u):
    -k a at u = 0 and -k (1 - a) at u = 1.
"""
import sympy as sp

x, t, y = sp.symbols("x t y", real=True)
a, k, sigma = sp.symbols("a k sigma", positive=True)
u = sp.Function("u")

J = sp.exp(-(x - y) ** 2 / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))
equations = [sp.Eq(u(x, t).diff(t), sp.Integral(J * u(y, t), (y, -sp.oo, sp.oo)) - u(x, t)
                   + k * u(x, t) * (1 - u(x, t)) * (u(x, t) - a))]
parameters = {a: 0.25, k: 1.0, sigma: 1.0}
wave = dict(kind="front", left={u(x, t): 1}, right={u(x, t): 0}, speed_guess=0.2, width=1.5)

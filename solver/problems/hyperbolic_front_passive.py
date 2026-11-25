"""A moving front of a damped wave system with a passive component: eigenvalues in closed form.

    tau u_tt + u_t = u_xx + u (1 - u) (u - a),
    tau v_tt + v_t = v_xx - v + sigma u (1 - u) v.

Wave: u = the front of hyperbolic_front.py, v = 0, c = c0 / sqrt(1 + c0^2 tau), c0 = (1 - 2a)/sqrt(2).
With z = sqrt(1 - c^2 tau) xi the v-block of the linearisation becomes
    q'' + c0 (1 + 2 tau lambda) q' + (sigma U0 (1 - U0) - 1) q = (tau lambda^2 + lambda) q,
U0 the Nagumo front.  The operator on the left with a drift kappa has the eigenvalues
mu_n + (c0^2 - kappa^2) / 4, where mu_n = (s - n)^2 / 8 - 1 - c0^2 / 4, s (s + 1) = 2 sigma, are those
of the problem of first order in time (cross_diffusion_front.py).  So
    (1 + c0^2 tau) (tau lambda^2 + lambda) = mu_n.
sigma = 15, tau = 1: lambda = 0.952966, 0.554093, 0.077350: unstable.
sigma = 3: all mu_n < 0: stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, sigma, tau = sp.symbols("a sigma tau", positive=True)
u = sp.Function("u")(x, t)
v = sp.Function("v")(x, t)
unknowns = [u, v]

equations = [sp.Eq(tau * u.diff(t, 2) + u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a)),
             sp.Eq(tau * v.diff(t, 2) + v.diff(t), v.diff(x, 2) - v + sigma * u * (1 - u) * v)]
parameters = {a: 0.25, sigma: 15.0, tau: 1.0}
wave = dict(kind="front", left={u: 1, v: 0}, right={u: 0, v: 0}, speed_guess=0.3, width=1.5)

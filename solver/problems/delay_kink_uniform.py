"""A standing kink with a passive component that has a distributed delay on a finite interval.

    u_t = u_xx + u - u^3,
    v_t = v_xx - v + sigma (1 - u^2) * integral over s in (0, tau) of  v(x, t - s) / tau.

Wave: u = tanh(z / sqrt(2)), v = 0, c = 0.  As in delay_kink_gamma.py:

    lambda + 1 = (s - n)^2 / 2,     s (s + 1) = 2 sigma (1 - exp(-lambda tau)) / (lambda tau).

For sigma = 6, tau = 2 there are real positive roots: unstable.  For sigma = 0.5: stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
s = sp.Symbol("s", positive=True)
sigma, tau = sp.symbols("sigma tau", positive=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) - u(x, t) ** 3),
             sp.Eq(v(x, t).diff(t), v(x, t).diff(x, 2) - v(x, t)
                   + sigma * (1 - u(x, t) ** 2) * sp.Integral(v(x, t - s) / tau, (s, 0, tau)))]
parameters = {sigma: 6.0, tau: 2.0}
wave = dict(kind="front", left={u(x, t): -1, v(x, t): 0}, right={u(x, t): 1, v(x, t): 0},
            speed_guess=0.0, width=1.2)

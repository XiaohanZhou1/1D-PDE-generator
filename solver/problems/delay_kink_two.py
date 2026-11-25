"""A standing kink with a passive component that has two discrete delays.

    u_t = u_xx + u - u^3,
    v_t = v_xx - v + sigma (1 - u^2) (omega v(x, t - tau1) + (1 - omega) v(x, t - tau2)).

Wave: u = tanh(z / sqrt(2)), v = 0, c = 0.  As in delay_kink_passive.py the v-block is a
Poeschl-Teller operator.  Its depth carries the factor E(lambda):

    lambda + 1 = (s - n)^2 / 2,     s (s + 1) = 2 sigma E(lambda),     n = 0, 1, ...,   Re(s - n) > 0,
    E(lambda) = omega exp(-lambda tau1) + (1 - omega) exp(-lambda tau2).

For sigma = 6, omega = 0.5, tau1 = 0.5, tau2 = 1.5 there are real positive roots: unstable.
For sigma = 0.5 there is no root to the right of the essential spectrum: stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
sigma, omega, tau1, tau2 = sp.symbols("sigma omega tau1 tau2", real=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) - u(x, t) ** 3),
             sp.Eq(v(x, t).diff(t), v(x, t).diff(x, 2) - v(x, t)
                   + sigma * (1 - u(x, t) ** 2) * (omega * v(x, t - tau1) + (1 - omega) * v(x, t - tau2)))]
parameters = {sigma: 6.0, omega: 0.5, tau1: 0.5, tau2: 1.5}
wave = dict(kind="front", left={u(x, t): -1, v(x, t): 0}, right={u(x, t): 1, v(x, t): 0},
            speed_guess=0.0, width=1.2)

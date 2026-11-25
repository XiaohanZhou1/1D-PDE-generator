"""A kink of a damped wave equation with a passive component of first order in time.

    tau u_tt + u_t = u_xx + u - u^3,
    v_t = v_xx - v + sigma (1 - u^2) v.

Wave: u = tanh(x / sqrt(2)), v = 0, c = 0.  One component is of second order in time, the other
of first order.  The v-block is the Poeschl-Teller operator of delay_kink_passive.py with tau = 0:
    lambda = (s - n)^2 / 2 - 1,   s (s + 1) = 2 sigma        (sigma = 6: 3.5, 1, -0.5).
The energy estimate for second-order time derivatives needs u_tt in every component, so the
search rectangle is given here.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
tau, sigma = sp.symbols("tau sigma", positive=True)
u = sp.Function("u")(x, t)
v = sp.Function("v")(x, t)
unknowns = [u, v]

equations = [sp.Eq(tau * u.diff(t, 2) + u.diff(t), u.diff(x, 2) + u - u ** 3),
             sp.Eq(v.diff(t), v.diff(x, 2) - v + sigma * (1 - u ** 2) * v)]
parameters = {tau: 1.0, sigma: 6.0}
wave = dict(kind="front", left={u: -1, v: 0}, right={u: 1, v: 0}, speed_guess=0.0, width=1.2)
options = dict(region=(6.0, 6.0))

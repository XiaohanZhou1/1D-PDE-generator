"""A standing kink with a delayed passive component: eigenvalues from a scalar equation.

    u_t = u_xx + u - u^3,
    v_t = v_xx - v + sigma (1 - u^2) v(x, t - tau).

Wave: u = tanh(z / sqrt(2)), v = 0, c = 0.  The linearisation is block triangular.
The v-block is a Poeschl-Teller operator whose depth carries the factor exp(-lambda tau):

    lambda + 1 = (s - n)^2 / 2,     s (s + 1) = 2 sigma exp(-lambda tau),     n = 0, 1, ...

with Re(s - n) > 0.  For sigma = 6, tau = 1 there are two real positive roots
(0.786749 for n = 0 and 0.257740 for n = 1): the kink is unstable.  For sigma = 0.5 there
is no root to the right of the essential spectrum: stable.  At tau = 0 and sigma = 6:
s = 3, roots 3.5, 1, -0.5.  The equation also has complex roots; for sigma = -6, tau = 2
(delayed negative feedback) a complex pair has positive real part: oscillatory instability.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
sigma, tau = sp.symbols("sigma tau", real=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) - u(x, t) ** 3),
             sp.Eq(v(x, t).diff(t), v(x, t).diff(x, 2) - v(x, t) + sigma * (1 - u(x, t) ** 2) * v(x, t - tau))]
parameters = {sigma: 6.0, tau: 1.0}
wave = dict(kind="front", left={u(x, t): -1, v(x, t): 0}, right={u(x, t): 1, v(x, t): 0},
            speed_guess=0.0, width=1.2)

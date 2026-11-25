"""A travelling front with a delayed passive component.

    u_t = u_xx + u (1 - u) (u - a),
    v_t = v_xx - v + sigma u (1 - u) v(x, t - tau).

Wave: u = the Nagumo front, v = 0, c = (1 - 2a)/sqrt(2).  The front moves, so the delayed
term is evaluated at z + c tau: the eigenvalue problem of the v-block,

    lambda q(z) = q'' + c q' - q + sigma U (1 - U) exp(-lambda tau) q(z + c tau),

contains a shift.  It has no closed form.  At tau = 0 its eigenvalues are
(s - n)^2 / 8 - 1 - c^2 / 4 with s (s + 1) = 2 sigma  (sigma = 15: 2.09375, 0.96875, 0.09375).
The reference values for tau > 0 come from an independent discretisation
(Chebyshev points in the delay, tests/test_core.py).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, sigma, tau = sp.symbols("a sigma tau", real=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)),
             sp.Eq(v(x, t).diff(t), v(x, t).diff(x, 2) - v(x, t)
                   + sigma * u(x, t) * (1 - u(x, t)) * v(x, t - tau))]
parameters = {a: 0.25, sigma: 15.0, tau: 1.0}
wave = dict(kind="front", left={u(x, t): 1, v(x, t): 0}, right={u(x, t): 0, v(x, t): 0},
            speed_guess=0.3, width=1.5)

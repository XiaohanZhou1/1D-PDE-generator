"""A standing kink with a passive component that has a distributed delay (Gamma kernel).

    u_t = u_xx + u - u^3,
    v_t = v_xx - v + sigma (1 - u^2) * integral over s in (0, oo) of  g(s) v(x, t - s),
    g(s) = s^(p - 1) exp(-s / T) / (T^p Gamma(p))        (mean delay p T).

Wave: u = tanh(z / sqrt(2)), v = 0, c = 0.  The v-block is a Poeschl-Teller operator.  Its
depth carries the Laplace transform of the kernel:

    lambda + 1 = (s - n)^2 / 2,     s (s + 1) = 2 sigma (1 + T lambda)^(-p),     n = 0, 1, ...,   Re(s - n) > 0.

For sigma = 6 there are real positive roots: unstable.  For sigma = 0.5: stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
s = sp.Symbol("s", positive=True)
sigma, T, p = sp.symbols("sigma T p", positive=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

kernel = s ** (p - 1) * sp.exp(-s / T) / (T ** p * sp.gamma(p))
equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) - u(x, t) ** 3),
             sp.Eq(v(x, t).diff(t), v(x, t).diff(x, 2) - v(x, t)
                   + sigma * (1 - u(x, t) ** 2) * sp.Integral(kernel * v(x, t - s), (s, 0, sp.oo)))]
parameters = {sigma: 6.0, T: 1.0, p: 1.0}
wave = dict(kind="front", left={u(x, t): -1, v(x, t): 0}, right={u(x, t): 1, v(x, t): 0},
            speed_guess=0.0, width=1.2)

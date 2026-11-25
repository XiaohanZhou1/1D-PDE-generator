"""A front that a Gaussian convolution destabilises in its core.

    u_t = u_xx + u (1 - u) (u - a) + m ( int K(x - y) g(u(y, t)) dy - B(u) ),
    K = Gaussian of width sigma,   B(u) = u^2 (1 - u)^2,   g(u) = B(u) (10 u (1 - u) - 1).

This is the equation of kernel_core_front.py with the exponential kernel replaced by a
Gaussian one, which has no rational Fourier transform: the auxiliary-unknown method of
nonlocal_core_front.py does not apply.  B and g vanish to second order at u = 0 and
u = 1: the end states and the essential spectrum (sup Re = -a) do not depend on m.
There is no closed form.  The reference values come from an independent discretisation
(tests/test_core.py):  m = 5: stable;  m = -9: one real eigenvalue 0.3696;
m = -15: a pair 0.1706 +- 0.5608 i.
"""
import sympy as sp

x, t, y = sp.symbols("x t y", real=True)
a, m, sigma = sp.symbols("a m sigma", real=True)
u = sp.Function("u")

B = lambda v: v ** 2 * (1 - v) ** 2
g = lambda v: B(v) * (10 * v * (1 - v) - 1)
K = sp.exp(-(x - y) ** 2 / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))
equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)
                   + m * (sp.Integral(K * g(u(y, t)), (y, -sp.oo, sp.oo)) - B(u(x, t))))]
parameters = {a: 0.25, m: -9.0, sigma: 1.4}
z = sp.Symbol("z", real=True)
wave = dict(kind="front", left={u(x, t): 1}, right={u(x, t): 0}, speed_guess=0.3,
            guess={u(x, t): 1 / (1 + sp.exp(z / 1.5))})

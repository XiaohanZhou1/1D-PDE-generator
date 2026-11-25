"""The equation of nonlocal_core_front.py, written with the convolution integral itself.

    u_t = u_xx + u (1 - u) (u - a) + m ( int K(x - y) g(u(y, t)) dy - B(u) ),
    K(s) = exp(-|s|) / 2,   B(u) = u^2 (1 - u)^2,   g(u) = B(u) (10 u (1 - u) - 1).

nonlocal_core_front.py removes this kernel with an auxiliary unknown and uses the Evans
engine.  Here the integral stays, and the band-determinant engine is used.  The two files
describe the same equation and the same wave: the results must agree.
Exact wave: u = 1/(1 + exp(z/sqrt(2))), c = (1 - 2a)/sqrt(2).
m = 5: stable.   m = -20: eigenvalues 1.0476 and 0.4937.   m = -50: 3.5237 +- 0.8525 i, 1.518, 0.398.
"""
import sympy as sp

x, t, y = sp.symbols("x t y", real=True)
a, m = sp.symbols("a m", real=True)
u = sp.Function("u")

B = lambda v: v ** 2 * (1 - v) ** 2
g = lambda v: B(v) * (10 * v * (1 - v) - 1)
K = sp.exp(-sp.Abs(x - y)) / 2
equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)
                   + m * (sp.Integral(K * g(u(y, t)), (y, -sp.oo, sp.oo)) - B(u(x, t))))]
parameters = {a: 0.25, m: -20.0}
# The same guess as in nonlocal_core_front.py: it selects the exact front (its width is 6 percent wrong).
z = sp.Symbol("z", real=True)
wave = dict(kind="front", left={u(x, t): 1}, right={u(x, t): 0}, speed_guess=0.3,
            guess={u(x, t): 1 / (1 + sp.exp(z / 1.5))})

"""Bistable equation with a Gaussian convolution (a kernel without a rational Fourier transform).

    u_t = u_xx + u (1 - u) (u - a) + mu ( int K(x - y) u(y, t) dy - u ),
    K(s) = exp(-s^2 / (2 sigma^2)) / (sigma sqrt(2 pi)).

For mu > 0 the equation has a comparison principle.
Known result: the front is stable (Bates, Fife, Ren and Wang 1997; Chen 1997).
For small sigma:  K * u - u = (sigma^2 / 2) u_xx + O(sigma^4), so the speed is
c = sqrt(1 + mu sigma^2 / 2) (1 - 2a)/sqrt(2) + O(sigma^4).
Essential spectrum at u = 0:  lambda(kappa) = -kappa^2 - a + mu (exp(-sigma^2 kappa^2 / 2) - 1).
For mu < 0 and |mu| sigma^2 > 2 its supremum is |mu| - a - (2 / sigma^2)(1 + ln(|mu| sigma^2 / 2)),
at a wavenumber different from zero (mu = -6, sigma = 1: +1.5528, unstable).
"""
import sympy as sp

x, t, y = sp.symbols("x t y", real=True)
a, mu, sigma = sp.symbols("a mu sigma", real=True)
u = sp.Function("u")

K = sp.exp(-(x - y) ** 2 / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))
equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)
                   + mu * (sp.Integral(K * u(y, t), (y, -sp.oo, sp.oo)) - u(x, t)))]
parameters = {a: 0.25, mu: 1.0, sigma: 1.0}
wave = dict(kind="front", left={u(x, t): 1}, right={u(x, t): 0}, speed_guess=0.3, width=1.5)

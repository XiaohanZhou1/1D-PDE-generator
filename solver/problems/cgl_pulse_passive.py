"""Hocking-Stewartson pulse with a passive component: eigenvalues in closed form.

    A_t = mu A + (1 + i alpha) A_xx + (1 + i beta) |A|^2 A,      A = u + i v,
    w_t = w_xx - m w + sigma |A|^2 w.

The rotation acts on (u, v) only.  On the pulse of cgl_pulse.py, |A|^2 = R^2 sech^2(kappa x) and w = 0.
The linearisation is block triangular; the w-block is a Poeschl-Teller operator with the eigenvalues
    lambda_n = -m + kappa^2 (s - n)^2,      s (s + 1) = sigma R^2 / kappa^2,      n = 0, 1, ... < s.
The other eigenvalues are those of the pulse itself (one real positive eigenvalue).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
mu, alpha, beta, m, sigma = sp.symbols("mu alpha beta m sigma", real=True)
u, v, w = sp.Function("u")(x, t), sp.Function("v")(x, t), sp.Function("w")(x, t)
z = sp.Symbol("z", real=True)

s = u ** 2 + v ** 2
equations = [sp.Eq(u.diff(t), mu * u + u.diff(x, 2) - alpha * v.diff(x, 2) + (u - beta * v) * s),
             sp.Eq(v.diff(t), mu * v + v.diff(x, 2) + alpha * u.diff(x, 2) + (v + beta * u) * s),
             sp.Eq(w.diff(t), w.diff(x, 2) - m * w + sigma * s * w)]
parameters = {mu: -1.0, alpha: 0.3, beta: 1.0, m: 1.0, sigma: 3.8}
wave = dict(kind="pulse", rest={u: 0, v: 0, w: 0}, speed=0.0, frequency_guess=1.2,
            guess={u: 1.4 / sp.cosh(1.1 * z), v: 0, w: 0})

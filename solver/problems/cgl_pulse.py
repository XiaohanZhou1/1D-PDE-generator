"""Cubic complex Ginzburg-Landau equation: the Hocking-Stewartson pulse (rotation symmetry, dissipative).

    A_t = mu A + (1 + i alpha) A_xx + (1 + i beta) |A|^2 A,      A = u + i v,      mu < 0.

The equations commute with the rotation of (u, v).  Exact standing pulse (Hocking and Stewartson 1972):
    A = R sech(kappa x)^(1 + i delta) exp(i w t),
    with  q = (1 + i beta) / (1 + i alpha),  Im(q) delta^2 + 3 Re(q) delta - 2 Im(q) = 0,
    kappa^2 = -mu / (1 - delta^2 - 2 alpha delta),  w = kappa^2 (alpha (1 - delta^2) + 2 delta),
    R^2 = kappa^2 (2 - delta^2) / Re(q).
The pulse is isolated: the frequency is a nonlinear eigenvalue.  It is an unknown here (frequency_guess).
The rest state is stable (mu < 0); the cubic term does not saturate.  Expected: the pulse is the
threshold between decay and blow-up, with one real positive eigenvalue (as for alpha = beta = 0,
where the eigenvalue is -3 mu).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
mu, alpha, beta = sp.symbols("mu alpha beta", real=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)
z = sp.Symbol("z", real=True)

s = u ** 2 + v ** 2
equations = [sp.Eq(u.diff(t), mu * u + u.diff(x, 2) - alpha * v.diff(x, 2) + (u - beta * v) * s),
             sp.Eq(v.diff(t), mu * v + v.diff(x, 2) + alpha * u.diff(x, 2) + (v + beta * u) * s)]
parameters = {mu: -1.0, alpha: 0.3, beta: 1.0}
wave = dict(kind="pulse", rest={u: 0, v: 0}, speed=0.0, frequency_guess=1.2,
            guess={u: 1.4 / sp.cosh(1.1 * z), v: 0})

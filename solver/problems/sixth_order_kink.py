"""A sixth-order equation with an exact kink and a spectrum in closed form.

    N(u) = u_xx + u - u^3,      D w = w_xx + (1 - 3 u^2) w,
    u_t = D (D N(u)) + beta N(u).

Every solution of N(u) = 0 is a steady state: the kink U = tanh(x / sqrt(2)) is a standing wave.
Linearisation at the kink:  L = A^3 + beta A,  A = d^2/dx^2 + 1 - 3 U^2  (see fourth_order_kink.py).
A has the eigenvalues 0 and -3/2 and the essential spectrum (-inf, -2].  So L has
  * the eigenvalues 0 (translation) and -27/8 - 3 beta / 2,
  * the essential spectrum up to -8 - 2 beta   (for beta >= -4).
beta = 0:  stable.
beta = -3: unstable (eigenvalue +1.125, essential spectrum up to -2).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
beta = sp.Symbol("beta", real=True)
u = sp.Function("u")(x, t)

N = u.diff(x, 2) + u - u ** 3


def D(w):
    return w.diff(x, 2) + (1 - 3 * u ** 2) * w


equations = [sp.Eq(u.diff(t), D(D(N)) + beta * N)]
parameters = {beta: 0.0}
wave = dict(kind="front", left={u: -1}, right={u: 1}, speed_guess=0.0, width=1.2 * sp.sqrt(2))

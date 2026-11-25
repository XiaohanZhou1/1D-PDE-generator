"""A fourth-order equation with an exact kink and a spectrum in closed form.

    N(u) = u_xx + u - u^3,
    u_t = -(d^2/dx^2 + 1 - 3 u^2) N(u) + alpha N(u).

For alpha = 0 this is the gradient flow of (1/2) ||N(u)||^2.  Every solution of N(u) = 0 is a
steady state: the Allen-Cahn kink U = tanh(x / sqrt(2)) is a standing wave (c = 0).
Linearisation at the kink (the terms with N(U) vanish):
    L = -A^2 + alpha A,      A = d^2/dx^2 + 1 - 3 U^2.
A has the eigenvalues 0 and -3/2 and the essential spectrum (-inf, -2].  So L has
  * the eigenvalues 0 (translation) and -9/4 - 3 alpha / 2,
  * the essential spectrum up to -4 - 2 alpha   (for alpha > -4).
alpha = 0:    stable (eigenvalue -2.25, essential spectrum up to -4).
alpha = -1.8: unstable (eigenvalue +0.45, essential spectrum up to -0.4).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
alpha = sp.Symbol("alpha", real=True)
u = sp.Function("u")(x, t)

N = u.diff(x, 2) + u - u ** 3
equations = [sp.Eq(u.diff(t), -(N.diff(x, 2) + (1 - 3 * u ** 2) * N) + alpha * N)]
parameters = {alpha: 0.0}
# The guess is not the exact kink: its width is 20 percent too large.
wave = dict(kind="front", left={u: -1}, right={u: 1}, speed_guess=0.0, width=1.2 * sp.sqrt(2))

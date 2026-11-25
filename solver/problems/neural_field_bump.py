"""Standing pulses (bumps) of a neural field with lateral inhibition (Amari 1977).

    u_t = -u + integral of K(x - y) S(u(y, t)) dy,
    K(s) = A g(s; 1) - B g(s; 2),   g(s; w) = exp(-s^2 / (2 w^2)) / (w sqrt(2 pi)),
    S(u) = 1 / (1 + exp(-beta (u - theta))) - 1 / (1 + exp(beta theta))        (S(0) = 0: u = 0 is a rest state).

There is no diffusion term; the bumps do not move (c = 0), so the profile equation is an
integral equation.  The linearisation  lambda p = -p + K * (S'(U) p)  is similar to a
symmetric operator: all eigenvalues are real.

Known result for a step-like S (Amari 1977): with W(x) = integral of K over (0, x), a bump of
width b exists if W(b) = theta, and it is stable if and only if K(b) < 0.  Its eigenvalues are
0 (translation) and 2 K(b) / (K(0) - K(b)).  For A = 2, B = 1, theta = 0.55:
  * narrow bump, b = 1.23: K(b) > 0, eigenvalue about +1.1: unstable;
  * wide bump,   b = 3.27: K(b) < 0, eigenvalue about -0.15: stable.
The parameter `bump` is the width in the guess; it selects the bump.  The guess is the bump of the step-like S.
"""
import sympy as sp

x, t, y = sp.symbols("x t y", real=True)
A, B, beta, theta, bump = sp.symbols("A B beta theta bump", positive=True)
u = sp.Function("u")


def gauss(s, w):
    return sp.exp(-s ** 2 / (2 * w ** 2)) / (w * sp.sqrt(2 * sp.pi))


def S(v):
    return 1 / (1 + sp.exp(-beta * (v - theta))) - 1 / (1 + sp.exp(beta * theta))


K = A * gauss(x - y, 1) - B * gauss(x - y, 2)
equations = [sp.Eq(u(x, t).diff(t), -u(x, t) + sp.Integral(K * S(u(y, t)), (y, -sp.oo, sp.oo)))]
parameters = {A: 2.0, B: 1.0, beta: 8.0, theta: 0.55, bump: 1.23}

z = sp.Symbol("z", real=True)


def W(s):                                              # integral of K over (0, s)
    return A * sp.erf(s / sp.sqrt(2)) / 2 - B * sp.erf(s / (2 * sp.sqrt(2))) / 2


wave = dict(kind="pulse", rest={u(x, t): 0}, speed_guess=0.0, width=1.0,
            guess={u(x, t): W(z + bump / 2) - W(z - bump / 2)})

"""Nonlinear Schroedinger equation: the solitary wave (rotation symmetry, Hamiltonian).

    i psi_t + psi_xx + N(|psi|^2) psi = 0,      N(s) = (1 - q) s + q s^3,      psi = u + i v:
    u_t = -v_xx - N(u^2 + v^2) v,
    v_t =  u_xx + N(u^2 + v^2) u.

q = 0 is the cubic equation (power sigma = 1), q = 1 the septic one (sigma = 3).
The equations commute with the rotation of (u, v).  Waves psi = exp(i w t) Phi(x - c t):
    c = 0:   Phi = ((sigma + 1) omega)^(1 / (2 sigma)) sech^(1 / sigma)(sigma sqrt(omega) z),  w = omega;
    c != 0:  Phi(z) exp(i c z / 2),  w = omega + c^2 / 4  (Galilean boost).
The waves form a family in (c, w): both are given in `wave`.
Known result (Vakhitov-Kolokolov criterion; Weinstein 1986): no growing mode for sigma < 2; one real
positive eigenvalue for sigma > 2.
The essential spectrum is the part |Im(lambda)| >= w of the imaginary axis: the label is
STABLE (neutral) at best.  No estimate bounds the search rectangle: it is given in `options`.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
omega, c, q = sp.symbols("omega c q", real=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)
z = sp.Symbol("z", real=True)

s = u ** 2 + v ** 2
N = (1 - q) * s + q * s ** 3
equations = [sp.Eq(u.diff(t), -v.diff(x, 2) - N * v),
             sp.Eq(v.diff(t), u.diff(x, 2) + N * u)]
parameters = {omega: 1.0, c: 0.0, q: 0.0}


def profile(sigma):
    return ((sigma + 1) * omega) ** sp.Rational(1, 2 * sigma) * sp.cosh(sigma * sp.sqrt(omega) * z / 1.1) ** sp.Rational(-1, sigma)


phi = 0.9 * ((1 - q) * profile(1) + q * profile(3))
wave = dict(kind="pulse", rest={u: 0, v: 0}, speed=c, frequency=omega + c ** 2 / 4,
            guess={u: phi * sp.cos(c * z / 2), v: phi * sp.sin(c * z / 2)})
options = dict(region=(4.0, 6.0))

"""Family A1 of Version_1 (four components), written in the frame where the pulse is at rest.

Known result (Version_1, q = 1, a = 2): eigenvalues 4.469956 +- 1.108045 i;
three zeros in Re(lambda) > -0.04 including the translation zero.
Exact pulse: (s, s', s^2, s^3) with s = sech(z).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
z = sp.Symbol("z", real=True)
q, a = sp.symbols("q a", real=True)
u, y, v, w = [sp.Function(name)(x, t) for name in "uyvw"]
unknowns = [u, y, v, w]


def P(j, s):
    return j * j * s ** j - j * (j + 1) * s ** (j + 2)


R, S = v - u ** 2, w - u ** 3
tau = 6 * u ** 2 - a

equations = [
    sp.Eq(u.diff(t), u.diff(x, 2) - q / 2 * u.diff(x) - P(1, u) + q * y / 2 + q * u * (3 * R + 2 * S)),
    sp.Eq(y.diff(t), 3 * y.diff(x, 2) + (6 * u ** 3).diff(x) - 3 * y),
    sp.Eq(v.diff(t), v.diff(x, 2) + q / 2 * v.diff(x) - P(2, u) - q * u * y + tau * R - 2 * S),
    sp.Eq(w.diff(t), (1 + q / 2) * w.diff(x, 2) - q * w.diff(x) - (1 + q / 2) * P(3, u)
          + 3 * q * u ** 2 * y + (tau + q * u) * S + 2 * R),
]
parameters = {q: 1.0, a: 2.0}

s = 1 / sp.cosh(z)
# The guess is the exact pulse with a 3 percent amplitude error in u.
wave = dict(kind="pulse", rest={u: 0, y: 0, v: 0, w: 0}, speed_guess=0.0,
            guess={u: 1.03 * s, y: sp.diff(s, z), v: s ** 2, w: s ** 3})

"""Generalised Korteweg-de Vries equation: the solitary wave.

    u_t + u^p u_x + u_xxx = 0.

Dispersive and in conservation form.  The solitary wave with the speed c is
    U = A sech^(2/p)(p sqrt(c) z / 2),      A^p = (p + 1)(p + 2) c / 2.
The essential spectrum is the imaginary axis: no exponential decay is possible, and "stable"
can only mean "no spectrum with a positive real part" (neutral).
Known result (Pego and Weinstein 1992): for p < 4 there is no such spectrum; for p > 4 there is
one real positive eigenvalue: the solitary wave is unstable.
No estimate bounds the search rectangle of a dispersive equation: it is given in `options`.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
p, c = sp.symbols("p c", positive=True)
u = sp.Function("u")(x, t)
z = sp.Symbol("z", real=True)

equations = [sp.Eq(u.diff(t) + u ** p * u.diff(x) + u.diff(x, 3), 0)]
parameters = {p: 2, c: 1.0}
amplitude = ((p + 1) * (p + 2) * c / 2) ** (1 / p)
wave = dict(kind="pulse", rest={u: 0}, speed=c, guess={u: 0.9 * amplitude * sp.cosh(p * sp.sqrt(c) * z / 2.2) ** (-2 / p)})
options = dict(region=(3.0, 8.0))

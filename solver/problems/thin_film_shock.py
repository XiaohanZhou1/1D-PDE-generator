"""Thin-film equation (driven film with Marangoni stress and gravity): a compressive shock.

    h_t + (h^2 - h^3)_x = -(h^3 h_xxx)_x.

A scalar conservation law of order four with the non-convex flux f(h) = h^2 - h^3.  The wave connects
the film thickness hinf behind the front to the precursor thickness b ahead of it; the jump condition
gives the speed c = (f(hinf) - f(b)) / (hinf - b).  The profile has a capillary ridge behind the front.
Known results (Bertozzi, Muench, Shearer and Zumbrun 2001): for b = 0.1 there is one compressive wave
for small hinf, and it is stable.  In a window of hinf (about 0.3323 to 0.3413) there are several
compressive waves with ridges of different height; their stability alternates.
The parameters A and w shape the guess only: A is the height of the ridge of the guess.
At hinf = 0.336: A = 0.08 gives the wave with the low ridge (maximum 0.439), A = 0.3 the one with
the high ridge (maximum 0.576).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
hinf, b = sp.symbols("hinf b", positive=True)
A, w = sp.symbols("A w", real=True)
h = sp.Function("h")(x, t)
z = sp.Symbol("z", real=True)

equations = [sp.Eq(h.diff(t) + (h ** 2 - h ** 3).diff(x), -(h ** 3 * h.diff(x, 3)).diff(x))]
parameters = {hinf: 0.336, b: 0.1, A: 0.08, w: 1.0}
wave = dict(kind="front", left={h: hinf}, right={h: b},
            guess={h: b + (hinf - b) * (1 - sp.tanh(z / w)) / 2 + A / sp.cosh((z + 2 * w) / (2 * w)) ** 2})

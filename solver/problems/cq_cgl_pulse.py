"""Cubic-quintic complex Ginzburg-Landau equation: the dissipative soliton (rotation symmetry).

    psi_t = delta psi + (beta + i D / 2) psi_xx + (eps + i) |psi|^2 psi + (mu + i nu) |psi|^4 psi,      psi = u + i v.

delta < 0: linear loss (the rest state is stable); beta > 0: spectral filtering; eps > 0: nonlinear gain;
mu < 0: saturation of the gain.  The standing pulse is isolated; its frequency is an unknown.
For one set of parameters there are two pulses: a low one (the threshold between decay and growth,
unstable) and a high one.  The guess selects the pulse: a0 = 2, w0 = 2 gives the high one,
a0 = 1, w0 = 0.5 the low one.
Literature (Soto-Crespo, Akhmediev and Ankiewicz 2000): with delta = -0.1, beta = 0.08, mu = -0.1,
nu = -0.1, D = 1 the high stationary soliton is stable for moderate gain and gives way to a pulsating
soliton when eps grows (reported pulsating at eps = 0.66).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
delta, beta, D, eps, mu, nu = sp.symbols("delta beta D eps mu nu", real=True)
a0, w0 = sp.symbols("a0 w0", positive=True)           # amplitude and frequency of the guess only
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)
z = sp.Symbol("z", real=True)

s = u ** 2 + v ** 2
equations = [sp.Eq(u.diff(t), delta * u + beta * u.diff(x, 2) - D / 2 * v.diff(x, 2)
                   + (eps * u - v) * s + (mu * u - nu * v) * s ** 2),
             sp.Eq(v.diff(t), delta * v + beta * v.diff(x, 2) + D / 2 * u.diff(x, 2)
                   + (eps * v + u) * s + (mu * v + nu * u) * s ** 2)]
parameters = {delta: -0.1, beta: 0.08, D: 1.0, eps: 0.5, mu: -0.1, nu: -0.1, a0: 2.0, w0: 2.0}
wave = dict(kind="pulse", rest={u: 0, v: 0}, speed=0.0, frequency_guess=w0,
            guess={u: a0 / sp.cosh(a0 * z), v: 0})

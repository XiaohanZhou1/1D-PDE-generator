"""Extended Fisher-Kolmogorov equation: the kink.

    u_t = -gamma u_xxxx + u_xx + u - u^3.

For gamma = 0 this is the Allen-Cahn equation with the kink tanh(x / sqrt(2)): eigenvalues 0
and -3/2, essential spectrum up to -2.  For 0 < gamma <= 1/8 there is one monotone kink
between -1 and 1 (Peletier and Troy 1995); it is stable.  For gamma > 1/8 the tails oscillate.
Essential spectrum for every gamma:  -gamma kappa^4 - kappa^2 - 2,  supremum -2.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
gamma = sp.Symbol("gamma", positive=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), -gamma * u.diff(x, 4) + u.diff(x, 2) + u - u ** 3)]
parameters = {gamma: 0.1}
wave = dict(kind="front", left={u: -1}, right={u: 1}, speed_guess=0.0, width=sp.sqrt(2))

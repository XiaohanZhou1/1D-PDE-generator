"""A front of a nonlocal bistable equation with an exact profile.

    u_t = u_xx + u (1 - u) (u - a) + mu (K * g(u) - u),      K(x) = exp(-|x|) / 2,

with g(u) = u - u (1 - u) (1 - 2u) / 2.  The convolution w = K * g(u) is the bounded
solution of  w_xx - w + g(u) = 0.  It is written here as a second unknown with a
constraint equation (no time derivative).

Exact wave for every mu: u = w = 1/(1 + exp(z/sqrt(2))), c = (1 - 2a)/sqrt(2).
(g is chosen such that K * g(U) = U on this profile.)
Essential spectrum at u = 0 in closed form: with f'(0) = -a and g'(0) = 1/2,
    lambda(kappa) = -kappa^2 - a - mu + mu / (2 (1 + kappa^2)).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, mu = sp.symbols("a mu", real=True)
u, w = sp.Function("u")(x, t), sp.Function("w")(x, t)
unknowns = [u, w]

g = u - u * (1 - u) * (1 - 2 * u) / 2
equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a) + mu * (w - u)),
             sp.Eq(0, w.diff(x, 2) - w + g)]
parameters = {a: 0.25, mu: 1.0}
wave = dict(kind="front", left={u: 1, w: 1}, right={u: 0, w: 0}, speed_guess=0.3, width=1.5)

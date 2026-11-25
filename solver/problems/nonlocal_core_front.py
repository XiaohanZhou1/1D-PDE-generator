"""A nonlocal bistable equation whose front is destabilised by the nonlocal term.

    u_t = u_xx + u (1 - u) (u - a) + m (K * g(u) - B(u)),      K(x) = exp(-|x|) / 2,
    B(u) = u^2 (1 - u)^2,      g(u) = B(u) (10 u (1 - u) - 1).

w = K * g(u) is written as a second unknown with the constraint  w_xx - w + g(u) = 0.

Exact wave for every m: u = 1/(1 + exp(z/sqrt(2))), w = B(u), c = (1 - 2a)/sqrt(2).
B and g vanish to second order at u = 0 and u = 1: the end states and the essential
spectrum (sup Re = -a) do not depend on m.  The nonlocal term acts in the core only.

Results of an independent discretisation of the integral form (tests/test_core.py):
m = 5: stable;  m = -20: real eigenvalues 1.0476 and 0.4938;
m = -50: a pair 3.524 +- 0.853 i and real eigenvalues 1.518 and 0.398.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, m = sp.symbols("a m", real=True)
u, w = sp.Function("u")(x, t), sp.Function("w")(x, t)
unknowns = [u, w]

B = u ** 2 * (1 - u) ** 2
equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a) + m * (w - B)),
             sp.Eq(0, w.diff(x, 2) - w + B * (10 * u * (1 - u) - 1))]
parameters = {a: 0.25, m: -20.0}

# The equation has several fronts when |m| is large: other fronts branch off where an eigenvalue
# of this one crosses zero.  The guess selects the exact front; its width is 6 percent wrong.
z = sp.Symbol("z", real=True)
u0 = 1 / (1 + sp.exp(z / 1.5))
wave = dict(kind="front", left={u: 1, w: 0}, right={u: 0, w: 0}, speed_guess=0.3,
            guess={u: u0, w: u0 ** 2 * (1 - u0) ** 2})

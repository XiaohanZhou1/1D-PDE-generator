"""Two components with different types of boundary condition; spectrum in closed form.

    u_t = u_xx - u,   v_t = v_xx - v                  on x < s(t),
    u = 1,   v_x + r v = 1 + r                        at x = s(t),
    ds/dt = kappa1 (u_x - 1) + kappa2 (v - 1)         at x = s(t).

Wave with c = 0: U = V = exp(z).
Eigenvalues to the right of the essential spectrum (-inf, -1]:
    lambda = mu^2 - 1,   (mu + 1 + kappa1)(mu + r) = kappa2,   Re mu > 0.
For r = 1, kappa1 = 1:  kappa2 = 12 gives mu = 2, lambda = 3 (unstable);
                        kappa2 = 1 gives no root with Re mu > 0 (stable).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
kappa1, kappa2, r = sp.symbols("kappa1 kappa2 r", real=True)
u = sp.Function("u")(x, t)
v = sp.Function("v")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) - u), sp.Eq(v.diff(t), v.diff(x, 2) - v)]
free_boundary = dict(conditions=[sp.Eq(u, 1), sp.Eq(v.diff(x) + r * v, 1 + r)],
                     speed=kappa1 * (u.diff(x) - 1) + kappa2 * (v - 1))
parameters = {kappa1: 1.0, kappa2: 12.0, r: 1.0}
wave = dict(kind="semi", far={u: 0, v: 0}, speed_guess=0.1, width=1.2)

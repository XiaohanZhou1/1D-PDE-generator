"""A free-boundary problem with a Robin condition and a spectrum in closed form.

    u_t = u_xx - u               on x < s(t),
    u_x + r u = 1 + r            at x = s(t)   (r = 0: Neumann condition),
    ds/dt = kappa (u - 1)        at x = s(t).

Waves U = A exp(m z), z = x - s(t) < 0, with A = (1 + r)/(m + r) and c = (1 - m^2)/m:
  * m = 1, c = 0 for every kappa;
  * the positive roots of m^2 + (1 + r - kappa) m + r = 0  (r = 0: m = kappa - 1 for kappa > 1).
Eigenvalues to the right of the essential spectrum (-inf, -1]:
    lambda = (mu - m)(mu + m + c),   (mu + m + c)(mu + r) = kappa A m,   Re mu > 0.
Wave with c = 0: (mu + 1)(mu + r) = kappa and lambda = mu^2 - 1; unstable if and only if kappa > 2 (1 + r).
  * r = 0, kappa = 6: lambda = 3;
  * r = 1, kappa = 9: lambda = 3;
  * r = 1, kappa = 2: lambda = -0.83, stable.
`speed_guess` and `width` select the wave.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
kappa, r = sp.symbols("kappa r", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) - u)]
free_boundary = dict(conditions=[sp.Eq(u.diff(x) + r * u, 1 + r)], speed=kappa * (u - 1))
parameters = {kappa: 6.0, r: 0.0}
# The default guess is far + (b - far) exp(z / width); b follows from the boundary condition.
wave = dict(kind="semi", far={u: 0}, speed_guess=0.1, width=1.2)

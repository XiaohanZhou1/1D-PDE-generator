"""A free-boundary problem with a spectrum in closed form.

    u_t = u_xx - u               on x < s(t),
    u = 1                        at x = s(t),
    ds/dt = kappa (u_x - 1)      at x = s(t)   (Stefan condition).

Waves U = exp(m z), z = x - s(t) < 0:
  * m = 1, c = 0 for every kappa;
  * m = -1/(1 + kappa), c = (1 - m^2)/m  for kappa < -1.
The linearisation has at most one eigenvalue to the right of the essential
spectrum (-inf, -1]:
  * wave with c = 0:   lambda = kappa (kappa + 2)   if kappa < -1;
  * second wave:       lambda = kappa (1 - m)       if c < 1.
For kappa = -3: the wave with c = 0 is unstable (lambda = 3); the wave with
c = 1.5 has no eigenvalue and is stable.  `speed_guess` and `width` select the wave.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
kappa = sp.Symbol("kappa", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) - u)]
free_boundary = dict(conditions=[sp.Eq(u, 1)], speed=kappa * (u.diff(x) - 1))
parameters = {kappa: -3.0}
# The default guess is far + (boundary value - far) exp(z / width).  It is not the exact wave.
wave = dict(kind="semi", far={u: 0}, speed_guess=0.1, width=1.2)

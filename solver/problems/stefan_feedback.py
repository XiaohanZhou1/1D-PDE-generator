"""Free boundary with a component without diffusion that acts back on the diffusing one.

    u_t = u_xx - u + a w,   w_t = b u - w      on x < s(t),
    u = 1,   w = 0                             at x = s(t)   (c > 0: w enters at the boundary),
    ds/dt = kappa (u_x - 1)                    at x = s(t).

Reference (independent of the solver; tests/test_core.py, class FreeBoundaryGeneral):
modes exp(sigma z) with u-component 1 and w-component b / (lambda + 1 - c sigma), where
    (sigma^2 + c sigma - 1 - lambda)(c sigma - 1 - lambda) = a b,   Re sigma > 0   (two modes).
The wave is a combination of the two modes with lambda = 0.  The eigenvalues are the
zeros of a 3 x 3 determinant (particular solution (eta / lambda) U', two modes, three
conditions at z = 0).  For a = -1, b = 0.5 the speed c fixes kappa = c / (U'(0) - 1):
    c = 1.0: kappa = -3.311785, one real eigenvalue lambda = 0.972616: unstable;
    c = 2.5: kappa = -3.966828, no eigenvalue with Re(lambda) > -0.05: stable.
kappa(c) is not monotone: one value of kappa can have two waves.  `speed_guess` selects the wave.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
kappa, a, b = sp.symbols("kappa a b", real=True)
u = sp.Function("u")(x, t)
w = sp.Function("w")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) - u + a * w), sp.Eq(w.diff(t), b * u - w)]
free_boundary = dict(conditions=[sp.Eq(u, 1), sp.Eq(w, 0)], speed=kappa * (u.diff(x) - 1))
parameters = {kappa: -3.311785, a: -1.0, b: 0.5}
wave = dict(kind="semi", far={u: 0, w: 0}, speed_guess=0.9, width=1.2)

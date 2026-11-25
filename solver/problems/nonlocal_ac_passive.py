"""A moving front without diffusion, with a passive component: eigenvalues by two routes.

    u_t = J * u - u + k u (1 - u) (u - a),
    v_t = -v + s u (1 - u) (J * v),            J(x) = exp(-|x|) / 2.

Wave: u = the front of nonlocal_ac_front.py (with the exponential kernel), v = 0, c != 0.
The linearisation is block triangular; the v-block is
    lambda q = c q' - q + s U (1 - U) (J * q).
It has no closed form.  Reference: J is the Green's function of 1 - d^2/dx^2, so the same
equations are the local system of nonlocal_ac_passive_local.py (two auxiliary unknowns),
which the first engine (Evans function) solves.
"""
import sympy as sp

x, t, y = sp.symbols("x t y", real=True)
a, k, s = sp.symbols("a k s", positive=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

J = sp.exp(-sp.Abs(x - y)) / 2
equations = [sp.Eq(u(x, t).diff(t), sp.Integral(J * u(y, t), (y, -sp.oo, sp.oo)) - u(x, t)
                   + k * u(x, t) * (1 - u(x, t)) * (u(x, t) - a)),
             sp.Eq(v(x, t).diff(t), -v(x, t)
                   + s * u(x, t) * (1 - u(x, t)) * sp.Integral(J * v(y, t), (y, -sp.oo, sp.oo)))]
parameters = {a: 0.25, k: 1.0, s: 12.0}
wave = dict(kind="front", left={u(x, t): 1, v(x, t): 0}, right={u(x, t): 0, v(x, t): 0},
            speed_guess=0.2, width=1.5)

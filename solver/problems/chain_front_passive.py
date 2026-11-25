"""delay_front_distributed.py written as a local system (linear chain trick).

    u_t = u_xx + u (1 - u) (u - a),
    v_t = v_xx - v + sigma u (1 - u) w,
    w_t = (v - w) / T.

w(x, t) = integral over s in (0, oo) of exp(-s / T) / T * v(x, t - s) solves the third equation.
w has no diffusion: the system is partly parabolic (class L, first engine).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, sigma, T = sp.symbols("a sigma T", positive=True)
u = sp.Function("u")(x, t)
v = sp.Function("v")(x, t)
w = sp.Function("w")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a)),
             sp.Eq(v.diff(t), v.diff(x, 2) - v + sigma * u * (1 - u) * w),
             sp.Eq(w.diff(t), (v - w) / T)]
parameters = {a: 0.25, sigma: 15.0, T: 1.0}
wave = dict(kind="front", left={u: 1, v: 0, w: 0}, right={u: 0, v: 0, w: 0}, speed_guess=0.3, width=1.5)

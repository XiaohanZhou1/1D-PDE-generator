"""delay_nagumo_distributed.py written as a local system (linear chain trick).

    u_t = u_xx + u (1 - u) (u - a) + k (w - u),
    w_t = (u - w) / T.

w has no diffusion: the system is partly parabolic (class L, first engine).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, k, T = sp.symbols("a k T", positive=True)
u = sp.Function("u")(x, t)
w = sp.Function("w")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a) + k * (w - u)),
             sp.Eq(w.diff(t), (u - w) / T)]
parameters = {a: 0.25, k: 0.5, T: 2.0}
wave = dict(kind="front", left={u: 1, w: 1}, right={u: 0, w: 0}, speed_guess=0.3, width=1.5)

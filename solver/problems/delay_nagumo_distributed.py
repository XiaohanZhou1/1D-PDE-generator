"""Bistable (Nagumo) equation with a distributed delay (exponential kernel).

    u_t = u_xx + u (1 - u) (u - a) + k (integral over s in (0, oo) of exp(-s / T) / T * u(x, t - s)  -  u).

The equilibria 0, a, 1 do not depend on k and T.  For k > 0 the delayed term is monotone.
With w = the integral, w_t = (u - w) / T: the same equation is the local system of
chain_nagumo.py (first engine).  Speeds, essential spectra and labels must agree.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
s = sp.Symbol("s", positive=True)
a, k, T = sp.symbols("a k T", positive=True)
u = sp.Function("u")

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)
                   + k * (sp.Integral(sp.exp(-s / T) / T * u(x, t - s), (s, 0, sp.oo)) - u(x, t)))]
parameters = {a: 0.25, k: 0.5, T: 2.0}
wave = dict(kind="front", left={u(x, t): 1}, right={u(x, t): 0}, speed_guess=0.3, width=1.5)

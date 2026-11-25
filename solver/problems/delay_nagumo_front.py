"""Bistable (Nagumo) equation with a delayed term.

    u_t = u_xx + u (1 - u) (u - a) + k (u(x, t - tau) - u).

The equilibria 0, a, 1 do not depend on k and tau.  For k > 0 the delayed term is
monotone (quasi-monotone case).
Known result: the front is globally asymptotically stable (Smith and Zhao 2000).
At tau = 0 the term vanishes: the front is the Nagumo front with c = (1 - 2a)/sqrt(2).
Essential spectrum at u = 0 in closed form (Lambert W function):
    sup Re = A + W0(k tau exp(-A tau)) / tau,      A = -a - k.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, k, tau = sp.symbols("a k tau", real=True)
u = sp.Function("u")

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)
                   + k * (u(x, t - tau) - u(x, t)))]
parameters = {a: 0.25, k: 0.5, tau: 2.0}
wave = dict(kind="front", left={u(x, t): 1}, right={u(x, t): 0}, speed_guess=0.3, width=1.5)

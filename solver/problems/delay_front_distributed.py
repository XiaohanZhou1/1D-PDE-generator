"""A travelling front with a passive component that has a distributed delay (exponential kernel).

    u_t = u_xx + u (1 - u) (u - a),
    v_t = v_xx - v + sigma u (1 - u) * integral over s in (0, oo) of  exp(-s / T) / T * v(x, t - s).

Wave: u = the Nagumo front, v = 0, c = (1 - 2a)/sqrt(2).  The front moves, so the delayed
values are taken at z + c s.  With w = the integral, w_t = (v - w) / T (linear chain):
the same equation is the local system of chain_front_passive.py.  That file is solved by
the first engine (Evans function).  Speeds, labels and eigenvalues must agree.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
s = sp.Symbol("s", positive=True)
a, sigma, T = sp.symbols("a sigma T", positive=True)
u, v = sp.Function("u"), sp.Function("v")
unknowns = [u(x, t), v(x, t)]

equations = [sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) + u(x, t) * (1 - u(x, t)) * (u(x, t) - a)),
             sp.Eq(v(x, t).diff(t), v(x, t).diff(x, 2) - v(x, t)
                   + sigma * u(x, t) * (1 - u(x, t)) * sp.Integral(sp.exp(-s / T) / T * v(x, t - s), (s, 0, sp.oo)))]
parameters = {a: 0.25, sigma: 15.0, T: 1.0}
wave = dict(kind="front", left={u(x, t): 1, v(x, t): 0}, right={u(x, t): 0, v(x, t): 0},
            speed_guess=0.3, width=1.5)

"""Keller-Segel model with logarithmic sensitivity, after the Hopf-Cole transformation.

    Original model:  n_t = D n_xx - chi (n (ln s)_x)_x,      s_t = eps s_xx - n s.
    With v = -(ln s)_x (Hopf-Cole) it becomes a system of viscous conservation laws:

    u_t - chi (u v)_x = D u_xx,
    v_t + (eps v^2 - u)_x = eps v_xx                      (u = n).

Characteristic speeds: the roots of  r^2 + (chi - 2 eps) v r - 2 eps chi v^2 - chi u = 0  (real for u >= 0).
A travelling band of the original model is a viscous shock of this system.  Here the right state
has v+ = 0; the jump conditions give
    v-^2 = (u- - u+) / (eps + chi u- / (u- - u+)),      c = chi u- v- / (u+ - u-).
Known result (Li and Wang 2011, energy estimates in the integrated variables): the waves are stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
chi, D, eps, um, up = sp.symbols("chi D eps um up", positive=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)

equations = [sp.Eq(u.diff(t) - chi * (u * v).diff(x), D * u.diff(x, 2)),
             sp.Eq(v.diff(t) + (eps * v ** 2 - u).diff(x), eps * v.diff(x, 2))]
parameters = {chi: 1.0, D: 1.0, eps: 0.5, um: 1.0, up: 0.2}
vm = -sp.sqrt((um - up) / (eps + chi * um / (um - up)))
wave = dict(kind="front", left={u: um, v: vm}, right={u: up, v: 0}, width=2.0)

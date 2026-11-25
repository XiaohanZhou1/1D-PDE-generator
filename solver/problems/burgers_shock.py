"""Viscous shock of the Burgers equation (a conservation law).

    u_t + (u^2 / 2)_x = u_xx.

Every constant is an equilibrium.  A shock connects u_- > u_+; the jump condition gives
c = (u_- + u_+) / 2, and U = c - (D / 2) tanh(D z / 4) with D = u_- - u_+.
Known result: the shock is stable; perturbations decay algebraically in L2 (the essential
spectrum touches the origin), and exponentially if they are exponentially localised
(Sattinger 1976; Zumbrun and Howard 1998).
The integrated operator  P_zz - (U - c) P_z  is conjugate, by the weight cosh(D z / 4), to
d^2/dz^2 - D^2 / 16: in the weighted space with the rates -D/4 and +D/4 its spectrum is
(-inf, -D^2 / 16], and there is no eigenvalue.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t) + (u ** 2 / 2).diff(x), u.diff(x, 2))]
parameters = {}
wave = dict(kind="front", left={u: 2}, right={u: 0}, width=1.0)

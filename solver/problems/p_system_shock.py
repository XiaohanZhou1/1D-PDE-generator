"""p-system (isentropic gas in Lagrangian coordinates) with artificial viscosity: a shock.

    v_t - u_x = nu v_xx,
    u_t + p(v)_x = nu u_xx,      p(v) = v^(-gamma).

v: specific volume, u: velocity.  Characteristic speeds -+sqrt(-p'(v)).  Jump conditions:
    c^2 = -(p(v+) - p(v-)) / (v+ - v-),      u+ - u- = -c (v+ - v-).
The file takes the 2-shock (c > 0) from v- (compressed) to v+.
Known result: viscous shocks of the p-system with a convex pressure law are stable
(Matsumura and Nishihara 1985 for the physical viscosity; Humpherys, Lyng and Zumbrun for artificial viscosity).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
gamma, nu, vm, vp = sp.symbols("gamma nu vm vp", positive=True)
v, u = sp.Function("v")(x, t), sp.Function("u")(x, t)

p = lambda s: s ** (-gamma)
equations = [sp.Eq(v.diff(t) - u.diff(x), nu * v.diff(x, 2)),
             sp.Eq(u.diff(t) + p(v).diff(x), nu * u.diff(x, 2))]
unknowns = [v, u]
parameters = {gamma: 1.4, nu: 1.0, vm: 1.0, vp: 2.0}
c = sp.sqrt(-(p(vp) - p(vm)) / (vp - vm))
wave = dict(kind="front", left={v: vm, u: 0}, right={v: vp, u: -c * (vp - vm)}, width=2.0)

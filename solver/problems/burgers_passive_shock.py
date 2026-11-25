"""A shock of a system of two viscous conservation laws: Burgers equation with a passive field.

    u_t + (u^2 / 2)_x = u_xx,
    v_t + (a v + g u)_x = d v_xx.

Characteristic speeds: u and a.  The shock of the u-field (u: 2 -> 0, speed c = 1) is a Lax shock of
the system if a differs from c.  The jump condition gives v+ - v- = g (u- - u+) / (a - c).
At the right end state the characteristic speeds relative to the shock have both signs (-1 and
a - c > 0): no exponential weight moves the essential spectrum away from the origin.  The label
comes from the determinant of the integrated operator on the closed half-plane Re(lambda) >= 0.
Known result: stable.  The system is triangular; the u-block is the Burgers shock, and the v-block
is an advection-diffusion operator without zeroth-order term (no eigenvalue in Re >= 0).
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, g, d = sp.symbols("a g d", real=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)

equations = [sp.Eq(u.diff(t) + (u ** 2 / 2).diff(x), u.diff(x, 2)),
             sp.Eq(v.diff(t) + (a * v + g * u).diff(x), d * v.diff(x, 2))]
parameters = {a: 3.0, g: 1.0, d: 0.5}
wave = dict(kind="front", left={u: 2, v: 0}, right={u: 0, v: 2 * g / (a - 1)}, width=1.0)

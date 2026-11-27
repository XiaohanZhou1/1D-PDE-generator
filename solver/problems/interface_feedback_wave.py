"""Exact two-component advancing interface wave with variable feedback."""
import sympy as sp

x, t = sp.symbols("x t", real=True)
gamma, beta = sp.symbols("gamma beta", real=True)
u, v = (sp.Function(name)(x, t) for name in ("u", "v"))
equations = [
    sp.Eq(u.diff(t), u.diff(x, 2) - u + gamma * (v - u**2)),
    sp.Eq(v.diff(t), v.diff(x, 2) - sp.Rational(5, 2) * v),
]
free_boundary = dict(conditions=[sp.Eq(u, 1), sp.Eq(v, 1)],
                     speed=sp.Rational(3, 2) + 3*(u.diff(x) - sp.Rational(1, 2)) + beta*(v.diff(x) - 1))
parameters = {gamma: 1, beta: -8}
z = sp.Symbol("z", real=True)
wave = dict(kind="semi", far={u: 0, v: 0}, speed=sp.Rational(3, 2),
            guess={u: sp.exp(z / 2), v: sp.exp(z)})

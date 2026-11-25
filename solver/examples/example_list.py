"""Example of the input format: the list `problems`.

Run from the folder that holds `twstab`:
    python -m twstab examples/example_list.py

Every entry gives the equations and the known conditions (parameter values; end states or speed where
the class needs them, see README.md).  No entry gives a guess of the wave.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u, v, m = (sp.Function(name)(x, t) for name in ("u", "v", "m"))
U = sp.Function("u")                                   # for delayed and nonlocal arguments: U(x, t - tau)
a, eps, chi, mu, k, kappa, tau, sigma, p = sp.symbols("a eps chi mu k kappa tau sigma p", positive=True)
y = sp.Symbol("y", real=True)

# 1. Reaction-diffusion (bistable).  Equation and parameter value only.
eqn1 = dict(equations=sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a)),
            parameters={a: 0.25})

# 2. Reaction-diffusion (monostable, Fisher-KPP).  No parameter: the entry is the equation itself.
eqn2 = sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u))

# 3. Conservation law (Burgers).  End states are required.
eqn3 = dict(equations=sp.Eq(u.diff(t) + (u ** 2 / 2).diff(x), u.diff(x, 2)),
            wave=dict(kind="front", left={u: 2}, right={u: 0}))

# 4. Haptotaxis towards a component without diffusion (system of two equations).
eqn4 = dict(equations=[sp.Eq(u.diff(t), eps * u.diff(x, 2) - chi * (u * m.diff(x)).diff(x) + u * (1 - u) * (u - a)),
                       sp.Eq(m.diff(t), mu * (1 - m) - k * u * m)],
            parameters={eps: 1.0, chi: 1.0, a: 0.25, mu: 0.5, k: 1.0},
            options=dict(equilibria_range=3.0))

# 5. Free boundary with a Stefan condition (Fisher-Stefan): u = 0 and ds/dt = -kappa u_x at x = s(t).
eqn5 = dict(equations=sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u)),
            free_boundary=dict(conditions=[sp.Eq(u, 0)], speed=-kappa * u.diff(x)),
            parameters={kappa: 1.0})

# 6. Time delay.
eqn6 = dict(equations=sp.Eq(U(x, t).diff(t), U(x, t).diff(x, 2) + U(x, t) * (1 - U(x, t)) * (U(x, t) - a)
                            + k * (U(x, t - tau) - U(x, t))),
            parameters={a: 0.25, k: 0.5, tau: 2.0})

# 7. Nonlocal term: convolution with a Gaussian kernel.
kernel = sp.exp(-(x - y) ** 2 / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))
eqn7 = dict(equations=sp.Eq(U(x, t).diff(t), U(x, t).diff(x, 2) + U(x, t) * (1 - U(x, t)) * (U(x, t) - a)
                            + mu * (sp.Integral(kernel * U(y, t), (y, -sp.oo, sp.oo)) - U(x, t))),
            parameters={a: 0.25, mu: 1.0, sigma: 1.0})

# 8. Second order in time, without damping (phi-4).
eqn8 = dict(equations=sp.Eq(u.diff(t, 2), u.diff(x, 2) + u - u ** 3),
            options=dict(equilibria_range=3.0))

# 9. Dispersive solitary wave (generalised Korteweg-de Vries).  Rest state and speed are required.
eqn9 = dict(equations=sp.Eq(u.diff(t) + u ** p * u.diff(x) + u.diff(x, 3), 0),
            parameters={p: 2},
            wave=dict(kind="pulse", rest={u: 0}, speed=1.0))

# 10. Fourth order, conservation form (Cahn-Hilliard).  End states are required.
eqn10 = dict(equations=sp.Eq(u.diff(t), -(u.diff(x, 2) + u - u ** 3).diff(x, 2)),
             wave=dict(kind="front", left={u: -1}, right={u: 1}))

# 11. Slow-fast system (FitzHugh-Nagumo), named by the user.
eqn11 = dict(name="fitzhugh_nagumo",
             equations=[sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a) - v),
                        sp.Eq(v.diff(t), eps * (u - mu * v))],
             parameters={a: 0.1, eps: 0.01, mu: 1.0})

problems = [eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9, eqn10, eqn11]

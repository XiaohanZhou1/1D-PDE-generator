"""A front of a cross-diffusion system with eigenvalues in closed form.

    u_t = u_xx + u (1 - u) (u - a) + beta (u v_x)_x,
    v_t = v_xx + (sigma u (1 - u) - 1) v.

Wave: v = 0 and u = the Nagumo front 1/(1 + exp(z/sqrt(2))), c = (1 - 2a)/sqrt(2).
The linearisation is block triangular.  The v-block is a Poeschl-Teller operator:
with s (s + 1) = 2 sigma its eigenvalues are

    lambda_n = (s - n)^2 / 8 - 1 - c^2 / 4,      n = 0, 1, ... < s.

The cross-diffusion term beta (u v_x)_x does not change the eigenvalues, but it
makes the second-order coefficient matrix non-symmetric (indefinite symmetric part
for beta > 2).
sigma = 15 (s = 5): eigenvalues 2.09375, 0.96875, 0.09375 > 0: unstable.
sigma = 3  (s = 2): largest eigenvalue -0.53125: stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, beta, sigma = sp.symbols("a beta sigma", positive=True)
u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a) + beta * (u * v.diff(x)).diff(x)),
             sp.Eq(v.diff(t), v.diff(x, 2) + (sigma * u * (1 - u) - 1) * v)]
parameters = {a: 0.25, beta: 3.0, sigma: 15.0}
wave = dict(kind="front", left={u: 1, v: 0}, right={u: 0, v: 0}, speed_guess=0.3, width=1.5)

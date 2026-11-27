"""Standing kink with localized negative delayed feedback on a shifted window."""
import sympy as sp

x, t = sp.symbols("x t", real=True)
s = sp.Symbol("s", positive=True)
sigma, tau, epsilon = sp.symbols("sigma tau epsilon", real=True)
u, v = sp.Function("u"), sp.Function("v")
u_now, v_now = u(x, t), v(x, t)
width = sp.Rational(1, 4)
memory = sp.Integral(v(x, t-s) / width, (s, tau, tau+width))
equations = [
    sp.Eq(u_now.diff(t), u_now.diff(x, 2) + u_now - u_now**3),
    sp.Eq(v_now.diff(t), v_now.diff(x, 2) - v_now
          - sigma*(1-u_now**2)*(1+epsilon*u_now)*memory),
]
parameters = {sigma: 5, tau: sp.Rational(3, 2), epsilon: sp.Rational(3, 4)}
wave = dict(kind="front", left={u_now: -1, v_now: 0}, right={u_now: 1, v_now: 0},
            speed_guess=0, width=1.2)

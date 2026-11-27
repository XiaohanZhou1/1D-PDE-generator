"""G4 delayed-kink numerical probe for the full spectral-count task."""
import sympy as sp

x, t = sp.symbols("x t", real=True)
s = sp.Symbol("s", positive=True)
epsilon = sp.Symbol("epsilon", real=True)
u, v = sp.Function("u"), sp.Function("v")
u_now, v_now = u(x, t), v(x, t)
memory = (sp.Integral(v(x, t-s), (s, 0, 1))/2
          + sp.Integral(v(x, t-s), (s, 0, 2))/4)
equations = [
    sp.Eq(u_now.diff(t), u_now.diff(x, 2) + u_now - u_now**3),
    sp.Eq(v_now.diff(t), v_now.diff(x, 2) - v_now
          + 6*(1-u_now**2)*(1+epsilon*u_now)*memory),
]
parameters = {epsilon: sp.Rational(1, 8)}
wave = dict(kind="front", left={u_now: -1, v_now: 0}, right={u_now: 1, v_now: 0},
            speed_guess=0.0, width=1.2)

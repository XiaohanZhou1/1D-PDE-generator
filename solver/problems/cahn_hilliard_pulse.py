"""Cahn-Hilliard equation: a stationary pulse (critical nucleus) on a metastable state.

    u_t = -(u_xx + u - u^3)_xx,      rest state u0 with 1/sqrt(3) < |u0| < 1.

The pulse solves  U'' + U - U^3 = u0 - u0^3  and does not move.  It is the saddle between the
metastable state and the nucleated phase.
Known result: the pulse is unstable (Bates and Fife 1993): a smaller bump shrinks, a larger one grows.
The essential spectrum touches the origin and there is no transport: the neutral mode applies.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
u0 = sp.Symbol("u0", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), -(u.diff(x, 2) + u - u ** 3).diff(x, 2))]
parameters = {u0: -0.8}
# The speed of a pulse of a conservation law is a free parameter: it is given.  Amplitude: U(0) - u0 = 0.75.
wave = dict(kind="pulse", rest={u: u0}, amplitude={u: 0.75}, width=1.6, speed=0.0)

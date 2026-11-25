"""A pushed front into an unstable state: cubic nonlinearity with a < 0.

    u_t = u_xx + u (1 - u) (u - a),      -1/2 < a < 0.

u = 0 is unstable (f'(0) = -a > 0): in L2 every front into it is unstable through the essential
spectrum.  The front U = 1 / (1 + exp(z / sqrt(2))) with the speed c = (1 - 2a) / sqrt(2) is the
pushed front: it is faster than the linear spreading speed 2 sqrt(-a) and decays with the fast
rate 1 / sqrt(2).  The fronts form a one-parameter family, so the speed is given.
Known results in the weighted space exp(r z) ahead of the front:
  * the essential spectrum is stable for -sqrt(2) a < r < 1 / sqrt(2); at r = c / 2 its supremum
    is -a - c^2 / 4 (a = -0.25: -0.03125);
  * the translation mode lies in the space (the front decays faster than the weight grows);
  * there is no other spectrum in Re(lambda) >= 0: the pushed front is stable.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a = sp.Symbol("a", real=True)
u = sp.Function("u")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a))]
parameters = {a: -0.25}
wave = dict(kind="front", left={u: 1}, right={u: 0}, speed=(1 - 2 * a) / sp.sqrt(2), width=1.5)

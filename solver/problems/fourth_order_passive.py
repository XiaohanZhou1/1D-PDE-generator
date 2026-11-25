"""A moving front with a passive component of fourth order: eigenvalues in closed form.

    u_t = u_xx + u (1 - u) (u - a),
    v_t = -c v_x - B (B v) + alpha B v,       B w = w_xx + c w_x + (sigma u (1 - u) - 1) w,
    c = (1 - 2a) / sqrt(2)                    (the speed of the Nagumo front; a coefficient of the equation).

Wave: u = the Nagumo front, v = 0, speed c.  In the frame of the wave the v-block of the
linearisation is  -B^2 + alpha B  with  B = d^2/dz^2 + c d/dz - 1 + sigma U (1 - U).
B is the operator of cross_diffusion_front.py: its eigenvalues are
    b_n = (s - n)^2 / 8 - 1 - c^2 / 4,     s (s + 1) = 2 sigma        (sigma = 15: 2.09375, 0.96875, 0.09375, ...).
So the eigenvalues of the fourth-order problem are  -b_n^2 + alpha b_n.
alpha = 1:    +0.084961 (n = 2) and +0.030273 (n = 1): unstable.
alpha = -0.4: all negative (the largest is -0.046289): stable.
Essential spectrum of the v-block: up to -1 - alpha; of the u-block: up to -a.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, sigma, alpha = sp.symbols("a sigma alpha", real=True)
u = sp.Function("u")(x, t)
v = sp.Function("v")(x, t)
unknowns = [u, v]

c = (1 - 2 * a) / sp.sqrt(2)


def B(w):
    return w.diff(x, 2) + c * w.diff(x) + (sigma * u * (1 - u) - 1) * w


equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a)),
             sp.Eq(v.diff(t), -c * v.diff(x) - B(B(v)) + alpha * B(v))]
parameters = {a: 0.25, sigma: 15.0, alpha: 1.0}
wave = dict(kind="front", left={u: 1, v: 0}, right={u: 0, v: 0}, speed_guess=0.3, width=1.5)

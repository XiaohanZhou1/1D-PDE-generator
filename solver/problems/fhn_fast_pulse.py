"""FitzHugh-Nagumo equations: the fast pulse.

    u_t = u_xx + u (1 - u) (u - a) - w,      w_t = eps (u - gamma w).

The second component has no diffusion (partly parabolic system).  For small eps
there are two pulses with the same end state: a fast one and a slow one.
Known result: the fast pulse is spectrally stable (Jones 1984, Yanagida 1985).

The guess is the singular (eps -> 0) construction: a front at z = 0, a back at
z = -W, and w rising linearly between them to the value ws of the back.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
z = sp.Symbol("z", real=True)
a, eps, gamma = sp.symbols("a epsilon gamma", positive=True)
u, w = sp.Function("u")(x, t), sp.Function("w")(x, t)

equations = [sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a) - w),
             sp.Eq(w.diff(t), eps * (u - gamma * w))]
parameters = {a: 0.1, eps: 0.01, gamma: 1.0}

c0 = (1 - 2 * a) / sp.sqrt(2)                 # speed of the front at eps = 0
um = (2 - a) / 3                              # middle root at the back
ws = um * (1 - um) * (um - a)                 # w at the back
ul = (1 + a - um - sp.sqrt((1 + a - um) ** 2 + 4 * ws / um)) / 2   # left root at the back
W = c0 * ws / (eps * (1 - ws / 2 - gamma * ws / 2))        # length of the excited state
front = (1 - sp.tanh(z / (2 * sp.sqrt(2)))) / 2
back = (1 + sp.tanh((z + W) / (2 * sp.sqrt(2)))) / 2


def pos(y):
    """Smooth positive part: y for y >> 1, 0 for y << -1.  No exponential, so no overflow on a long domain."""
    return (y + sp.sqrt(y ** 2 + 1)) / 2


tail = sp.exp(-pos(-(z + W)) * eps * (gamma + 1 / a) / c0)     # recovery behind the back
ramp = (pos(-z) - pos(-z - W)) / W                             # from 0 (z > 0) to 1 (z < -W)
wt = ws * tail
k = (-ws / ul - a) / ws                       # left branch of the cubic: u = -w / (a + k w)
wave = dict(kind="pulse", rest={u: 0, w: 0}, speed_guess=c0,
            guess={u: front * back - (1 - back) * wt / (a + k * wt),
                   w: ws * ramp * back + (1 - back) * wt})

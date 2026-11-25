"""nonlocal_ac_passive.py written as a local system (the exponential kernel is a Green's function).

    u_t = m - u + k u (1 - u) (u - a),        0 = m_xx - m + u        (m = J * u),
    v_t = -v + s u (1 - u) n,                 0 = n_xx - n + v        (n = J * v).

u and v have no diffusion; m and n are constraint unknowns.  The energy estimate does not
cover this combination, so the search rectangle must be given (`options['region']`, or the
keyword `region` of twstab.label.analyse).  The first engine (Evans function) does the rest.
"""
import sympy as sp

x, t = sp.symbols("x t", real=True)
a, k, s = sp.symbols("a k s", positive=True)
u = sp.Function("u")(x, t)
v = sp.Function("v")(x, t)
m = sp.Function("m")(x, t)
n = sp.Function("n")(x, t)
unknowns = [u, v, m, n]

equations = [sp.Eq(u.diff(t), m - u + k * u * (1 - u) * (u - a)),
             sp.Eq(v.diff(t), -v + s * u * (1 - u) * n),
             sp.Eq(0, m.diff(x, 2) - m + u),
             sp.Eq(0, n.diff(x, 2) - n + v)]
parameters = {a: 0.25, k: 1.0, s: 12.0}
wave = dict(kind="front", left={u: 1, v: 0, m: 1, n: 0}, right={u: 0, v: 0, m: 0, n: 0},
            speed_guess=0.2, width=1.5)
options = dict(region=(6.0, 12.0))

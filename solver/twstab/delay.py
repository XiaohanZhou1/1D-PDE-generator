"""Sub-solver D: systems with time delays (DESIGN.md, sections 2.9 and 3.7).

    u_t = G(u, u_x, u_xx, u(x, t - tau_1), ..., u(x, t - tau_Q), I_1, ..., I_r),
    I = integral over s of  g(s) F(u(x, t - s))        (distributed delay).

A distributed delay is replaced by discrete delays: quadrature nodes s_q with the
weights omega_q g(s_q).  After that the problem has the delays tau_1, ..., tau_Q only.

In the frame z = x - c t a delayed term is U(z + c tau_k).  The eigenvalue problem

    lambda p(z) = L0 p(z) + sum_k exp(-lambda tau_k) B_k(z) p(z + c tau_k)

contains shifts in z and is not polynomial in lambda: it is not an ODE system,
and the Evans engine does not apply.  This module provides the second engine:

* DelaySystem      symbolic reduction; the delayed values are extra arguments
* essential_bound  rightmost roots of the dispersion relation (Chebyshev in the delay)
* solve_wave       Newton iteration on a uniform grid, continuation in the size of the delays
* DelayEngine      banded matrix T(lambda); zeros are counted by the winding number
                   of det T(lambda) / det (lambda - R), R a reference operator
* generator_matrix independent discretisation: the delay interval is discretised
                   too, which gives an ordinary sparse eigenvalue problem
"""
from __future__ import annotations

import warnings

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.linalg.lapack import zgbtrf
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import spsolve
from sympy.core.function import AppliedUndef

from .problem import Problem
from .symbolic import LocalSystem, StructureError, _vectorised
from .wave import Wave, WaveError
from .wave import solve_wave as solve_local_wave

GAUSS = np.polynomial.legendre.leggauss(16)
CUTOFF = 1e-10                # neglected tail of the kernel of a distributed delay, relative to its integral
QUAD_TOL = 1e-7               # tolerance of the quadrature of a kernel, relative to its integral
FREQUENCY = 12.0              # first quadrature: exact up to this frequency, in units of 1 / (mean delay)
BAND = 40                     # the coarsest spectral grid has at most this many nodes inside c * (longest delay)
SCAN_TAIL = 1e-4              # the Chebyshev discretisations leave out the last part of a kernel with this weight


def find_delay(problem):
    """Delayed terms of the equations.  Returns None, or dict(discrete=..., integrals=...).

    discrete : {u_j(x, t - tau): (component j, tau)} for the terms outside integrals
    integrals: distributed delays  Integral(g(s) F(u(x, t - s)), (s, a, b)).  Every entry holds
               atom, s, lower, upper, kernel g(s), body F with placeholders, terms {placeholder: (j, delay)}.
               F contains delayed values only.
    """
    x, t = problem.x, problem.t
    funcs = [u.func for u in problem.unknowns]

    def delay_of(f):
        if f.func in funcs and len(f.args) == 2 and f.args[0] == x and f.args[1] != t:
            delay = sp.simplify(t - f.args[1])
            if delay.has(x) or delay.has(t):
                raise StructureError("a delay must be a constant")
            return delay
        return None

    discrete, integrals = {}, []
    for eq in problem.equations:
        e = sp.sympify(eq.lhs - eq.rhs)
        hidden = {}
        for integral in e.atoms(sp.Integral):
            delayed = {f: delay_of(f) for f in integral.function.atoms(AppliedUndef)}
            delayed = {f: d for f, d in delayed.items() if d is not None}
            if not delayed:
                continue                               # a convolution in space: kernel.py
            hidden[integral] = sp.Dummy()
            if any(integral == entry["atom"] for entry in integrals):
                continue
            if len(integral.limits) != 1 or len(integral.limits[0]) != 3:
                raise StructureError("a distributed delay must have one integration variable and two limits")
            s, lower, upper = integral.limits[0]
            if integral.function.atoms(sp.Derivative):
                raise StructureError("derivatives inside a distributed delay are not supported")
            if not all(d.has(s) for d in delayed.values()):
                raise StructureError("inside a distributed delay, every delayed term must contain "
                                     "the integration variable")
            W = {f: sp.Dummy(f"W{funcs.index(f.func)}", real=True) for f in delayed}
            body = integral.function.xreplace(W)
            if body.atoms(AppliedUndef):
                raise StructureError("the integrand of a distributed delay may contain the unknowns "
                                     "at (x, t - s) only; write other factors outside the integral")
            kernel, F = body.as_independent(*W.values(), as_Add=False)
            if F.has(s):
                kernel, F = sp.factor(body).as_independent(*W.values(), as_Add=False)
            if F.has(s):
                raise StructureError("the integrand of a distributed delay must be a product g(s) F(u(x, t - s))")
            integrals.append(dict(atom=integral, s=s, lower=lower, upper=upper, kernel=kernel, body=F,
                                  terms={W[f]: (funcs.index(f.func), delayed[f]) for f in delayed}))
        for f in e.xreplace(hidden).atoms(AppliedUndef):
            delay = delay_of(f)
            if delay is not None:
                discrete[f] = (funcs.index(f.func), delay)
    if not discrete and not integrals:
        return None
    return dict(discrete=discrete, integrals=integrals)


def _quadrature(g, lower, upper, frequency=None):
    """Nodes and weights (kernel included) for  integral of g(s) f(s) ds  over [lower, upper].

    Composite Gauss-Legendre rule with 16 points on a panel.  A panel is bisected until
    the integrals of g and of g exp(-i frequency s) agree with those of its two halves.
    An infinite interval is cut where the remaining integral of |g| is below CUTOFF.
    Returns dict(nodes, weights, mass, mean, upper, decay, frequency).
    """
    G = lambda s: np.broadcast_to(np.asarray(g(s), dtype=float), np.shape(s))
    size = lambda s: float(np.abs(G(np.array([s], dtype=float)))[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mass = quad(size, lower, upper, limit=200)[0]
        if not np.isfinite(mass) or mass <= 0:
            raise StructureError("the kernel of a distributed delay is not integrable")
        integral = quad(lambda s: float(G(np.array([s], dtype=float))[0]), lower, upper, limit=200)[0]
        mean = quad(lambda s: (s - lower) * size(s), lower, upper, limit=200)[0] / mass + lower
        if not np.isfinite(mean) or mean <= 0:
            raise StructureError("the kernel of a distributed delay has no finite mean delay")
        decay = np.inf                                 # exponential decay rate of the kernel
        if np.isinf(upper):
            upper = lower + (mean - lower)
            for _ in range(60):
                tail = quad(size, upper, np.inf, limit=200)[0]
                if tail <= CUTOFF * mass:
                    break
                upper = lower + 1.5 * (upper - lower)
                if upper - lower > 200 * (mean - lower):
                    raise StructureError("the kernel of a distributed delay decays too slowly")
            decay = float(np.log(mass / max(tail, 1e-300)) / upper)
    frequency = FREQUENCY / mean if frequency is None else float(frequency)

    def panel(a, b):
        s = (a + b) / 2 + (b - a) / 2 * GAUSS[0]
        w = (b - a) / 2 * GAUSS[1] * G(s)
        return s, w, np.array([w.sum(), (w * np.exp(-1j * frequency * s)).sum()])

    nodes, weights = [], []

    def split(a, b, whole, depth):
        mid = (a + b) / 2
        halves = panel(a, mid), panel(mid, b)
        error = np.abs(halves[0][2] + halves[1][2] - whole[2]).max()
        if depth >= 40 or error <= QUAD_TOL * mass * (b - a) / (upper - lower):
            nodes.append(whole[0])
            weights.append(whole[1])
        else:
            split(a, mid, halves[0], depth + 1)
            split(mid, b, halves[1], depth + 1)

    split(lower, upper, panel(lower, upper), 0)
    nodes, weights = np.concatenate(nodes), np.concatenate(weights)
    if abs(weights.sum()) > 0.1 * mass:                # the equilibria must not move: keep the integral of g
        weights *= integral / weights.sum()
    keep = weights != 0
    return dict(nodes=nodes[keep], weights=weights[keep], mass=float(mass), mean=float(mean),
                upper=float(upper), decay=decay, frequency=frequency)


class DelaySystem:
    """Local functions of (U, U', U'', V) with V = the delayed values U_j(z + c tau_k).

    The rows of V are the pairs (component j, delay k).  A discrete delayed term is one argument
    of the compiled functions.  A distributed delay is one argument too: the value of its integral,
    I = sum_q w_q F(V at the node q).
    """

    def __init__(self, problem, frequency=None):
        found = find_delay(problem)
        us = problem.unknowns
        for eq in problem.equations:
            for d in sp.sympify(eq.lhs - eq.rhs).atoms(sp.Derivative):
                if d.expr in found["discrete"]:
                    raise StructureError("derivatives of the delayed terms are not supported")

        def number(expression, what):
            try:
                return float(sp.sympify(expression).subs(problem.parameters))
            except TypeError as error:
                raise StructureError(f"{what} is not a number once the parameters are inserted") from error

        taus, tails, pairs = [], [], []

        def index(value, tail):
            """Index of the delay `value`; tail: weight of the kernel beyond it (1 for a discrete delay)."""
            if not value > 0:
                raise StructureError("a delay must be positive")
            for k, old in enumerate(taus):
                if abs(old - value) <= 1e-12 * max(old, value):
                    tails[k] = max(tails[k], tail)
                    return k
            taus.append(value)
            tails.append(tail)
            return len(taus) - 1

        def row(j, k):
            if (j, k) not in pairs:
                pairs.append((j, k))
            return pairs.index((j, k))

        delayed_map, present_map = {}, {}              # delayed terms -> symbols; -> the present values
        discrete = {}                                  # symbol of a discrete delayed term -> row of V
        for term, (j, delay) in found["discrete"].items():
            k = index(number(delay, "a delay"), 1.0)
            delayed_map[term] = sp.Symbol(f"delayed_{us[j].func.__name__}_{k}", real=True)
            discrete[delayed_map[term]] = row(j, k)
            present_map[term] = us[j]
        self.integrals, kernels = [], []
        for count, entry in enumerate(found["integrals"]):
            lower = number(entry["lower"], "a limit of a distributed delay")
            upper = np.inf if entry["upper"] == sp.oo else number(entry["upper"], "a limit of a distributed delay")
            if not 0 <= lower < upper:
                raise StructureError("a distributed delay needs the limits 0 <= a < b")
            rule = _quadrature(sp.lambdify(entry["s"], entry["kernel"].subs(problem.parameters), "numpy"),
                               lower, upper, frequency)
            size = np.abs(rule["weights"])
            beyond = (size.sum() - np.cumsum(size) + size) / size.sum()      # weight of the kernel from a node on
            W = list(entry["terms"])
            rows = np.array([[row(j, index(number(delay.subs(entry["s"], node), "a delay"), float(tail)))
                              for j, delay in (entry["terms"][w] for w in W)]
                             for node, tail in zip(rule["nodes"], beyond)])              # (nodes, placeholders)
            F = entry["body"].subs(problem.parameters)
            value = sp.Symbol(f"distributed_{count}", real=True)
            delayed_map[entry["atom"]] = value
            present_map[entry["atom"]] = float(rule["weights"].sum()) * entry["body"].xreplace(
                {w: us[entry["terms"][w][0]] for w in W})
            kernels.append(dict(symbol=value, weights=rule["weights"], rows=rows,
                                components=[entry["terms"][w][0] for w in W],
                                F=_vectorised(W, [F]), dF=_vectorised(W, [sp.diff(F, w) for w in W])))
            self.integrals.append(dict(lower=lower, upper=rule["upper"], nodes=int(rule["nodes"].size),
                                       mass=rule["mass"], mean_delay=rule["mean"], decay=rule["decay"],
                                       frequency=rule["frequency"]))
        self.frequency = min([i["frequency"] for i in self.integrals], default=np.inf)
        self.decay = min([i["decay"] for i in self.integrals], default=np.inf)

        def replaced(mapping, parameters):
            equations = [sp.Eq(eq.lhs.xreplace(mapping), eq.rhs.xreplace(mapping)) for eq in problem.equations]
            return Problem(path=problem.path, name=problem.name, x=problem.x, t=problem.t, equations=equations,
                           unknowns=us, parameters=parameters, wave=problem.wave, options=problem.options)

        # All delays zero: the delayed terms become the present ones.  This local problem starts the continuation.
        self.problem0 = replaced(present_map, problem.parameters)
        self.system0 = LocalSystem(self.problem0)
        extra = list(discrete) + [k["symbol"] for k in kernels]
        self.local = local = LocalSystem(replaced(delayed_map, {**problem.parameters, **{v: 0.0 for v in extra}}))
        if any(m != 2 for m in local.mu) or np.any(local.mass == 0):
            raise StructureError("with a delay, every component must have a diffusion term "
                                 "(no constraint and no diffusion-free component)")
        names = {str(p) for p in local.params}
        extra = [v for v in extra if str(v) in names]                 # a term that cancels leaves no argument
        if not extra:
            raise StructureError("the delayed terms cancel")
        self.pairs = pairs                                            # (component, delay index) of every row of V
        self.taus, self.tails = np.array(taus), np.array(tails)
        self.tau = float(self.taus.max())                             # the longest delay
        self.delayed = sorted({j for j, _ in pairs})
        self.n, self.K, self.N0, self.offset = local.n, local.K, local.N, local.offset
        self.N = local.N + len(pairs)                                 # extended state: jets and delayed values
        self.mass, self.first_order_components = local.mass, []
        self._component = np.array([j for j, _ in pairs])
        self._discrete = [(discrete[v], local.parameter_index(v), position)        # (row of V, parameter, argument)
                          for position, v in enumerate(extra) if v in discrete]
        self._kernels = [dict(k, index=local.parameter_index(k["symbol"]), position=extra.index(k["symbol"]))
                         for k in kernels if k["symbol"] in extra]
        sym = local.symbols
        self._dG = _vectorised(sym["low"] + sym["T"] + [sym["c"]] + local.params,
                               [sp.diff(g, v) for g in sym["G"] for v in extra])  # dG / d(extra arguments)
        self._extra = len(extra)

    def theta(self, V):
        """Parameter tuple of the local system with the delayed values inserted (arrays are allowed)."""
        theta = list(self.local.theta)
        for row, index, _ in self._discrete:
            theta[index] = V[row]
        for kernel in self._kernels:
            rows = kernel["rows"]
            values = kernel["F"](*[V[rows[:, w]] for w in range(rows.shape[1])])[0]      # (nodes, m)
            theta[kernel["index"]] = np.tensordot(kernel["weights"], values, axes=1)
        return tuple(theta)

    def pair_matrices(self, Y0, top, c, V):
        """dG_i / d(row of V), shape (n, rows of V, m).  The local system must hold theta(V)."""
        m = Y0.shape[1]
        dG = self._dG(*Y0, *top, c, *self.local.theta).reshape(self.n, self._extra, m)
        out = np.zeros((self.n, len(self.pairs), m))
        for row, _, position in self._discrete:
            out[:, row, :] += dG[:, position, :]
        for kernel in self._kernels:
            rows = kernel["rows"]
            dF = kernel["dF"](*[V[rows[:, w]] for w in range(rows.shape[1])])            # (placeholders, nodes, m)
            for w in range(rows.shape[1]):
                out[:, rows[:, w], :] += (dG[:, None, kernel["position"], :]
                                          * (kernel["weights"][:, None] * dF[w])[None])
        return out

    def state(self, u):
        """Extended state of the constant solution u."""
        return np.concatenate((self.local.state(u), np.asarray(u, dtype=float)[self._component]))

    def equilibrium_residual(self, u):
        return self.system0.equilibrium_residual(u)

    def _split(self, Y):
        Y = np.asarray(Y, dtype=float).reshape(self.N, -1)
        return Y[:self.N0], Y[self.N0:]

    def coefficients(self, Y, c):
        """Coefficients a^(k) of the local part L0 at extended states, shape (K+1, m, n, n)."""
        Y0, V = self._split(Y)
        keep, self.local.theta = self.local.theta, self.theta(V)
        try:
            return self.local.coefficients(Y0, c)
        finally:
            self.local.theta = keep

    def delay_matrices(self, Y, c):
        """B_k = dG/d(delayed values of the delay k) as n x n matrices, shape (Q, m, n, n)."""
        Y0, V = self._split(Y)
        keep, self.local.theta = self.local.theta, self.theta(V)
        try:
            values = self.pair_matrices(Y0, self.local._solve_top(Y0, c)[1], c, V)
        finally:
            self.local.theta = keep
        B = np.zeros((self.taus.size, Y0.shape[1], self.n, self.n))
        for row, (j, k) in enumerate(self.pairs):
            B[k, :, :, j] = values[:, row, :].T
        return B

    def end_state(self, u, c):
        Y = self.state(u)
        return self.coefficients(Y, c)[:, 0], self.delay_matrices(Y, c)[:, 0]


# ------------------------------------------------------------------------------------------
# Essential spectrum
# ------------------------------------------------------------------------------------------
def chebyshev(M):
    """Chebyshev points cos(k pi / M) and the differentiation matrix on [-1, 1]."""
    k = np.arange(M + 1)
    x = np.cos(np.pi * k / M)
    weight = np.where((k == 0) | (k == M), 2.0, 1.0) * (-1.0) ** k
    D = (weight[:, None] / weight[None, :]) / (x[:, None] - x[None, :] + np.eye(M + 1))
    D -= np.diag(D.sum(axis=1))
    return x, D


def _history(taus, tau_max, M):
    """Lagrange basis of the Chebyshev points of [-tau_max, 0] at the points -tau_k, shape (Q, M + 1)."""
    k = np.arange(M + 1)
    x = np.cos(np.pi * k / M)
    weight = np.where((k == 0) | (k == M), 0.5, 1.0) * (-1.0) ** k
    distance = (1 - 2 * np.asarray(taus, dtype=float) / tau_max)[:, None] - x[None, :]
    hit = np.abs(distance) < 1e-13
    terms = weight[None, :] / np.where(hit, 1.0, distance)
    out = terms / terms.sum(axis=1, keepdims=True)
    rows = hit.any(axis=1)
    out[rows] = hit[rows].astype(float)
    return out


def _symbol(a_k, kappa, c):
    """sum_k a_k (i kappa)^k - i kappa c, shape (len(kappa), n, n): the local part for mu = lambda - i kappa c."""
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    powers = (1j * kappa[:, None]) ** np.arange(a_k.shape[0])[None, :]
    return np.einsum("lk,kij->lij", powers, a_k) - (1j * kappa * c)[:, None, None] * np.eye(a_k.shape[1])[None]


def _generator(a_k, B, c, taus, kappa, M):
    """Eigenvalues mu of the generator of the delay equation at the wavenumbers kappa.

    The delay interval [-tau_max, 0] is discretised with M + 1 Chebyshev points
    (infinitesimal generator approach).  With mu = lambda - i kappa c the dispersion relation is
        det(sum_k a_k (i kappa)^k - i kappa c + sum_q B_q exp(-mu tau_q) - mu) = 0.
    The rightmost eigenvalues converge spectrally to its roots.  The other eigenvalues of the
    matrix approximate nothing.  Shape of the result: (len(kappa), n (M + 1)).
    """
    local = _symbol(a_k, kappa, c)
    n = a_k.shape[1]
    tau_max = float(np.max(taus))
    D = chebyshev(M)[1] * 2 / tau_max
    G = np.zeros((local.shape[0], n * (M + 1), n * (M + 1)), dtype=complex)
    G[:, n:, :] = np.kron(D[1:], np.eye(n))[None]
    G[:, :n, :n] = local
    first = np.einsum("ql,qij->ilj", _history(taus, tau_max, M), B)
    G[:, :n, :] += first.reshape(n, n * (M + 1))[None]
    return np.linalg.eigvals(G)


def _newton_step(local, B, taus, mu):
    """Newton step f / f' of  f(mu) = det(local + sum_q B_q exp(-mu tau_q) - mu)  at the points mu.

    local: (count, n, n), mu: (count,).  Not finite where the exponentials overflow.
    """
    eye = np.eye(local.shape[1])[None]

    def f(m):
        with np.errstate(over="ignore", invalid="ignore"):
            return np.linalg.det(local + np.einsum("aq,qij->aij", np.exp(-m[:, None] * taus[None, :]), B)
                                 - m[:, None, None] * eye)

    step = 1e-6 * (1 + np.abs(mu))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return f(mu) / ((f(mu + step) - f(mu - step)) / (2 * step))


def _rightmost(a_k, B, c, taus, kappa, M, keep, floor):
    """Rightmost root mu of the dispersion relation at every wavenumber: (real parts, roots).

    Candidates: the eigenvalues of the generator (delays `keep` only) with real part above `floor`.
    A candidate counts if one Newton step of the dispersion relation with all delays moves it by
    less than 1 percent.  Without such a candidate the value is `floor`.
    """
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    live = np.linalg.norm(B, axis=(1, 2)) > 0                          # delays that act at this end state
    if not np.any(live & keep):
        values = np.linalg.eigvals(_symbol(a_k, kappa, c))
    else:
        values = _generator(a_k, B[live & keep], c, taus[live & keep], kappa, M)
        row, col = np.nonzero(values.real >= floor)
        mu = values[row, col]
        step = _newton_step(_symbol(a_k, kappa, c)[row], B[live], taus[live], mu)
        good = np.isfinite(step) & (np.abs(step) <= 1e-2 * (1 + np.abs(mu)))
        values = np.full(values.shape, floor - 1.0, dtype=complex)
        values[row[good], col[good]] = mu[good]
    best = values.real.argmax(axis=1)
    roots = values[np.arange(kappa.size), best]
    return np.maximum(roots.real, floor), roots


def dispersion(a_k, B, c, taus, kappa, M=16):
    """Largest real part of the roots of the dispersion relation at the wavenumbers kappa."""
    B = np.asarray(B, dtype=float)
    B, taus = (B[None] if B.ndim == 2 else B), np.atleast_1d(np.asarray(taus, dtype=float))
    rate = float(np.linalg.norm(a_k[0], 2) + np.linalg.norm(B, 2, axis=(1, 2)).sum())
    return _rightmost(a_k, B, c, taus, kappa, M, np.ones(taus.size, dtype=bool), -1e3 * max(rate, 1.0))[0]


def essential_bound(end_minus, end_plus, c, taus, tails=None, kappa_max=1.0e3, samples=1201):
    """Supremum of Re(lambda) over the dispersion curves of the two end states.

    The scan over the wavenumbers uses the Chebyshev generator; with many delays it leaves out
    the last part of a kernel (tails < SCAN_TAIL).  Every candidate is tested on the dispersion
    relation with all delays, and the largest value is corrected by a Newton iteration on it.
    Values below -4 (norm of the zero-order terms) are not resolved: this floor is returned instead.
    """
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    keep = np.ones(taus.size, dtype=bool) if tails is None else np.asarray(tails) >= SCAN_TAIL
    sides = {}
    for name, (a_k, B) in (("left", end_minus), ("right", end_plus)):
        B = np.asarray(B, dtype=float)
        B = B[None] if B.ndim == 2 else B
        rate = float(np.linalg.norm(a_k[0], 2) + np.linalg.norm(B, 2, axis=(1, 2)).sum())
        floor = -4 * max(rate, 1e-300)
        M = 16 if taus.size == 1 else int(np.clip(16 + np.ceil(0.7 * rate * taus[keep].max()), 16, 32))
        count = samples if M == 16 else max(301, int(samples * (17 / (M + 1)) ** 3))
        kappa = np.sinh(np.linspace(0.0, np.arcsinh(kappa_max), count))
        scan = lambda q: _rightmost(a_k, B, c, taus, q, M, keep, floor)
        re = scan(kappa)[0]
        i = int(np.argmax(re))
        lo, hi = kappa[max(i - 1, 0)], kappa[min(i + 1, count - 1)]
        s, where = float(re[i]), float(kappa[i])
        if hi > lo:
            fit = minimize_scalar(lambda q: -scan(q)[0][0], bounds=(lo, hi),
                                  method="bounded", options=dict(xatol=1e-10 * max(1.0, hi)))
            if -fit.fun > s:
                s, where = float(-fit.fun), float(fit.x)
        corrected = False
        live = np.linalg.norm(B, axis=(1, 2)) > 0
        if s > floor and live.any():                   # Newton iteration with all delays
            mu, local = scan(where)[1], _symbol(a_k, where, c)
            for _ in range(30):
                step = _newton_step(local, B[live], taus[live], mu)
                if not np.isfinite(step[0]):
                    break
                mu = mu - step
                if abs(step[0]) <= 1e-13 * (1 + abs(mu[0])):
                    corrected = abs(mu[0].real - s) <= 1e-2 * (1 + abs(mu[0]))
                    break
            if corrected:
                s = float(mu[0].real)
        sides[name] = dict(s=s, kappa=where, chebyshev_points=M + 1, corrected_by_newton=bool(corrected),
                           growing_at_large_kappa=bool(i >= count - 3 and re[-1] > re[-2] > re[-3]))
    return max(v["s"] for v in sides.values()), sides


# ------------------------------------------------------------------------------------------
# Finite differences and the shift on a uniform grid
# ------------------------------------------------------------------------------------------
def _stencils(h, order):
    if order == 2:
        return ({-1: -0.5 / h, 1: 0.5 / h}, {-1: 1 / h ** 2, 0: -2 / h ** 2, 1: 1 / h ** 2})
    return ({-2: 1 / (12 * h), -1: -8 / (12 * h), 1: 8 / (12 * h), 2: -1 / (12 * h)},
            {-2: -1 / (12 * h * h), -1: 16 / (12 * h * h), 0: -30 / (12 * h * h),
             1: 16 / (12 * h * h), 2: -1 / (12 * h * h)})


def _shift(s, order):
    """Weights of f(z_i + s h) = sum_d w[d] f(z_{i+d}), and their derivative with respect to s."""
    j = int(np.floor(s))
    x = s - j
    if order == 2:
        return {j: 1 - x, j + 1: x}, {j: -1.0, j + 1: 1.0}
    w = {j - 1: -x * (x - 1) * (x - 2) / 6, j: (x + 1) * (x - 1) * (x - 2) / 2,
         j + 1: -(x + 1) * x * (x - 2) / 2, j + 2: (x + 1) * x * (x - 1) / 6}
    dw = {j - 1: -(3 * x * x - 6 * x + 2) / 6, j: (3 * x * x - 4 * x - 1) / 2,
          j + 1: -(3 * x * x - 2 * x - 2) / 2, j + 2: (3 * x * x - 1) / 6}
    return w, dw


def _apply(weights, F, left, right):
    """sum_d w[d] F[:, i + d] for a grid function F (rows: components), constant values outside."""
    m = F.shape[1]
    pad = max(abs(d) for d in weights)
    P = np.concatenate((np.repeat(left[:, None], pad, axis=1), F, np.repeat(right[:, None], pad, axis=1)), axis=1)
    out = np.zeros_like(F, dtype=float)
    for d, w in weights.items():
        out += w * P[:, pad + d:pad + d + m]
    return out


def _blocks(system, wave, z, h, order):
    """Blocks of the discretised operators, shape (m, n, n): row node i, column node i + d.

    Returns ({d: A_d}, [{d: B_d} for every delay k]) with
    (L0 p)_i = sum_d A_d[i] p_{i+d},   (B_k p(. + c tau_k))_i = sum_d B_d[i] p_{i+d}.
    """
    Y = wave.Y(z)
    a = system.coefficients(Y, wave.c)
    B = system.delay_matrices(Y, wave.c)
    w1, w2 = _stencils(h, order)
    local = {d: a[2] * w2.get(d, 0.0) + a[1] * w1.get(d, 0.0) + (a[0] if d == 0 else 0.0)
             for d in sorted(set(w1) | set(w2) | {0})}
    delayed = [{d: B[k] * w for d, w in _shift(wave.c * tau / h, order)[0].items() if w != 0.0}
               for k, tau in enumerate(system.taus)]
    return local, delayed


def _sparse(blocks, m, n):
    """Sparse matrix (node-major ordering) from the blocks {d: (m, n, n)}."""
    rows, cols, data = [], [], []
    p, q = np.arange(n)[None, :, None], np.arange(n)[None, None, :]
    for d, block in blocks.items():
        i = np.arange(max(0, -d), min(m, m - d))
        if i.size == 0:
            continue
        rows.append(((i * n)[:, None, None] + p + 0 * q).ravel())
        cols.append((((i + d) * n)[:, None, None] + q + 0 * p).ravel())
        data.append(block[i].ravel())
    if not data:
        return sparse.csc_matrix((m * n, m * n))
    return sparse.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(m * n, m * n))


# ------------------------------------------------------------------------------------------
# The wave
# ------------------------------------------------------------------------------------------
def _newton(system, z, U, c, scale, left, right, ref, dref, tol=1e-11, max_iter=40):
    """Solve the discretised profile problem with a phase condition; every delay is multiplied by `scale`.

        G(U, U', U'', U(. + c scale tau_1), ...) + c U' = 0.
    """
    local, n, m = system.local, system.n, z.size
    h = z[1] - z[0]
    off = system.offset
    w1, w2 = _stencils(h, 4)
    uL, uR = np.asarray(left, float), np.asarray(right, float)
    p, q = np.arange(n)[None, :, None], np.arange(n)[None, None, :]
    taus = scale * system.taus
    component = system._component
    delay = np.array([k for _, k in system.pairs])

    def parts(U, c):
        dU, ddU = _apply(w1, U, uL, uR), _apply(w2, U, uL, uR)       # end-state values outside the grid
        shifts = [_shift(c * tau / h, 4) for tau in taus]
        moved = np.array([_apply(ws, U, uL, uR) for ws, _ in shifts])                    # (Q, n, m)
        slope = np.array([_apply(dws, U, uL, uR) * tau / h for (_, dws), tau in zip(shifts, taus)])
        Y = np.empty((system.N0, m))
        Y[off], Y[off + 1] = U, dU
        return Y, dU, ddU, moved[delay, component], slope[delay, component], shifts      # V and dV / dc

    def residual(U, c):
        Y, dU, ddU, V, dV, shifts = parts(U, c)
        theta = system.theta(V)
        H = local._H(*Y, c, *theta).reshape(n, n, m)
        E = np.einsum("ijm,jm->im", H, ddU) + local._r(*Y, c, *theta)
        phase = h * ((U - ref) * dref).sum()
        return np.concatenate((E.T.ravel(), [phase])), (Y, dU, ddU, V, dV, shifts, theta)

    for iteration in range(max_iter):
        F, (Y, dU, ddU, V, dV, shifts, theta) = residual(U, c)
        norm = np.abs(F).max()
        if norm < tol:
            return U, c, iteration, norm
        a = local._a_k(*Y, *ddU, c, *theta).reshape(3, n, n, m)
        keep, local.theta = local.theta, theta
        try:
            B = system.pair_matrices(Y, ddU, c, V)
        finally:
            local.theta = keep
        blocks = {d: a[2] * w2.get(d, 0.0) + a[1] * w1.get(d, 0.0) + (a[0] if d == 0 else 0.0)
                  for d in set(w1) | set(w2) | {0}}
        column = system.mass[:, None] * dU
        for index, (j, k) in enumerate(system.pairs):
            for d, w in shifts[k][0].items():
                if w != 0.0:
                    blocks.setdefault(d, np.zeros((n, n, m)))[:, j, :] += w * B[:, index, :]
            column[:] += B[:, index, :] * dV[index]
        rows, cols, data = [], [], []
        for d, block in blocks.items():
            i = np.arange(max(0, -d), min(m, m - d))
            if i.size == 0:
                continue
            rows.append(((i * n)[:, None, None] + p + 0 * q).ravel())
            cols.append((((i + d) * n)[:, None, None] + q + 0 * p).ravel())
            data.append(block[:, :, i].transpose(2, 0, 1).ravel())
        size = m * n
        rows += [np.arange(size), np.full(size, size)]
        cols += [np.full(size, size), np.arange(size)]
        data += [column.T.ravel(), (h * dref).T.ravel()]
        J = sparse.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(size + 1, size + 1))
        step = spsolve(J, -F)
        if not np.all(np.isfinite(step)):
            raise WaveError("the Newton iteration of the delayed profile problem broke down")
        t = 1.0
        while t > 1 / 128:
            U_new, c_new = U + t * step[:-1].reshape(m, n).T, c + t * step[-1]
            if np.abs(residual(U_new, c_new)[0]).max() < (1 - 1e-4 * t) * norm:
                break
            t /= 2
        else:
            raise WaveError("the Newton iteration of the delayed profile problem stalled")
        U, c = U_new, c_new
    raise WaveError("the Newton iteration of the delayed profile problem did not converge")


def solve_wave(problem, system, tol=1e-8, check_tol=5e-10, tail_tol=1e-9, L_max=2000.0, h_factor=0.04,
               max_nodes=40000):
    """Wave of the delayed problem by continuation in the size of the delays, from the problem without delay."""
    spec = problem.wave
    left, right = np.asarray(spec["left"], float), np.asarray(spec["right"], float)
    base = solve_local_wave(system.problem0, system.system0, tol=tol, check_tol=check_tol,
                            tail_tol=tail_tol, L_max=L_max, max_nodes=max_nodes)
    if not base.info["speed_is_unknown"]:
        raise WaveError("with a delay, only waves with an isolated speed are supported")
    off = system.offset
    rate = max(np.abs(np.linalg.eigvals(system.system0.end_state(u, base.c)[0])).max() for u in (left, right))

    def continuation(h, L):
        z = np.arange(-L + h, L - h / 2, h)
        Y0 = base.Y(z)
        U, c = Y0[off].copy(), base.c
        ref, dref = Y0[off].copy(), Y0[off + 1].copy()               # phase reference: the wave without delay
        scale, step, steps = 0.0, 0.25, 0
        while scale < 1 - 1e-12:
            target = min(scale + step, 1.0)
            try:
                U, c, _, _ = _newton(system, z, U, c, target, left, right, ref, dref)
                scale, steps = target, steps + 1
            except WaveError:
                step /= 2
                if step < 1 / 4096:
                    raise WaveError("the continuation in the delay failed near "
                                    f"{scale:.4g} times the delay(s)") from None
        return z, U, c, steps

    h = h_factor / rate
    L = base.L
    for extension in range(4):
        z, U, c, steps = continuation(h, L)
        edge = max(int(0.02 * z.size), 3)
        tails = max(np.abs(U[:, :edge] - left[:, None]).max(), np.abs(U[:, -edge:] - right[:, None]).max())
        scale = max(1.0, float(np.abs(U).max()))
        if tails < 1e3 * tail_tol * scale or L >= L_max:
            break
        L = min(1.5 * L, L_max)
    if tails > 1e3 * tail_tol * scale:
        raise WaveError(f"the profile has not reached its end states at the ends of the domain "
                        f"(distance {tails:.2g}, L = {L:.4g})")
    z2, U2, c2, _ = continuation(2 * h, L)                             # error estimate: twice the step
    grid = np.concatenate(([-L], z, [L]))
    spline = CubicSpline(grid, np.hstack((left[:, None], U, right[:, None])), axis=1)
    coarse = CubicSpline(np.concatenate(([-L], z2, [L])), np.hstack((left[:, None], U2, right[:, None])), axis=1)
    probe = np.linspace(-L, L, 4001)
    profile_error = float(np.abs(spline(probe) - coarse(probe)).max()) / 15     # fourth-order scheme
    if spec["kind"] == "pulse" and np.abs(U - left[:, None]).max() < 1e-6 * scale:
        raise WaveError("the solution collapsed to the constant state")
    taus, N0 = system.taus, system.N0
    rows = [[index for index, (_, k) in enumerate(system.pairs) if k == delay] for delay in range(taus.size)]
    components = [system._component[r] for r in rows]

    def interpolant(zz):
        zz = np.atleast_1d(np.asarray(zz, dtype=float))
        Y = np.empty((system.N, zz.size))
        Y[off], Y[off + 1] = spline(zz), spline(zz, 1)
        for k, tau in enumerate(taus):
            shifted = zz + c * tau
            V = spline(np.clip(shifted, -L, L))
            V[:, shifted > L] = right[:, None]
            V[:, shifted < -L] = left[:, None]
            Y[N0 + np.array(rows[k], dtype=int)] = V[components[k]]
        return Y

    info = dict(speed_is_unknown=True, speed_without_delay=base.c, delay=system.tau, delays=int(taus.size),
                continuation_steps=int(steps), nodes=int(z.size), grid_step=float(h),
                speed_error=abs(c - c2) / 15, profile_error=profile_error, tail_error=float(tails),
                started_from="continuation in the delay")
    return Wave(c=float(c), L=float(L), N=system.N, Y_minus=system.state(left), Y_plus=system.state(right),
                interpolant=interpolant, mesh=grid, info=info, theta=())


# ------------------------------------------------------------------------------------------
# Point spectrum: determinant of the banded matrix T(lambda)
# ------------------------------------------------------------------------------------------
class BandDeterminant:
    """log|D|, phase of the normalised determinant of  T(lambda) = lambda - A - sum_k exp(-lambda tau_k) B_k.

    A, B_k are banded matrices given by their blocks {offset d: block}, shape (m, n, n):
    row node i, column node i + d, zero values outside the grid.  det T grows like the
    number of nodes.  It is divided by the factor that the constant-coefficient problems of
    the two end states give for the same number of nodes (Szego limit):

        exp( m_- phi_-(lambda) + m_+ phi_+(lambda) ),
        phi(lambda) = mean over theta of  log det(lambda - a(theta) - sum_k exp(-lambda tau_k) b_k(theta)),

    a, b_k the symbols of the stencils at the end state.  This factor is analytic and has no
    zeros to the right of the (discrete) essential spectrum, so it does not change the
    count.  It plays the role of the cell factors of the Evans engine.
    D is real on the real axis.

    blocks: {d: A_d};  delays: [(tau_k, {d: B_d})];
    sides : [(number of nodes, a(theta), b(theta) with shape (Q, THETA, n, n) or None), ...].
    """

    THETA = 1024                                    # quadrature points of the symbol

    def __init__(self, n, m, blocks, delays, sides):
        self.n, self.sides = n, sides
        self.taus = np.array([tau for tau, _ in delays], dtype=float)
        self.cells, self.size = m, m * n
        offsets = sorted(set(blocks).union(*[set(parts) for _, parts in delays]))
        self.kl = max(-offsets[0], 0) * n + n - 1
        self.ku = max(offsets[-1], 0) * n + n - 1
        base = self.base = self.kl + self.ku
        self.shape = (2 * self.kl + self.ku + 1, self.size)
        p, q = np.arange(n)[None, :, None], np.arange(n)[None, None, :]

        def entries(d):
            """Nodes i with a column i + d inside the grid, and the flat positions of their blocks."""
            i = np.arange(max(0, -d), min(m, m - d))
            rows = (i * n)[:, None, None] + p + 0 * q
            cols = ((i + d) * n)[:, None, None] + q + 0 * p
            return i, ((base + rows - cols) * self.size + cols).ravel()

        self.ab_A = np.zeros(self.shape[0] * self.size, dtype=complex)
        for d, A_d in blocks.items():
            i, flat = entries(d)
            if i.size:
                self.ab_A[flat] -= A_d[i].ravel()
        position, column, data = [], [], []
        for k, (_, parts) in enumerate(delays):
            for d, B_d in parts.items():
                i, flat = entries(d)
                if i.size:
                    position.append(flat)
                    column.append(np.full(flat.size, k))
                    data.append(-B_d[i].ravel())
        self.delay_map = None                      # (band entries) x (delays): entries of -B_k
        if data:
            self.delay_map = sparse.csr_matrix(
                (np.concatenate(data).astype(complex), (np.concatenate(position), np.concatenate(column))),
                shape=(self.ab_A.size, len(delays)))
        self.diagonal = base * self.size + np.arange(self.size)
        self.eye = np.eye(n)[None]
        self.cache = {}

    @classmethod
    def theta(cls):
        return 2 * np.pi * np.arange(cls.THETA) / cls.THETA

    def _normaliser(self, lam, weights):
        """m_- phi_- + m_+ phi_+ as a complex number (real part: log modulus, imaginary part: angle)."""
        total = 0j
        for count, a_sym, b_sym in self.sides:
            matrix = lam * self.eye - a_sym
            if b_sym is not None:
                matrix = matrix - np.tensordot(weights, b_sym, axes=1)
            det = np.linalg.det(matrix)
            if np.abs(det).min() == 0:
                raise ArithmeticError("lambda lies on the discrete essential spectrum")
            angle = np.unwrap(np.angle(np.concatenate((det, det[:1]))))
            if abs(angle[-1] - angle[0]) > 1e-6:
                raise ArithmeticError(f"lambda = {lam:.6g} is inside the essential spectrum of the difference scheme")
            total += count * (np.log(np.abs(det)).mean() + 1j * angle[:-1].mean())
        return total

    def __call__(self, lam):
        lam = complex(lam)
        hit = self.cache.get(lam)
        if hit is not None:
            return hit
        weights = np.exp(-lam * self.taus)
        ab = self.ab_A.copy()
        if self.delay_map is not None:
            ab += self.delay_map @ weights
        ab[self.diagonal] += lam
        lu, piv, info = zgbtrf(np.asfortranarray(ab.reshape(self.shape)), self.kl, self.ku, overwrite_ab=True)
        if info < 0:
            raise ArithmeticError(f"band LU: illegal argument {-info}")
        if info > 0:
            value = (-np.inf, 0j, 1.0 + 0j)
        else:
            diag = lu[self.base]
            norm = self._normaliser(lam, weights)
            angle = np.angle(diag).sum() + np.pi * np.count_nonzero(piv != np.arange(self.size)) - norm.imag
            value = (float(np.log(np.abs(diag)).sum() - norm.real), complex(np.exp(1j * angle)), 1.0 + 0j)
        self.cache[lam] = value
        return value


def DelayEngine(system, wave, La, Lb, h, order=4):
    """Band determinant of the delay problem on a uniform grid of [-La, Lb] with step close to h."""
    m = max(8, int(round((La + Lb) / h)) - 1)
    z = np.linspace(-La, Lb, m + 2)[1:-1]
    h = z[1] - z[0]
    local, delayed = _blocks(system, wave, z, h, order)
    theta = BandDeterminant.theta()
    w1, w2 = _stencils(h, order)
    symbol = lambda w: sum(v * np.exp(1j * d * theta) for d, v in w.items())
    s1, s2 = symbol(w1), symbol(w2)
    shifts = np.array([symbol(_shift(wave.c * tau / h, order)[0]) for tau in system.taus])     # (Q, THETA)
    sides = []
    for Y, count in ((wave.Y_minus, int((z < 0).sum())), (wave.Y_plus, int((z >= 0).sum()))):
        a = system.coefficients(Y, wave.c)[:, 0]
        B = system.delay_matrices(Y, wave.c)[:, 0]
        sides.append((count, a[2][None] * s2[:, None, None] + a[1][None] * s1[:, None, None] + a[0][None],
                      B[:, None] * shifts[:, :, None, None]))
    return BandDeterminant(system.n, m, local, list(zip(system.taus, delayed)), sides)


def generator_matrix(system, wave, La, Lb, nz, M=None):
    """Independent discretisation: second-order differences in z, Chebyshev points in the delay.

    The unknowns are p(z_i, t + theta_l), theta_l in [-tau_max, 0].  The delayed values
    p(., t - tau_k) come from the Chebyshev interpolant.  The eigenvalues of this sparse
    matrix approximate the eigenvalues of the delay problem; no exponential of lambda appears.
    With a distributed delay the last part of the kernel (weight SCAN_TAIL) is left out.
    """
    keep = np.nonzero(system.tails >= SCAN_TAIL)[0]
    taus = system.taus[keep]
    if M is None:
        M = 12 if system.taus.size == 1 else 16
    z = np.linspace(-La, Lb, nz + 2)[1:-1]
    n = system.n
    local, delayed = _blocks(system, wave, z, z[1] - z[0], 2)
    basis = _history(taus, taus.max(), M)                              # (Q, M + 1)
    first = []
    for l in range(M + 1):
        combined = dict(local) if l == 0 else {}
        for row, k in enumerate(keep):
            if basis[row, l] != 0.0:
                for d, block in delayed[k].items():
                    combined[d] = combined[d] + basis[row, l] * block if d in combined else basis[row, l] * block
        first.append(_sparse(combined, nz, n) if combined else None)
    D = chebyshev(M)[1] * 2 / taus.max()
    eye = sparse.identity(nz * n, format="csc")
    rows = [first] + [[D[k, l] * eye for l in range(M + 1)] for k in range(1, M + 1)]
    return sparse.bmat(rows, format="csc")


# ------------------------------------------------------------------------------------------
# Interface used by label.py (the same names exist in kernel.py)
# ------------------------------------------------------------------------------------------
System = DelaySystem
engine = DelayEngine
probe_matrix = generator_matrix


def details(system):
    out = dict(delay=system.tau, delays=int(system.taus.size), delayed_components=system.delayed)
    if system.integrals:
        out.update(distributed_delays=system.integrals)
    return out


def rate(system, spec, wave):
    """Largest spatial rate of the local part at the end states; it sets the first step of the spectral grids."""
    return max(np.abs(np.linalg.eigvals(system.system0.end_state(spec[s], wave.c)[0])).max()
               for s in ("left", "right"))


def core_spectrum(system, wave):
    """No essential spectrum from the core of the wave (kernel.py: possible without diffusion)."""
    return None


def region(system, wave, delta):
    """No search rectangle of its own: the energy estimate applies (kernel.py: not without diffusion)."""
    return None


def min_step(system, wave):
    """Smallest coarse step of the spectral grids: it keeps the band of the shifted terms narrow."""
    return abs(wave.c) * system.tau / BAND


def delta_cap(system):
    """Largest admissible delta: exp(-lambda s) g(s) must stay integrable for Re(lambda) >= -delta."""
    return 0.5 * system.decay


def resolution(system, y_max, c):
    """None, or the frequency for which the quadrature of the distributed delays must be built again.

    The quadrature must be exact for exp(-(lambda - i kappa c) s) with |Im lambda| <= y_max and
    the wavenumbers kappa of the eigenfunctions.  y_max contains the term |c| sup(kappa).
    """
    needed = y_max * (2.0 if abs(c) * system.tau > 1e-8 else 1.0)
    if not system.integrals or needed <= system.frequency * (1 + 1e-9):
        return None
    return 1.25 * needed


def end_bounds(system, spec, c):
    """Essential spectrum: (s_ess, sides, scale of the zero-order coefficients)."""
    ends = [system.end_state(spec[side], c) for side in ("left", "right")]
    s_ess, sides = essential_bound(ends[0], ends[1], c, system.taus, system.tails)
    return s_ess, sides, max(np.linalg.norm(ends[0][0][0], 2), np.linalg.norm(ends[1][0][0], 2), 1e-300)


def coupling(system, wave, delta):
    """(sup of sum_k ||B_k||, constant added to the energy bound): |exp(-lambda tau)| <= exp(delta tau)."""
    z = np.linspace(-wave.L, wave.L, 4001)
    norms = np.linalg.norm(system.delay_matrices(wave.Y(z), wave.c), 2, axis=(2, 3))          # (Q, m)
    return (float(norms.sum(axis=0).max()),
            lambda d: float((np.exp(d * system.taus)[:, None] * norms).sum(axis=0).max()))

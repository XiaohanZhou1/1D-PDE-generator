"""Sub-solver N: convolution terms with a general kernel (DESIGN.md, sections 2.10 and 3.8).

    u_t = G(u, u_x, u_xx, V),      V_k(x, t) = int K_k(x - y) g_k(u(y, t)) dy,
or, without diffusion (neural fields, nonlocal phase-transition models),
    u_t = G(u, V).

A kernel with a rational Fourier transform can be removed by an auxiliary unknown
(section 2.7).  A general kernel cannot: the eigenvalue problem

    lambda p = L0 p + B(z) int K(z - y) g'(U(y)) p(y) dy

is an integro-differential equation.  This module uses the band-determinant engine
of delay.py: the kernel is cut off where it is negligible, so that the discretised
convolution is a banded matrix.

* KernelSystem     symbolic reduction; the convolutions are extra arguments
* end_bounds       essential spectrum with the Fourier transform of the kernels
* solve_wave       Newton iteration on a uniform grid; homotopy from the local limit
                   (every kernel replaced by its mass times a delta function)
* region           search rectangle without diffusion (numerical range, resolvent estimate)
* engine           band determinant of  lambda - A - C
* probe_matrix     independent discretisation (second-order differences) for the probe
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import spsolve
from sympy.core.function import AppliedUndef

from .delay import BandDeterminant, _apply, _stencils
from .problem import Problem
from .symbolic import LocalSystem, StructureError, _vectorised
from .wave import Wave, WaveError, end_data, reference_from_guess
from .wave import solve_wave as solve_local_wave

CUTOFF = 1e-10                # the kernel is neglected where |K| < CUTOFF max|K|
MAX_RADIUS = 200.0
BAND = 40                     # the coarsest Evans-type grid has at most this many nodes inside the kernel radius
GAUSS = np.polynomial.legendre.leggauss(8)


def find_kernels(problem):
    """Convolution integrals of the equations.  Returns None or a list of descriptions."""
    x, t = problem.x, problem.t
    funcs = [u.func for u in problem.unknowns]
    found = []
    for eq in problem.equations:
        for integral in sp.sympify(eq.lhs - eq.rhs).atoms(sp.Integral):
            if any(integral == k["atom"] for k in found):
                continue
            if len(integral.limits) != 1 or len(integral.limits[0]) != 3:
                raise StructureError("an integral must have one variable and two limits")
            y, lower, upper = integral.limits[0]
            if lower != -sp.oo or upper != sp.oo:
                raise StructureError("an integral must extend over the whole line")
            body = integral.function
            atoms = body.atoms(AppliedUndef)
            places = {f.args[0] for f in atoms if len(f.args) == 2}
            if (not atoms or len(places) != 1
                    or any(f.func not in funcs or len(f.args) != 2 or f.args[1] != t for f in atoms)):
                raise StructureError("an integrand may contain the unknowns at one place (y, t) only")
            # The unknowns are taken at xi(x, y); write xi = x - s and integrate over s.
            s = sp.Dummy("s", real=True)
            solutions = sp.solve(sp.Eq(places.pop(), x - s), y)
            if len(solutions) != 1 or sp.simplify(sp.diff(solutions[0], s) ** 2 - 1) != 0:
                raise StructureError("the unknowns in an integrand must be taken at y or at x - y")
            W = {f: sp.Dummy(f"W{funcs.index(f.func)}", real=True) for f in atoms}
            F = body.xreplace(W).subs(y, solutions[0])
            if F.has(x):
                F = sp.simplify(F)
            if F.has(x) or F.has(t):
                raise StructureError("an integrand must depend on x - y only (convolution)")
            kernel, g = F.as_independent(*W.values(), as_Add=False)
            if g.has(s) or kernel.has(*W.values()):        # factor() only if needed: it can give a form that overflows
                kernel, g = sp.factor(F).as_independent(*W.values(), as_Add=False)
            if g.has(s) or kernel.has(*W.values()):
                raise StructureError("an integrand must be a product K(x - y) g(u(y, t))")
            found.append(dict(atom=integral, s=s, kernel=kernel, g=g,
                              inner={sym: funcs.index(f.func) for f, sym in W.items()}))
    return found or None


class KernelSystem:
    """Local functions of (U, U', U'', V) with V_k = the convolution K_k * g_k(U)."""

    def __init__(self, problem):
        found = find_kernels(problem)
        us, n = problem.unknowns, len(problem.unknowns)
        Vsym = [sp.Symbol(f"convolution_{k}", real=True) for k in range(len(found))]
        self.kernels, self.kernel_mass, self.radius, self.norm1 = [], [], [], []
        self.length, self.variation = [], []          # root mean square width and total variation of a kernel
        self._weights = {}
        local_limit, to_symbol = {}, {}
        for item, V in zip(found, Vsym):
            K = sp.lambdify(item["s"], item["kernel"].subs(problem.parameters), "numpy")
            kernel = lambda v, K=K: np.broadcast_to(np.asarray(K(np.asarray(v, dtype=float)), dtype=float),
                                                    np.shape(v)).copy()
            grid = np.linspace(-MAX_RADIUS, MAX_RADIUS, 400001)
            values = np.abs(kernel(grid))
            if not np.all(np.isfinite(values)):
                raise StructureError("a kernel is not finite on the real line")
            significant = np.nonzero(values > CUTOFF * values.max())[0]
            radius = float(max(abs(grid[significant[0]]), abs(grid[significant[-1]]))) + 1e-3
            if radius > 0.9 * MAX_RADIUS:
                raise StructureError("a kernel decays too slowly (no cut-off radius below 180)")
            mass = quad(kernel, -radius, 0.0, limit=200)[0] + quad(kernel, 0.0, radius, limit=200)[0]
            self.kernels.append(kernel)
            self.radius.append(radius)
            self.kernel_mass.append(mass)
            self.norm1.append(float(np.trapezoid(values, grid)))
            self.length.append(float(np.sqrt(np.trapezoid(grid ** 2 * values, grid) / self.norm1[-1])))
            self.variation.append(float(np.abs(np.diff(kernel(grid))).sum()))
            to_symbol[item["atom"]] = V
            local_limit[item["atom"]] = mass * item["g"].xreplace({sym: us[j] for sym, j in item["inner"].items()})

        def replaced(mapping, parameters):
            equations = [sp.Eq(eq.lhs.xreplace(mapping), eq.rhs.xreplace(mapping)) for eq in problem.equations]
            return Problem(path=problem.path, name=problem.name, x=problem.x, t=problem.t, equations=equations,
                           unknowns=us, parameters=parameters, wave=problem.wave, options=problem.options)

        # Local limit: every kernel becomes its mass times a delta function.  It starts the homotopy.
        self.problem0 = replaced(local_limit, problem.parameters)
        self.system0 = LocalSystem(self.problem0)
        self.local = local = LocalSystem(replaced(to_symbol, {**problem.parameters, **{V: 0.0 for V in Vsym}}))
        orders = local.orders
        if np.any(local.mass == 0) or not (all(m == 2 for m in orders) or all(m == 0 for m in orders)):
            raise StructureError("with a convolution, either every component has a diffusion term or the "
                                 "equations contain no x-derivative at all (and there is no constraint)")
        self.diffusive = local.K == 2                                 # False: u_t = G(u, V)
        self.n, self.K, self.N0, self.offset = n, local.K, local.N, local.offset
        self.count = len(found)
        self.N = local.N + self.count                                 # extended state: jets and convolutions
        self.first_order_components = []
        self.mass = local.mass                                        # mass-matrix entries, all 1
        self._index = [local.parameter_index(V) for V in Vsym]
        sym = local.symbols
        self._outer = _vectorised(sym["low"] + sym["T"] + [sym["c"]] + local.params,
                                  [sp.diff(g, V) for g in sym["G"] for V in Vsym])
        Wall = [sp.Dummy(f"U{j}", real=True) for j in range(n)]
        g_exprs = [item["g"].subs(problem.parameters).xreplace({sym_: Wall[j] for sym_, j in item["inner"].items()})
                   for item in found]
        self._g = _vectorised(Wall, g_exprs)
        self._dg = _vectorised(Wall, [sp.diff(e, w) for e in g_exprs for w in Wall])

    def theta(self, V):
        theta = list(self.local.theta)
        for k, i in enumerate(self._index):
            theta[i] = V[k]
        return tuple(theta)

    def g(self, U):
        """g_k(U), shape (count, m)."""
        return self._g(*np.asarray(U, dtype=float).reshape(self.n, -1))

    def inner(self, U):
        """dg_k/dU_j, shape (m, count, n)."""
        U = np.asarray(U, dtype=float).reshape(self.n, -1)
        return self._dg(*U).reshape(self.count, self.n, U.shape[1]).transpose(2, 0, 1)

    def state(self, u):
        """Extended state of the constant solution u: the convolutions are mass * g(u)."""
        u = np.asarray(u, dtype=float)
        return np.concatenate((self.local.state(u), np.array(self.kernel_mass) * self.g(u)[:, 0]))

    def equilibrium_residual(self, u):
        return self.system0.equilibrium_residual(u)

    def _split(self, Y):
        Y = np.asarray(Y, dtype=float).reshape(self.N, -1)
        return Y[:self.N0], Y[self.N0:]

    def _top(self, Y0, c):
        """Highest derivatives for the compiled functions.

        Without diffusion E = G(U, V) + c U': the coefficients do not depend on U', and the
        profile equation cannot be solved for U' when c = 0.  Zeros are passed.
        """
        return self.local._solve_top(Y0, c)[1] if self.diffusive else np.zeros_like(Y0)

    def coefficients(self, Y, c):
        """Coefficients a^(k) of the local part L0 at extended states, shape (K+1, m, n, n)."""
        Y0, V = self._split(Y)
        keep, self.local.theta = self.local.theta, self.theta(V)
        try:
            values = self.local._a_k(*Y0, *self._top(Y0, c), c, *self.local.theta)
        finally:
            self.local.theta = keep
        return values.reshape(self.K + 1, self.n, self.n, Y0.shape[1]).transpose(0, 3, 1, 2)

    def outer(self, Y, c):
        """dG/dV, shape (m, n, count)."""
        Y0, V = self._split(Y)
        keep, self.local.theta = self.local.theta, self.theta(V)
        try:
            values = self._outer(*Y0, *self._top(Y0, c), c, *self.local.theta)
        finally:
            self.local.theta = keep
        return values.reshape(self.n, self.count, Y0.shape[1]).transpose(2, 0, 1)

    def weights(self, h, k):
        """Weights w[d] of  (K_k * f)(z_i) = sum_d w[d] f(z_{i+d})  on a grid of step h.

        Product integration of fourth order: on every grid interval f is replaced by its
        cubic interpolant through the four nearest nodes, and the kernel is integrated
        exactly against the interpolation polynomials (Gauss rule on each interval).
        A kink or a jump of the kernel at a grid point does no harm.  The weights are
        scaled to the exact mass of the kernel.
        """
        key = (round(h, 14), k)
        if key not in self._weights:
            half = int(np.ceil(self.radius[k] / h)) + 1
            j = np.arange(-half, half)                                # intervals [j h, (j + 1) h]
            nodes, gauss = GAUSS
            x = (nodes + 1) / 2                                       # Gauss points in [0, 1]
            lagrange = {-1: -x * (x - 1) * (x - 2) / 6, 0: (x + 1) * (x - 1) * (x - 2) / 2,
                        1: -(x + 1) * x * (x - 2) / 2, 2: (x + 1) * x * (x - 1) / 6}
            values = self.kernels[k](-(j[:, None] + x[None, :]) * h)  # K(z_i - y), y = z_i + (j + x) h
            d = np.arange(-half - 1, half + 2)
            w = np.zeros(d.size)
            for shift, basis in lagrange.items():
                w[j + shift + half + 1] += (h / 2) * (values * (basis * gauss)[None, :]).sum(axis=1)
            total = w.sum()
            if abs(total) > 1e-12 * np.abs(w).sum():
                w *= self.kernel_mass[k] / total
            self._weights[key] = (d, w)
        return self._weights[key]

    def fourier(self, k, kappa):
        """Fourier transform int K_k(s) exp(-i kappa s) ds; set to 0 where the quadrature cannot resolve it."""
        R = self.radius[k]
        s = np.linspace(-R, R, 4001)
        values = self.kernels[k](s)
        out = np.zeros(kappa.size, dtype=complex)
        ok = kappa * (s[1] - s[0]) < 0.5
        for chunk in np.array_split(np.nonzero(ok)[0], max(1, ok.sum() // 200)):
            out[chunk] = np.trapezoid(values[None, :] * np.exp(-1j * kappa[chunk][:, None] * s[None, :]), s, axis=1)
        return out


# ------------------------------------------------------------------------------------------
# Essential spectrum
# ------------------------------------------------------------------------------------------
def end_bounds(system, spec, c, kappa_max=1.0e4, samples=3001):
    """Supremum of Re(lambda) on  det(sum_k a_k (i kappa)^k + sum_j B_j Khat_j(kappa) g_j' - lambda) = 0."""
    kappa = np.sinh(np.linspace(0.0, np.arcsinh(kappa_max), samples))
    sides, scale = {}, 1e-300

    def symbol(u, q):
        Y = system.state(u)
        a = system.coefficients(Y, c)[:, 0]
        outer, inner = system.outer(Y, c)[0], system.inner(u)[0]
        q = np.atleast_1d(q)
        powers = (1j * q[:, None]) ** np.arange(a.shape[0])[None, :]
        matrix = np.einsum("lk,kij->lij", powers, a)
        for j in range(system.count):
            matrix = matrix + system.fourier(j, q)[:, None, None] * np.outer(outer[:, j], inner[j])[None]
        return matrix, np.linalg.norm(a[0], 2)

    for name in ("left", "right"):
        u = np.asarray(spec[name], dtype=float)
        matrix, norm = symbol(u, kappa)
        scale = max(scale, norm)
        re = np.linalg.eigvals(matrix).real.max(axis=1)
        # Without diffusion the curves stay bounded: Khat -> 0, so lambda -> i kappa c + eig(a0).
        limit = None
        if not system.diffusive:
            limit = float(np.linalg.eigvals(system.coefficients(system.state(u), c)[0, 0]).real.max())
        i = int(np.argmax(re))
        lo, hi = kappa[max(i - 1, 0)], kappa[min(i + 1, samples - 1)]
        s, where = float(re[i]), float(kappa[i])
        if hi > lo:
            fit = minimize_scalar(lambda q: -np.linalg.eigvals(symbol(u, q)[0]).real.max(), bounds=(lo, hi),
                                  method="bounded", options=dict(xatol=1e-10 * max(1.0, hi)))
            if -fit.fun > s:
                s, where = float(-fit.fun), float(fit.x)
        growing = bool(i >= samples - 3 and re[-1] > re[-2] > re[-3])
        if limit is not None:
            if re[-1] <= limit + 1e-6 * max(1.0, abs(limit)):
                growing = False                        # a curve that approaches its limit from below
            if limit > s:
                s, where = limit, np.inf
        sides[name] = dict(s=s, kappa=where, growing_at_large_kappa=growing, limit_at_infinity=limit)
    return max(v["s"] for v in sides.values()), sides, scale


# ------------------------------------------------------------------------------------------
# Discretisation on a uniform grid
# ------------------------------------------------------------------------------------------
def _convolutions(system, U, h, left, right):
    """V_k = sum_d w_k[d] g_k(U_{i+d}) with the end states outside the grid, shape (count, m)."""
    g, gL, gR = system.g(U), system.g(left)[:, 0], system.g(right)[:, 0]
    out = np.empty_like(g)
    for k in range(system.count):
        d, w = system.weights(h, k)
        out[k] = _apply(dict(zip(d.tolist(), w.tolist())), g[k:k + 1], gL[k:k + 1], gR[k:k + 1])[0]
    return out


def _local_blocks(a, h, order):
    """{offset d: block (m, n, n)} of  sum_k a^(k) D_k;  a has the shape (K + 1, m, n, n), K = 1 or 2."""
    stencils = ({0: 1.0},) + _stencils(h, order)
    return {d: sum(a[k] * stencils[k].get(d, 0.0) for k in range(a.shape[0]))
            for d in sorted(set().union(*stencils[:a.shape[0]]))}


def _blocks(system, wave, z, h, order):
    """{offset d: block (m, n, n)} of  L0 + sum_k outer_k w_k[d] inner_k(node i + d)."""
    Y = wave.Y(z)
    m = z.size
    a = system.coefficients(Y, wave.c)
    outer, inner = system.outer(Y, wave.c), system.inner(Y[system.offset])
    blocks = _local_blocks(a, h, order)
    for k in range(system.count):
        offsets, w = system.weights(h, k)
        for d, weight in zip(offsets.tolist(), w.tolist()):
            if weight == 0.0 or abs(d) >= m:
                continue
            i = np.arange(max(0, -d), min(m, m - d))
            block = blocks.setdefault(d, np.zeros((m, system.n, system.n)))
            block[i] += weight * outer[i][:, :, k, None] * inner[i + d][:, None, k, :]
    return blocks


def engine(system, wave, La, Lb, h, order=4):
    """Band determinant of  lambda - A - C  on a uniform grid of [-La, Lb] with step close to h."""
    m = max(8, int(round((La + Lb) / h)) - 1)
    z = np.linspace(-La, Lb, m + 2)[1:-1]
    h = z[1] - z[0]
    blocks = _blocks(system, wave, z, h, order)
    theta = BandDeterminant.theta()
    symbols = [sum(v * np.exp(1j * d * theta) for d, v in w.items()) for w in ({0: 1.0},) + _stencils(h, order)]
    sides = []
    for Y, count in ((wave.Y_minus, int((z < 0).sum())), (wave.Y_plus, int((z >= 0).sum()))):
        a = system.coefficients(Y, wave.c)[:, 0]
        outer, inner = system.outer(Y, wave.c)[0], system.inner(Y[system.offset])[0]
        total = sum(a[k][None] * symbols[k][:, None, None] for k in range(a.shape[0]))
        for k in range(system.count):
            offsets, w = system.weights(h, k)
            kernel_symbol = (w[None, :] * np.exp(1j * theta[:, None] * offsets[None, :])).sum(axis=1)
            total = total + kernel_symbol[:, None, None] * np.outer(outer[:, k], inner[k])[None]
        sides.append((count, total, None))
    return BandDeterminant(system.n, m, blocks, [], sides)


def probe_matrix(system, wave, La, Lb, nz):
    """Sparse matrix of  L0 + C  with second-order differences: an ordinary eigenvalue problem."""
    z = np.linspace(-La, Lb, nz + 2)[1:-1]
    n = system.n
    rows, cols, data = [], [], []
    p, q = np.arange(n)[None, :, None], np.arange(n)[None, None, :]
    for d, block in _blocks(system, wave, z, z[1] - z[0], 2).items():
        i = np.arange(max(0, -d), min(nz, nz - d))
        if i.size == 0:
            continue
        rows.append(((i * n)[:, None, None] + p + 0 * q).ravel())
        cols.append((((i + d) * n)[:, None, None] + q + 0 * p).ravel())
        data.append(block[i].ravel())
    return sparse.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(nz * n, nz * n))


def coupling(system, wave, delta=None):
    """(norm of the nonlocal term, constant added to the energy bound).  Young: ||K * f|| <= ||K||_1 ||f||."""
    z = np.linspace(-wave.L, wave.L, 4001)
    Y = wave.Y(z)
    outer, inner = system.outer(Y, wave.c), system.inner(Y[system.offset])
    bound = sum(np.linalg.norm(outer[:, :, k], axis=1).max() * system.norm1[k]
                * np.linalg.norm(inner[:, k, :], axis=1).max() for k in range(system.count))
    return float(bound), lambda d: float(bound)


def details(system):
    return dict(kernels=system.count, kernel_radius=system.radius, kernel_mass=system.kernel_mass,
                diffusion=bool(system.diffusive))


def rate(system, spec, wave):
    """Largest spatial rate of the local part at the end states; it sets the first step of the spectral grids.

    Without diffusion the kernels set the step (min_step): infinity is returned.
    """
    if not system.diffusive:
        return np.inf
    return max(np.abs(np.linalg.eigvals(system.system0.end_state(spec[s], wave.c)[0])).max()
               for s in ("left", "right"))


def _standing(system, wave, a0):
    """True if the wave does not move: its shift in one time unit of the local part is far below the kernel width."""
    return abs(wave.c) <= 1e-8 * min(system.length) * max(float(np.linalg.norm(a0, 2, axis=(1, 2)).max()), 1e-300)


def core_spectrum(system, wave):
    """Standing wave without diffusion: sup Re of the eigenvalues of a0(z) along the wave, else None.

    The local part is then the multiplication by a0(z); all its values belong to the essential spectrum.
    """
    if system.diffusive:
        return None
    a0 = system.coefficients(wave.Y(np.linspace(-wave.L, wave.L, 4001)), wave.c)[0]
    return float(np.linalg.eigvals(a0).real.max()) if _standing(system, wave, a0) else None


def region(system, wave, delta, safety=1.1):
    """Search rectangle without diffusion (DESIGN 2.10): ((x_max, y_max), source, data), or None.

    lambda p = c p' + a0(z) p + C p,   C p = sum_k o_k(z) K_k * w_k,   w_k = i_k(z) . p,   ||p|| = 1.
    (a) Numerical range:  Re lambda <= c0 + ||C||,  c0 = sup lambda_max(sym a0);
        for a standing wave also  |Im lambda| <= sup ||skew a0|| + ||C||.
    (b) Decay from the equation.  lambda w_k = i_k . (c p' + a0 p + C p) and
        K * (i . p') = K' * (i . p) - K * (i' . p)  give  ||K_k * w_k|| <= Theta_k / |lambda|,
            Theta_k = |c| (TV(K_k) iota_k + ||K_k||_1 iota_k') + ||K_k||_1 iota_k (sup ||a0|| + ||C||).
        With  (Re lambda - c0) <= ||C p||:   |lambda| <= sum_k sup|o_k| Theta_k / (-delta - c0).
        It needs -delta > c0.
    (c) Fourier multiplier, for a local part that is nearly constant along the wave.  With mid-values
        M, o_k and R(kappa) = (lambda - M - i c kappa)^-1 an eigenfunction satisfies
            1 <= sum_k sup_kappa ||R(kappa) o_k|| |Khat_k(kappa)| iota_k + sup_kappa ||R(kappa)|| rest,
        rest = sup ||a0 - M|| + sum_k sup ||o_k(z) - o_k|| ||K_k||_1 iota_k.
    The rectangle is the smallest that (a), (b), (c) allow; None if neither (b) nor (c) bounds Im.
    """
    if system.diffusive:
        return None
    z = np.linspace(-wave.L, wave.L, 4001)
    Y, c, n = wave.Y(z), wave.c, system.n
    a0 = system.coefficients(Y, c)[0]
    outer, inner = system.outer(Y, c), system.inner(Y[system.offset])        # (m, n, count), (m, count, n)
    count = range(system.count)
    c0 = float(np.linalg.eigvalsh((a0 + a0.transpose(0, 2, 1)) / 2).max())
    c2 = float(np.linalg.norm((a0 - a0.transpose(0, 2, 1)) / 2, 2, axis=(1, 2)).max())
    young = coupling(system, wave)[0]
    iota = np.linalg.norm(inner, axis=2).max(axis=0)
    floor = 4 * delta
    x_max = c0 + young
    data = dict(sym_max=c0, skew_max=c2, nonlocal_norm=young)
    candidates = {}
    if _standing(system, wave, a0):
        candidates["numerical range"] = c2 + young
    if -delta > c0:                                    # (b)
        slope = np.linalg.norm(np.gradient(inner, z, axis=0), axis=2).max(axis=0)
        a_sup = float(np.linalg.norm(a0, 2, axis=(1, 2)).max())
        theta = [abs(c) * (system.variation[k] * iota[k] + system.norm1[k] * slope[k])
                 + system.norm1[k] * iota[k] * (a_sup + young) for k in count]
        radius = float(sum(np.linalg.norm(outer[:, :, k], axis=1).max() * theta[k] for k in count) / (-delta - c0))
        candidates["decay from the equation"] = radius
        x_max = min(x_max, radius)
        data.update(decay_radius=radius)
    multiplier = None if "numerical range" in candidates else _multiplier_bound(
        system, c, a0, outer, iota, delta, x_max, young)
    if multiplier is not None:
        candidates["resolvent estimate"] = multiplier[0]
        data.update(variable_part=multiplier[1], resolvent_limit=multiplier[2])
    if not candidates:
        return None
    source = min(candidates, key=candidates.get)
    return (float(safety * max(x_max, floor)), float(safety * max(candidates[source], floor))), source, data


def _multiplier_bound(system, c, a0, outer, iota, delta, x_max, young):
    """Estimate (c) of region(): (bound of |Im lambda|, rest, sup ||R|| rest), or None if it gives no bound.

    |Khat| <= (total variation of K) / |kappa| is used where the transform is not computed.
    """
    n, count = system.n, range(system.count)
    mid = lambda A: (A.max(axis=0) + A.min(axis=0)) / 2
    M, o_mid = mid(a0), mid(outer)
    rest = float(np.linalg.norm(a0 - M[None], 2, axis=(1, 2)).max()
                 + sum(np.linalg.norm(outer[:, :, k] - o_mid[None, :, k], axis=1).max() * system.norm1[k] * iota[k]
                       for k in count))
    scale = float(np.linalg.norm(M, 2) + young + abs(x_max) + delta)
    half = np.sinh(np.linspace(0.0, np.arcsinh(1e6 * max(1.0, 1.0 / min(system.length))), 2001))
    transform = []
    for k in count:
        value = np.abs(system.fourier(k, half))
        resolved = half * (2 * system.radius[k] / 4000) < 0.5
        bound = np.minimum(system.norm1[k], system.variation[k] / np.maximum(half, 1e-300))
        value = np.where(resolved, np.minimum(value + 1e-6 * system.norm1[k], system.norm1[k]), bound)
        transform.append(np.concatenate((value[:0:-1], value)))
    kappa = np.concatenate((-half[:0:-1], half))
    positive = np.geomspace(1e-6 * scale, 1e9 * scale, 4000)
    eta = np.concatenate((-positive[::-1], [0.0], positive))
    omega = np.concatenate(([0.0], np.geomspace(1e-4 * scale, 1e8 * scale, 601)))
    eye = np.eye(n)[None]
    y_found, limit = 0.0, 0.0
    for x in np.linspace(-delta, max(x_max, -delta), 5):
        with np.errstate(all="ignore"):
            resolvent = np.linalg.inv((x + 1j * eta)[:, None, None] * eye - M[None])
        if not np.all(np.isfinite(resolvent)):
            return None
        rho = float(np.linalg.norm(resolvent, 2, axis=(1, 2)).max())
        limit = max(limit, rho * rest)
        if rho * rest >= 1.0:
            return None                                # the non-constant part of the local coefficients is too large
        psi = np.full(omega.size, rho * rest)
        for k in count:
            g = np.linalg.norm(resolvent @ o_mid[:, k], axis=1)                          # ||R o_k|| on the eta grid
            psi += iota[k] * (np.interp(omega[:, None] - c * kappa[None, :], eta, g) * transform[k][None, :]).max(axis=1)
        possible = np.nonzero(psi >= 1.0)[0]
        if possible.size and possible[-1] == omega.size - 1:
            return None
        if possible.size:
            y_found = max(y_found, float(omega[possible[-1] + 1]))
    return y_found, rest, limit


def min_step(system, wave=None):
    """Smallest coarse step of the spectral grids: it keeps the band of the convolution narrow."""
    return max(system.radius) / BAND


def delta_cap(system):
    """No upper limit of delta (delay.py has one)."""
    return np.inf


def resolution(system, y_max, c):
    """The discretisation of the kernels does not depend on the search region (delay.py: it does)."""
    return None


System = KernelSystem


# ------------------------------------------------------------------------------------------
# The wave
# ------------------------------------------------------------------------------------------
def _newton(system, z, U, c, eta, left, right, ref, dref, tol=1e-11, max_iter=40):
    """Discretised profile problem with the convolutions  eta K * g(U) + (1 - eta) mass g(U)."""
    local, n, m = system.local, system.n, z.size
    h = z[1] - z[0]
    off = system.offset
    w1, w2 = _stencils(h, 4)
    uL, uR = np.asarray(left, float), np.asarray(right, float)
    p, q = np.arange(n)[None, :, None], np.arange(n)[None, None, :]
    mass = np.array(system.kernel_mass)[:, None]
    kernel_weights = [system.weights(h, k) for k in range(system.count)]

    def residual(U, c):
        dU = _apply(w1, U, uL, uR)
        if system.diffusive:                           # state (U, U'), highest derivatives U''
            Y, ddU = np.empty((system.N0, m)), _apply(w2, U, uL, uR)
            Y[off], Y[off + 1] = U, dU
        else:                                          # state U, highest derivatives U'
            Y, ddU = U, dU
        V = eta * _convolutions(system, U, h, uL, uR) + (1 - eta) * mass * system.g(U)
        theta = system.theta(V)
        H = local._H(*Y, c, *theta).reshape(n, n, m)
        E = np.einsum("ijm,jm->im", H, ddU) + local._r(*Y, c, *theta)
        return np.concatenate((E.T.ravel(), [h * ((U - ref) * dref).sum()])), (Y, dU, ddU, theta)

    for iteration in range(max_iter):
        F, (Y, dU, ddU, theta) = residual(U, c)
        norm = np.abs(F).max()
        if norm < tol:
            return U, c
        a = local._a_k(*Y, *ddU, c, *theta).reshape(system.K + 1, n, n, m).transpose(0, 3, 1, 2)
        outer = system._outer(*Y, *ddU, c, *theta).reshape(n, system.count, m).transpose(2, 0, 1)
        inner = system.inner(U)
        blocks = _local_blocks(a, h, 4)
        for k, (offsets, w) in enumerate(kernel_weights):
            blocks[0] = blocks[0] + (1 - eta) * mass[k, 0] * outer[:, :, k, None] * inner[:, None, k, :]
            if eta == 0.0:
                continue
            for d, weight in zip(offsets.tolist(), w.tolist()):
                if weight == 0.0 or abs(d) >= m:
                    continue
                i = np.arange(max(0, -d), min(m, m - d))
                block = blocks.get(d)
                if block is None:
                    block = blocks[d] = np.zeros((m, n, n))
                block[i] = block[i] + eta * weight * outer[i][:, :, k, None] * inner[i + d][:, None, k, :]
        rows, cols, data = [], [], []
        for d, block in blocks.items():
            i = np.arange(max(0, -d), min(m, m - d))
            if i.size == 0:
                continue
            rows.append(((i * n)[:, None, None] + p + 0 * q).ravel())
            cols.append((((i + d) * n)[:, None, None] + q + 0 * p).ravel())
            data.append(block[i].ravel())
        size = m * n
        rows += [np.arange(size), np.full(size, size)]
        cols += [np.full(size, size), np.arange(size)]
        data += [(local.mass[:, None] * dU).T.ravel(), (h * dref).T.ravel()]
        J = sparse.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(size + 1, size + 1))
        step = spsolve(J, -F)
        if not np.all(np.isfinite(step)):
            raise WaveError("the Newton iteration of the nonlocal profile problem broke down")
        t = 1.0
        while t > 1 / 128:
            U_new, c_new = U + t * step[:-1].reshape(m, n).T, c + t * step[-1]
            if np.abs(residual(U_new, c_new)[0]).max() < (1 - 1e-4 * t) * norm:
                break
            t /= 2
        else:
            raise WaveError("the Newton iteration of the nonlocal profile problem stalled")
        U, c = U_new, c_new
    raise WaveError("the Newton iteration of the nonlocal profile problem did not converge")


def solve_wave(problem, system, tol=1e-8, check_tol=5e-10, tail_tol=1e-9, L_max=2000.0, h_factor=0.04,
               max_nodes=40000):
    """Wave of the nonlocal problem on a uniform grid.

    First attempt: Newton iteration from the guess of the problem file.  If it fails:
    homotopy from the local limit (eta = 0, every kernel replaced by its mass times a
    delta function) to the full kernel (eta = 1).
    """
    spec = problem.wave
    left, right = np.asarray(spec["left"], float), np.asarray(spec["right"], float)
    off, system0 = system.offset, system.system0
    c_guess = spec["speed"] if spec["speed"] is not None else spec["speed_guess"]
    guess = reference_from_guess(system.problem0, system0)
    radius = max(system.radius)
    if system.diffusive:
        ends = end_data(system0, left, right, c_guess, True)
        if not ends["free_c"]:
            raise WaveError("with a convolution, only waves with an isolated speed are supported")
        # Step: fine enough for the local terms, and at most 4 BAND nodes inside the kernel radius.
        h = max(h_factor / ends["fast"], radius / (4 * BAND))
        L = float(np.clip(np.log(1.0 / tail_tol) / ends["slow"], max(6.0, 3 * radius), L_max))
    else:
        # Without diffusion the kernels set the scales.  The domain grows until the tails are small.
        h = radius / (4 * BAND) * (h_factor / 0.04)
        L = float(np.clip(max(6.0, 1.5 * radius, 12 * spec["width"]), None, L_max))
    route = {}

    def solve(h, L, start=None):
        z = np.arange(-L + h, L - h / 2, h)
        if start is not None:                                       # a solution on another grid
            U0, c0 = start
            return z, *_newton(system, z, U0(z), c0, 1.0, left, right, U0(z), U0(z, 1))
        U0, dU0 = guess.U(z), guess.dU(z)
        try:
            route["start"] = "the guess of the problem file"
            return z, *_newton(system, z, U0.copy(), c_guess, 1.0, left, right, U0, dU0)
        except WaveError:
            if not system.diffusive:
                raise WaveError("no wave was found: Newton from the guess failed (without diffusion "
                                "there is no local limit to start a homotopy)") from None
        route["start"] = "homotopy from the local limit"
        base = solve_local_wave(system.problem0, system0, tol=tol, check_tol=check_tol,
                                tail_tol=tail_tol, L_max=L_max, max_nodes=max_nodes)
        Yb = base.Y(z)
        U, c, eta, step = Yb[off].copy(), base.c, 0.0, 0.25
        while eta < 1.0 - 1e-12:
            target = min(eta + step, 1.0)
            try:
                U, c = _newton(system, z, U, c, target, left, right, Yb[off], Yb[off + 1])
                eta = target
            except WaveError:
                step /= 2
                if step < 1 / 4096:
                    raise WaveError(f"no wave was found: Newton from the guess failed, and the homotopy "
                                    f"from the local limit failed near eta = {eta:.4g}") from None
        return z, U, c

    for _ in range(4):
        z, U, c = solve(h, L)
        edge = max(int(0.02 * z.size), 3)
        tails = max(np.abs(U[:, :edge] - left[:, None]).max(), np.abs(U[:, -edge:] - right[:, None]).max())
        scale = max(1.0, float(np.abs(U).max()))
        if tails < 1e3 * tail_tol * scale or L >= L_max:
            break
        L = min(1.5 * L, L_max)
    if tails > 1e3 * tail_tol * scale:
        raise WaveError(f"the profile has not reached its end states at the ends of the domain "
                        f"(distance {tails:.2g}, L = {L:.4g})")
    if spec["kind"] == "pulse" and np.abs(U - left[:, None]).max() < 1e-6 * scale:
        raise WaveError("the solution collapsed to the constant state")
    grid = np.concatenate(([-L], z, [L]))
    spline = CubicSpline(grid, np.hstack((left[:, None], U, right[:, None])), axis=1)
    z2, U2, c2 = solve(2 * h, L, start=(spline, c))                    # error estimate: twice the step
    coarse = CubicSpline(np.concatenate(([-L], z2, [L])), np.hstack((left[:, None], U2, right[:, None])), axis=1)
    V = _convolutions(system, U, h, left, right)
    V_ends = np.array(system.kernel_mass)[:, None] * np.hstack((system.g(left), system.g(right)))
    V_spline = CubicSpline(grid, np.hstack((V_ends[:, :1], V, V_ends[:, 1:])), axis=1)
    probe = np.linspace(-L, L, 4001)
    # The product integration of the convolution is second order: the error is a third of the difference.
    profile_error = float(np.abs(spline(probe) - coarse(probe)).max()) / 3
    N0 = system.N0

    def interpolant(zz):
        zz = np.atleast_1d(np.asarray(zz, dtype=float))
        Y = np.empty((system.N, zz.size))
        Y[off], Y[N0:] = spline(zz), V_spline(zz)
        if system.diffusive:
            Y[off + 1] = spline(zz, 1)
        return Y

    info = dict(speed_is_unknown=True, nodes=int(z.size), grid_step=float(h), speed_error=abs(c - c2) / 3,
                profile_error=profile_error, tail_error=float(tails), started_from=route["start"])
    return Wave(c=float(c), L=float(L), N=system.N, Y_minus=system.state(left), Y_plus=system.state(right),
                interpolant=interpolant, mesh=grid, info=info, theta=())

"""Discrete Evans function of  P' = (M0(z) + lambda M1(z)) P  (DESIGN.md, section 3.2).

With second-order time derivatives the system is  P' = (M0 + lambda M1 + lambda^2 M2) P;
the cell relation is then a polynomial of degree four in lambda.

The unknowns are the values of P at all mesh nodes.  Each cell contributes one
fourth-order Lobatto (Hermite-Simpson) relation.  The boundary rows are

    B_L(lambda) = C_L Ps_-(lambda),      B_R(lambda) = C_R Pu_+(lambda),

with spectral projectors of the end-state matrices and constant matrices C.
These rows are analytic in lambda and have no poles.  The determinant vanishes
at the eigenvalues and, possibly, at points where B_L or B_R loses rank.  The
function gamma(lambda) certifies the absence of such points (winding number 0).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg.lapack import zgbtrf

from .asymptotic import SlowModeError, split, split_touching, reference_rows


class SplittingError(ArithmeticError):
    """lambda is outside the region of consistent splitting."""


MESH_EXPONENT = 0.25          # equidistribution exponent of a fourth-order scheme


def wave_variation(system, wave, samples=4001):
    """Relative rate of change s(z) of the profile, scaled to a maximum of 1.

    Each state component is measured against its own range along the wave.
    """
    z = np.linspace(-wave.L, wave.hi, samples)
    Y = wave.Y(z)
    span = Y.max(axis=1) - Y.min(axis=1)
    scale = np.maximum(span, 1e-6 * max(span.max(), 1e-300))
    s = np.abs(np.gradient(Y, z, axis=1) / scale[:, None]).max(axis=0)
    return z, s / max(s.max(), 1e-300)


def step_factor(variation, cap):
    """Local step divided by h:  min(cap, s^(-1/4))."""
    z, s = variation
    return z, np.minimum(cap, np.maximum(s, 1e-300) ** (-MESH_EXPONENT))


def graded_cells(variation, La, Lb, h, cap):
    """Number of cells of the graded mesh on [-La, Lb]."""
    z, factor = step_factor(variation, cap)
    grid = np.unique(np.concatenate(([-La], z[(z > -La) & (z < Lb)], [Lb])))
    density = 1.0 / (h * np.interp(grid, z, factor))
    cumulative = np.concatenate(([0.0], np.cumsum(np.diff(grid) * (density[1:] + density[:-1]) / 2)))
    return grid, cumulative, max(4, int(np.ceil(cumulative[-1])))


def graded_nodes(variation, La, Lb, h, cap):
    """Mesh on [-La, Lb] whose local step is h * min(cap, s(z)^(-1/4)).

    The step is h where the wave varies fastest and grows where the wave is flat.
    """
    grid, cumulative, cells = graded_cells(variation, La, Lb, h, cap)
    return np.interp(np.linspace(0.0, cumulative[-1], cells + 1), cumulative, grid)


class Asymptotics:
    """End-state data as functions of lambda, shared by all meshes of one wave."""

    def __init__(self, system, left, right, c, lam_ref):
        minus = [M[0] for M in system.pencil(system.state(left), c)]
        plus = [M[0] for M in system.pencil(system.state(right), c)]
        self.M0m, self.M1m, self.M0p, self.M1p = minus[0], minus[1], plus[0], plus[1]
        self.M2m, self.M2p = (minus[2], plus[2]) if len(minus) > 2 else (None, None)
        self.N = N = system.N
        # A real reference point gives real matrices C: the determinant is then real on the real axis.
        ref = complex(lam_ref).real if complex(lam_ref).imag == 0 else complex(lam_ref)
        km, Pum, *_ = self._split(-1, ref)
        kp, Pup, *_ = self._split(+1, ref)
        if km != kp:
            raise SplittingError("the end states have different unstable dimensions at the reference point")
        self.k = km
        self.lam_ref = complex(lam_ref)
        self.CL = reference_rows(np.eye(N) - Pum, N - km)
        self.CR = reference_rows(Pup, km)
        self.cache = {}

    def _split(self, side, lam):
        """Splitting of the end-state matrix at -infinity (side < 0) or +infinity."""
        return split(self.minus(lam) if side < 0 else self.plus(lam))

    def minus(self, lam):
        """End-state matrix at -infinity."""
        M = self.M0m + lam * self.M1m
        return M if self.M2m is None else M + lam * lam * self.M2m

    def plus(self, lam):
        """End-state matrix at +infinity."""
        M = self.M0p + lam * self.M1p
        return M if self.M2p is None else M + lam * lam * self.M2p

    def at(self, lam):
        lam = complex(lam)
        hit = self.cache.get(lam)
        if hit is not None:
            return hit
        N = self.N
        km, Pum, eum, esm = self._split(-1, lam)
        kp, Pup, eup, esp = self._split(+1, lam)
        if km != self.k or kp != self.k:
            raise SplittingError(f"lambda = {lam:.6g} is outside the region of consistent splitting")
        BL = self.CL @ (np.eye(N) - Pum)
        BR = self.CR @ Pup
        gamma = 1.0 + 0j
        if BL.shape[0]:
            gamma *= np.linalg.det(BL @ self.CL.conj().T)
        if BR.shape[0]:
            gamma *= np.linalg.det(BR @ self.CR.conj().T)
        value = (BL, BR, np.concatenate((eum, esm)), np.concatenate((eup, esp)), complex(gamma))
        self.cache[lam] = value
        return value


class TouchingAsymptotics(Asymptotics):
    """End-state data on the closed half-plane Re(lambda) >= 0, for the integrated operator of a
    conservation law: the slow eigenvalues are assigned by their direction of motion (split_touching)."""

    def _split(self, side, lam):
        try:
            return (split_touching(self.minus(lam), self.M1m) if side < 0
                    else split_touching(self.plus(lam), self.M1p))
        except SlowModeError as error:
            raise SplittingError(str(error)) from error


class Evans:
    """Normalised matching determinant on one mesh.  Call with lambda."""

    def __init__(self, asym, nodes, M0, M1, M2=None):
        self.asym = asym
        self.nodes = nodes = np.asarray(nodes, dtype=float)
        N, k = asym.N, asym.k
        h = np.diff(nodes)
        self.cells = m = h.size
        hh = h[:, None, None]
        eye = np.eye(N)
        a0, m0, b0 = M0[0:-1:2], M0[1::2], M0[2::2]
        a1, m1, b1 = M1[0:-1:2], M1[1::2], M1[2::2]
        # Cell relation:  -Aplus Y_j + Aminus Y_{j+1} = 0, each block quadratic in lambda.
        self.left = (-(eye + hh / 6 * (a0 + 2 * m0) + hh ** 2 / 12 * (m0 @ a0)),
                     -(hh / 6 * (a1 + 2 * m1) + hh ** 2 / 12 * (m0 @ a1 + m1 @ a0)),
                     -(hh ** 2 / 12 * (m1 @ a1)))
        self.right = (eye - hh / 6 * (b0 + 2 * m0) + hh ** 2 / 12 * (m0 @ b0),
                      -hh / 6 * (b1 + 2 * m1) + hh ** 2 / 12 * (m0 @ b1 + m1 @ b0),
                      hh ** 2 / 12 * (m1 @ b1))
        if M2 is not None:                             # the same relation with polynomials of degree two
            a, mid, b = ([M[0:-1:2] for M in (M0, M1, M2)], [M[1::2] for M in (M0, M1, M2)],
                         [M[2::2] for M in (M0, M1, M2)])
            zero = np.zeros_like(a[0])

            def product(p, q):
                return [sum((p[i] @ q[k - i] for i in range(3) if 0 <= k - i < 3), zero) for k in range(5)]

            def padded(p, k):
                return p[k] if k < 3 else zero

            ma, mb = product(mid, a), product(mid, b)
            self.left = tuple(-((eye if k == 0 else 0) + hh / 6 * (padded(a, k) + 2 * padded(mid, k))
                                + hh ** 2 / 12 * ma[k]) for k in range(5))
            self.right = tuple((eye if k == 0 else 0) - hh / 6 * (padded(b, k) + 2 * padded(mid, k))
                               + hh ** 2 / 12 * mb[k] for k in range(5))
        r0 = N - k
        self.kl, self.ku = 2 * N - k - 1, N + k - 1
        self.size = (m + 1) * N
        base = self.base = self.kl + self.ku
        cell = np.arange(m)[:, None, None] * N
        rows = r0 + cell + np.arange(N)[None, :, None]
        cols = cell + np.arange(N)[None, None, :]
        self.index_left = (base + rows - cols, cols + 0 * rows)
        self.index_right = (base + rows - cols - N, cols + N + 0 * rows)
        i, a = np.arange(r0)[:, None], np.arange(N)[None, :]
        self.index_bl = (base + i - a, a + 0 * i)
        i = np.arange(k)[:, None]
        self.index_br = (base + r0 + i - a, m * N + a + 0 * i)
        mid = (nodes[1:] + nodes[:-1]) / 2
        self.h_left, self.h_right = h[mid < 0], h[mid >= 0]
        self.cache = {}

    @classmethod
    def build(cls, system, wave, asym, La, Lb, h, cap=1.0, variation=None):
        """Mesh on [-La, Lb] with step h where the wave varies fastest.

        cap <= 1: uniform mesh with a node at 0 and spacing close to h (the mesh of M1).
        cap > 1 : graded mesh, see graded_nodes(); `variation` comes from wave_variation().
        """
        if cap > 1.0 and variation is not None:
            nodes = graded_nodes(variation, La, Lb, h, cap)
        else:
            nl, nr = max(2, int(np.ceil(La / h))), max(2, int(np.ceil(Lb / h)))
            nodes = np.concatenate((np.linspace(-La, 0.0, nl + 1), np.linspace(0.0, Lb, nr + 1)[1:]))
        points = np.empty(2 * nodes.size - 1)
        points[0::2] = nodes
        points[1::2] = (nodes[1:] + nodes[:-1]) / 2
        return cls(asym, nodes, *system.pencil(wave.Y(points), wave.c))

    def _log_norm(self, ev_minus, ev_plus):
        total = 0j
        for hs, ev in ((self.h_left, ev_minus), (self.h_right, ev_plus)):
            v = np.outer(hs, ev * np.sign(ev.real))
            total += np.log(1 + v / 2 + v * v / 12).sum()
        return total

    def __call__(self, lam):
        """Return (log|D|, D/|D|, gamma) of the normalised determinant D at lambda."""
        lam = complex(lam)
        hit = self.cache.get(lam)
        if hit is not None:
            return hit
        BL, BR, ev_minus, ev_plus, gamma = self.asym.at(lam)
        ab = np.zeros((2 * self.kl + self.ku + 1, self.size), dtype=complex, order="F")
        ab[self.index_left] = self._horner(self.left, lam)
        ab[self.index_right] = self._horner(self.right, lam)
        ab[self.index_bl] = BL
        ab[self.index_br] = BR
        lu, piv, info = zgbtrf(ab, self.kl, self.ku, overwrite_ab=True)
        if info < 0:
            raise ArithmeticError(f"band LU: illegal argument {-info}")
        if info > 0:
            value = (-np.inf, 0j, gamma)
        else:
            diag = lu[self.base]
            swaps = np.count_nonzero(piv != np.arange(self.size))
            norm = self._log_norm(ev_minus, ev_plus)
            angle = np.angle(diag).sum() + np.pi * swaps - norm.imag
            value = (float(np.log(np.abs(diag)).sum() - norm.real), complex(np.exp(1j * angle)), gamma)
        self.cache[lam] = value
        return value

    @staticmethod
    def _horner(coefficients, lam):
        value = coefficients[-1]
        for coefficient in coefficients[-2::-1]:
            value = coefficient + lam * value
        return value

    def many(self, lams):
        """Batch interface: evaluate at a sequence of lambda values."""
        return [self(lam) for lam in lams]

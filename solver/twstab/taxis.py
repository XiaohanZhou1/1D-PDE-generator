"""Transport towards a component without diffusion: chemotaxis and haptotaxis (DESIGN.md, section 2.17).

    u_t = d/dx( A(u, w) u_x + B(u, w) w_x ) + R(u, u_x, w, w_x),      w_t = g(u, w).

u: components with diffusion (set P); w: components whose equation has no x-derivative (set H).
The second derivative of w appears in the equations of u (B is not zero), but w has no diffusion:
the matrix of the highest derivatives is singular, and the jet form of class L does not exist.

First-order form in flux variables.  With the flux  Phi = A U' + B W'  the travelling-wave problem is
    W' = (s_w - g) / c,      U' = A^(-1) (Phi - B W'),      Phi' = s_u - R - c U',
with the sources s = 0 for the profile and s = lambda (p, q) for the eigenvalue problem.  The
linearisation is linear in lambda; no derivative of a coefficient along the wave is needed.

TaxisSystem   the same interface as symbolic.LocalSystem, with the state (U_j, Phi_j) for the
              components with diffusion and (W_k) for the others
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from .symbolic import StructureError, _vectorised


class TaxisSystem:
    """Profile equation and eigenvalue problem in flux variables.  `base`: the LocalSystem of the problem."""

    quadratic = False
    conservative = False

    def __init__(self, problem, base):
        if (getattr(base, "quadratic", False) or np.any(base.mass == 0) or base.K != 2
                or getattr(problem, "boundary", None) is not None):
            raise StructureError("transport towards a component without diffusion needs equations of first order "
                                 "in time and second order in space, without constraints and without a free boundary")
        n, sym = base.n, base.symbols
        G, c = sym["G"], sym["c"]
        U = [[sym["low"][base.offset[j] + k] for k in range(base.mu[j])] + [sym["T"][j]] for j in range(n)]
        derivatives = [s for row in U for s in row[1:]]
        H = [j for j in range(n) if not G[j].has(*derivatives)]              # w_t = g(u, w)
        P = [j for j in range(n) if j not in H]
        with_second = [j for j in H if base.mu[j] == 2]
        if not P or not with_second:
            raise StructureError("no component without diffusion has a second derivative in another equation")
        if any(base.mu[j] != 2 for j in P):
            raise StructureError("every component with x-derivatives in its own equation must be of second order")
        zeroth = [U[j][0] for j in range(n)]
        A = sp.Matrix(len(P), len(P), lambda a, b: sp.diff(G[P[a]], U[P[b]][2]))
        B = sp.Matrix(len(P), len(H), lambda a, b: sp.diff(G[P[a]], U[H[b]][2]) if base.mu[H[b]] == 2 else 0)
        allowed = set(zeroth) | set(base.params)
        if any(e.free_symbols - allowed for e in list(A) + list(B)):
            raise StructureError("the coefficients of the second derivatives may depend on the unknowns only")

        # --- flux form  G_P = (A U' + B W')' + R ------------------------------------------------
        total = lambda e: sum(sp.diff(e, U[l][0]) * U[l][1] for l in range(n))
        second = [U[j][2] for j in range(n) if base.mu[j] == 2]
        R = []
        for a, i in enumerate(P):
            flux_x = sum(A[a, b] * U[j][2] + total(A[a, b]) * U[j][1] for b, j in enumerate(P))
            flux_x += sum(B[a, b] * U[j][2] + total(B[a, b]) * U[j][1] for b, j in enumerate(H) if base.mu[j] == 2)
            rest = sp.expand(G[i] - flux_x)
            if rest.has(*second):
                raise StructureError("the equations are not affine in the second derivatives")
            R.append(rest)

        # --- state, sources, vector field --------------------------------------------------------
        self.base, self.n, self.K = base, n, 2
        self.diffusive, self.first_order_components = P, H
        self.mu = [2 if j in P else 1 for j in range(n)]
        self.orders = list(self.mu)
        self.offset = np.concatenate(([0], np.cumsum(self.mu)[:-1])).astype(int)
        self.N = N = int(sum(self.mu))
        self.top_rows = self.offset + np.array(self.mu) - 1
        self.mass, self.inertia = np.ones(n), np.zeros(n)
        self.params = base.params
        flux = [sp.Dummy(f"flux{j}", real=True) for j in P]
        source = [sp.Dummy(f"source{j}", real=True) for j in range(n)]
        g = sp.Matrix([G[j] for j in H])
        Wp = (sp.Matrix([source[j] for j in H]) - g) / c
        Up = A.LUsolve(sp.Matrix(flux) - B * Wp)
        first = {U[j][1]: Up[a] for a, j in enumerate(P)}
        first.update({U[j][1]: Wp[b] for b, j in enumerate(H)})
        Fp = [source[i] - R[a].xreplace(first) - c * Up[a] for a, i in enumerate(P)]
        state, field = [], []
        for j in range(n):
            if j in P:
                a = P.index(j)
                state += [U[j][0], flux[a]]
                field += [Up[a], Fp[a]]
            else:
                state.append(U[j][0])
                field.append(Wp[H.index(j)])
        off = {s: 0 for s in source}
        args = state + [c] + list(base.params)
        self._V = _vectorised(args, [v.xreplace(off) for v in field])
        self._JY = _vectorised(args, [sp.diff(v, y).xreplace(off) for v in field for y in state])
        self._JS = _vectorised(args, [sp.diff(v, s).xreplace(off) for v in field for s in source])
        self._A = _vectorised(zeroth + list(base.params), list(A))

        # --- the jets of the original equations along the wave: for the coefficients a_k ---------
        V0 = [v.xreplace(off) for v in field]
        along = lambda e: sum(sp.diff(e, y) * v for y, v in zip(state, V0))    # d/dz on the wave
        value = {U[j][0]: U[j][0] for j in range(n)}
        for j in range(n):
            value[U[j][1]] = first[U[j][1]].xreplace(off)
            if base.mu[j] == 2:
                value[U[j][2]] = along(value[U[j][1]])
        self._jets = _vectorised(args, [value[s] for s in list(sym["low"]) + list(sym["T"])])
        # Flux of a guess (U, U', W):  Phi = A U' + B W'  with  W' = -g / c.
        slope = [U[j][1] for j in P]
        guess_flux = A * sp.Matrix(slope) + B * (-g / c)
        self._guess_flux = _vectorised(zeroth + slope + [c] + list(base.params), list(guess_flux))

    # --- parameters are those of the base system ---------------------------------------------------
    @property
    def theta(self):
        return self.base.theta

    @theta.setter
    def theta(self, value):
        self.base.theta = value

    def set_parameters(self, parameters):
        self.base.set_parameters(parameters)

    def parameter_index(self, name):
        return self.base.parameter_index(name)

    # --- the interface of LocalSystem -----------------------------------------------------------------
    def state(self, u):
        Y = np.zeros(self.N)
        Y[self.offset] = np.asarray(u, dtype=float)
        return Y

    def equilibrium_residual(self, u):
        return self.base.equilibrium_residual(u)

    def equilibrium(self, u, **kwargs):
        return self.base.equilibrium(u, **kwargs)

    def top_matrix_condition(self, u, c):
        if c == 0:
            return np.inf                                # W' = -g / c: the speed must not vanish
        m = len(self.diffusive)
        return float(np.linalg.cond(self._A(*np.asarray(u, dtype=float), *self.theta).reshape(m, m)))

    def _split(self, Y):
        return np.asarray(Y, dtype=float).reshape(self.N, -1)

    def rhs(self, Y, c):
        return self._V(*self._split(Y), c, *self.theta)

    def first_order(self, Y, c):
        """M0, M1 of  P' = (M0 + lambda M1) P  for P = (p_j, flux_j; q_k); shape (m, N, N)."""
        Y = self._split(Y)
        N, n, m = self.N, self.n, Y.shape[1]
        M0 = self._JY(*Y, c, *self.theta).reshape(N, N, m).transpose(2, 0, 1)
        M1 = np.zeros((m, N, N))
        M1[:, :, self.offset] = self._JS(*Y, c, *self.theta).reshape(N, n, m).transpose(2, 0, 1)
        return M0, M1

    pencil = first_order

    def coefficients(self, Y, c):
        """Coefficients a_k of the linearisation of the original equations on the wave; (3, m, n, n)."""
        Y = self._split(Y)
        jets = self._jets(*Y, c, *self.theta)
        nb, n = self.base.N, self.n
        a = self.base._a_k(*jets[:nb], *jets[nb:], c, *self.theta)
        return a.reshape(self.K + 1, n, n, Y.shape[1]).transpose(0, 3, 1, 2)

    def end_state(self, u, c):
        Y = self.state(u)
        M0, M1 = self.first_order(Y, c)
        return M0[0], M1[0], self.coefficients(Y, c)[:, 0]

    def from_jets(self, Yjet, c):
        """State in flux variables from a guess in jets: rows (U, U') and (W)."""
        Yjet = np.asarray(Yjet, dtype=float).reshape(self.N, -1)
        zeroth = Yjet[self.offset]
        slope = Yjet[[self.offset[j] + 1 for j in self.diffusive]]
        out = Yjet.copy()
        out[[self.offset[j] + 1 for j in self.diffusive]] = self._guess_flux(*zeroth, *slope, c, *self.theta)
        return out

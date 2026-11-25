"""Travelling-wave reduction and linearisation of a local system.

Mathematics: DESIGN.md, sections 2.1 to 2.3.

For  u_t = G(u, u_x, ...)  and  u = U(z), z = x - c t,  the profile equation is
    E(U, U', ...; c) = G + c U' = 0.
An equation without a time derivative is a constraint  0 = G_j  (DESIGN.md 2.7).
Its component has mass 0: no term c U_j' in E, and no lambda in the eigenvalue problem.
Component j has order mu_j = max(m_j, 1).  The state Y holds U_j^(k), k < mu_j.
E is affine in the top derivatives T:  E = H(Y, c) T + r(Y, c).
The linearised eigenvalue problem  lambda p = L p  becomes
    P' = (M0(z) + lambda M1(z)) P.

Second order in time (DESIGN.md 2.12).  A component with u_tt is solved for u_tt = G(u, ..., u_t):
    E = G(U, U', U'', u_t = -c U') - c^2 U'' = 0,
    L p = lambda (N1 p - 2 c N2 p') + lambda^2 N2 p,      N1 = -dG/du_t,  N2 = 1 for that component,
and  P' = (M0 + lambda M1 + lambda^2 M2) P.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.solvers.solveset import NonlinearError


class StructureError(ValueError):
    """The equations are outside the structure handled by this sub-solver."""


def _vectorised(args, exprs):
    """Compile expressions to one function that returns an array (len(exprs), *shape)."""
    exprs = list(exprs)
    f = sp.lambdify(args, exprs, modules="numpy", cse=True)

    def call(*values):
        shape = np.broadcast_shapes(*[np.shape(v) for v in values])
        return np.array([np.broadcast_to(np.asarray(o, dtype=float), shape) for o in f(*values)])
    return call


class LocalSystem:
    """Compiled travelling-wave data of a class-L problem."""

    def __init__(self, problem):
        x, t = problem.x, problem.t
        us = list(problem.unknowns)
        n = self.n = len(us)

        # --- solve the equations for the time derivatives:  u_t = G -------------
        dots = [sp.Dummy(f"dt{j}") for j in range(n)]
        ddots = [sp.Dummy(f"dtt{j}") for j in range(n)]
        time = {sp.Derivative(u, t): d for u, d in zip(us, dots)}
        time.update({sp.Derivative(u, (t, 2)): d for u, d in zip(us, ddots)})
        res = [sp.sympify(eq.lhs - eq.rhs).doit().xreplace(time) for eq in problem.equations]
        for r in res:
            if any(t in d.variables for d in r.atoms(sp.Derivative)):
                raise StructureError("only the time derivatives u_t and u_tt of the unknowns are supported "
                                     "(no mixed x-t derivative, no order above two)")
        second = [j for j in range(n) if any(r.has(ddots[j]) for r in res)]     # components with u_tt
        try:
            A, b = sp.linear_eq_to_matrix(res, [ddots[j] if j in second else dots[j] for j in range(n)])
        except NonlinearError as error:
            raise StructureError("the equations are not linear in the time derivatives") from error
        if A.has(sp.Derivative):
            raise StructureError("differential operators act on the time derivatives")
        if second:
            if A.has(AppliedUndef) or A.det() == 0:
                raise StructureError("with u_tt, the coefficients of the highest time derivatives must be "
                                     "constants and there must be no constraint equation")
            G, mass = list(A.LUsolve(b)), [1] * n      # u_tt = G or u_t = G; G may contain u_t of the others
        elif A.det() != 0:
            G = list(A.LUsolve(b)) if A != sp.eye(n) else list(b)
            mass = [1] * n
        else:
            # Constraint equations: rows without a time derivative.  The other rows
            # are solved for the time derivatives that they contain.
            dyn_rows = [i for i in range(n) if any(A[i, j] != 0 for j in range(n))]
            con_rows = [i for i in range(n) if i not in dyn_rows]
            dyn_cols = [j for j in range(n) if any(A[i, j] != 0 for i in dyn_rows)]
            con_cols = [j for j in range(n) if j not in dyn_cols]
            if len(dyn_rows) != len(dyn_cols) or A.extract(dyn_rows, dyn_cols).det() == 0:
                raise StructureError("the time derivatives cannot be isolated (singular mass matrix)")
            solved = A.extract(dyn_rows, dyn_cols).LUsolve(b.extract(dyn_rows, [0]))
            G, mass = [None] * n, [0] * n
            for index, j in enumerate(dyn_cols):
                G[j], mass[j] = solved[index], 1
            for row, j in zip(con_rows, con_cols):
                G[j] = b[row]
        self.mass = np.array(mass, dtype=float)

        # --- jets: replace derivatives by independent symbols --------------------
        orders = [0] * n
        for g in G:
            for d in g.atoms(sp.Derivative):
                if d.expr not in us or set(d.variables) != {x}:
                    raise StructureError(f"unsupported derivative {d}")
                j = us.index(d.expr)
                orders[j] = max(orders[j], len(d.variables))
        for j in second:
            orders[j] = max(orders[j], 2)              # the profile equation holds c^2 U''
        self.orders = orders
        if any(m == 0 for m, w in zip(orders, mass) if not w):
            raise StructureError("a constraint equation must contain an x-derivative of its own unknown; "
                                 "eliminate a purely algebraic unknown by hand")
        mu = self.mu = [max(m, 1) for m in orders]
        # Components without a second derivative in a system of second order (set H of DESIGN 2.4).
        self.first_order_components = [j for j in range(n) if orders[j] < 2] if max(mu) == 2 else []
        U = [[sp.Dummy(f"U{j}_{k}", real=True) for k in range(mu[j] + 1)] for j in range(n)]
        rep = {u: U[j][0] for j, u in enumerate(us)}
        for g in G:
            for d in g.atoms(sp.Derivative):
                rep[d] = U[us.index(d.expr)][len(d.variables)]
        G = [g.xreplace(rep) for g in G]
        jets = {s for row in U for s in row}
        for g in G:
            if g.has(sp.Derivative) or g.has(x) or g.has(t):
                raise StructureError("the right-hand side is not a local function of u and its x-derivatives")

        # --- free boundary: conditions and Stefan condition as functions of the state ----
        boundary = getattr(problem, "boundary", None)
        low_jets = [U[j][k] for j in range(n) for k in range(mu[j])]
        bexpr = []
        if boundary is not None:
            full = {u: U[j][0] for j, u in enumerate(us)}
            for j, u in enumerate(us):
                for k in range(1, mu[j] + 1):
                    full[sp.Derivative(u, (x, k))] = U[j][k]
            for e in [eq.lhs - eq.rhs for eq in boundary["conditions"]] + [boundary["speed"]]:
                e = sp.sympify(e).doit().xreplace(full)
                if e.has(sp.Derivative) or e.has(x) or e.has(t) or not (e.free_symbols & jets) <= set(low_jets):
                    raise StructureError("a boundary condition may contain the unknowns and their "
                                         "x-derivatives below the highest order only")
                bexpr.append(e)

        self.params = sorted(set().union(*[g.free_symbols for g in G + bexpr]) - jets - set(dots), key=str)
        missing = [p for p in self.params if p not in problem.parameters]
        if missing:
            raise StructureError(f"no value for parameter(s) {missing}")
        self.theta = tuple(float(problem.parameters[p]) for p in self.params)

        # --- profile equation and its structure -----------------------------------
        c = sp.Dummy("c", real=True)
        # On a wave u_t = -c U' and u_tt = c^2 U''.
        velocity = {dots[j]: -c * U[j][1] for j in second}
        damping = sp.Matrix(n, n, lambda i, j: (-sp.diff(G[i], dots[j]) if j in second else sp.Integer(i == j))
                            ).xreplace(velocity)
        self.inertia = np.array([float(j in second) for j in range(n)])           # N2 = diag(inertia)
        self.quadratic = bool(second)                                             # lambda^2 in the eigenvalue problem
        G = [g.xreplace(velocity) for g in G]
        E = [G[i] - c ** 2 * U[i][2] if i in second else G[i] + mass[i] * c * U[i][1] for i in range(n)]
        T = [U[j][mu[j]] for j in range(n)]
        H = sp.Matrix(E).jacobian(T)
        if any(h.has(*T) for h in H):
            raise StructureError("the equations are not affine in the highest derivatives")
        r = [e.xreplace({s: 0 for s in T}) for e in E]

        self.offset = np.concatenate(([0], np.cumsum(mu)[:-1])).astype(int)
        N = self.N = int(sum(mu))
        self.top_rows = self.offset + np.array(mu) - 1
        low = [U[j][k] for j in range(n) for k in range(mu[j])]
        K = self.K = max(mu)
        a_low = [sp.diff(E[i], s) for i in range(n) for s in low]
        a_k = [sp.diff(E[i], U[j][k]) if k <= mu[j] else sp.Integer(0)
               for k in range(K + 1) for i in range(n) for j in range(n)]

        # Every constant state is an equilibrium: conservation form (not handled in M1).
        zero = {U[j][k]: 0 for j in range(n) for k in range(1, mu[j] + 1)}
        self.conservative = all(sp.expand(g.xreplace(zero)) == 0 for g in G)
        self._n1 = _vectorised(low + T + [c] + self.params, list(damping)) if second else None

        # Symbolic pieces for sub-solvers that differentiate with respect to more arguments (delay).
        self.symbols = dict(G=G, low=low, T=T, c=c)
        args_low = low + [c] + self.params
        args_full = low + T + [c] + self.params
        self._H = _vectorised(args_low, list(H))
        self._r = _vectorised(args_low, r)
        self._a_low = _vectorised(args_full, a_low)
        self._a_k = _vectorised(args_full, a_k)
        self.n_boundary = max(len(bexpr) - 1, 0)
        if bexpr:
            args_b = low + self.params
            conditions, stefan = bexpr[:-1], bexpr[-1]
            self._bc = _vectorised(args_b, conditions)
            self._bc_jac = _vectorised(args_b, [sp.diff(e, s) for e in conditions for s in low])
            self._stefan = _vectorised(args_b, [stefan] + [sp.diff(stefan, s) for s in low])

    # ----------------------------------------------------------------------------
    def set_parameters(self, parameters):
        """Replace the parameter values.  `parameters` maps symbols or names to numbers."""
        by_name = {str(k): float(v) for k, v in parameters.items()}
        self.theta = tuple(by_name.get(str(p), old) for p, old in zip(self.params, self.theta))

    def parameter_index(self, name):
        names = [str(p) for p in self.params]
        if str(name) not in names:
            raise StructureError(f"the equations do not contain a parameter named {name}")
        return names.index(str(name))

    def reaction_jacobian(self, u):
        """dG/du at the constant state u, shape (n, n).  It does not depend on the speed."""
        Y = self.state(u).reshape(self.N, 1)
        zero = np.zeros((self.n, 1))
        return self._a_k(*Y, *zero, 0.0, *self.theta).reshape(self.K + 1, self.n, self.n)[0]

    def equilibrium(self, u, tol=1e-13, max_iter=30):
        """Constant solution near u at the current parameter values (Newton iteration)."""
        u = np.array(u, dtype=float)
        for _ in range(max_iter):
            r = self.equilibrium_residual(u)
            if np.abs(r).max() <= tol * max(1.0, np.abs(u).max()):
                return u
            u = u - np.linalg.solve(self.reaction_jacobian(u), r)
        raise StructureError("the end state could not be followed (Newton iteration failed)")

    def boundary_residual(self, Y):
        """Left-hand sides of the boundary conditions at the state Y (zero when they hold)."""
        return self._bc(*np.asarray(Y, dtype=float), *self.theta)

    def boundary_jacobian(self, Y):
        """Derivative of the boundary conditions with respect to the state, shape (n_boundary, N)."""
        return self._bc_jac(*np.asarray(Y, dtype=float), *self.theta).reshape(self.n_boundary, self.N)

    def stefan(self, Y):
        """Speed of the boundary given by the Stefan condition, and its gradient (N,)."""
        out = self._stefan(*np.asarray(Y, dtype=float), *self.theta)
        return float(out[0]), out[1:]

    def state(self, u):
        """First-order state of the constant solution u."""
        Y = np.zeros(self.N)
        Y[self.offset] = np.asarray(u, dtype=float)
        return Y

    def _points(self, Y):
        return np.asarray(Y, dtype=float).reshape(self.N, -1)

    def _solve_top(self, Y, c):
        m = Y.shape[1]
        H = self._H(*Y, c, *self.theta).reshape(self.n, self.n, m).transpose(2, 0, 1)
        r = self._r(*Y, c, *self.theta).T
        return H, -np.linalg.solve(H, r[..., None])[..., 0].T

    def top(self, Y, c):
        """Highest derivatives T on the profile, shape (n, m)."""
        return self._solve_top(self._points(Y), c)[1]

    def rhs(self, Y, c):
        """Right-hand side f(Y; c) of the first-order profile system."""
        Y = self._points(Y)
        T = self._solve_top(Y, c)[1]
        dY = np.empty_like(Y)
        for j in range(self.n):
            o, m = self.offset[j], self.mu[j]
            dY[o:o + m - 1] = Y[o + 1:o + m]
            dY[o + m - 1] = T[j]
        return dY

    def equilibrium_residual(self, u):
        """G(u, 0, ..., 0): zero if and only if u is a constant solution."""
        Y = self.state(u).reshape(self.N, 1)
        return self._r(*Y, 0.0, *self.theta)[:, 0]

    def first_order(self, Y, c):
        """Matrices M0, M1 of P' = (M0 + lambda M1) P at the given profile points.

        Returns arrays of shape (m, N, N).
        """
        Y = self._points(Y)
        m, n, N = Y.shape[1], self.n, self.N
        H, T = self._solve_top(Y, c)
        Hinv = np.linalg.inv(H)
        a_low = self._a_low(*Y, *T, c, *self.theta).reshape(n, N, m).transpose(2, 0, 1)
        M0 = np.zeros((m, N, N))
        M1 = np.zeros((m, N, N))
        for j in range(n):
            o = self.offset[j]
            for k in range(self.mu[j] - 1):
                M0[:, o + k, o + k + 1] = 1.0
        M0[:, self.top_rows, :] = -Hinv @ a_low
        M1[:, self.top_rows[:, None], self.offset[None, :]] = Hinv * self.mass      # H^{-1} diag(mass)
        return M0, M1

    def damping(self, Y, c):
        """N1(z) of  L p = lambda (N1 p - 2 c N2 p') + lambda^2 N2 p,  shape (m, n, n).  First order in time: diag(mass)."""
        Y = self._points(Y)
        m = Y.shape[1]
        if not self.quadratic:
            return np.broadcast_to(np.diag(self.mass), (m, self.n, self.n)).copy()
        T = self._solve_top(Y, c)[1]
        return self._n1(*Y, *T, c, *self.theta).reshape(self.n, self.n, m).transpose(2, 0, 1)

    def pencil(self, Y, c):
        """Matrices of the first-order system: (M0, M1), or (M0, M1, M2) with second-order time derivatives."""
        M0, M1 = self.first_order(Y, c)
        if not self.quadratic:
            return M0, M1
        rows, cols = self.top_rows[:, None], self.offset[None, :]
        Hinv = M1[:, rows, cols].copy()                                # mass = 1: these entries are H^{-1}
        N1, N2 = self.damping(Y, c), np.diag(self.inertia)
        M1, M2 = np.zeros_like(M0), np.zeros_like(M0)
        M1[:, rows, cols] = Hinv @ N1
        heavy = np.nonzero(self.inertia)[0]                            # components with u_tt: p' is a state variable
        M1[:, rows, (self.offset[heavy] + 1)[None, :]] += -2 * c * (Hinv @ N2)[:, :, heavy]
        M2[:, rows, cols] = Hinv @ N2
        return M0, M1, M2

    def coefficients(self, Y, c):
        """Operator coefficients a^(k)_{ij}(z) of L, shape (K+1, m, n, n)."""
        Y = self._points(Y)
        m, n = Y.shape[1], self.n
        T = self._solve_top(Y, c)[1]
        return self._a_k(*Y, *T, c, *self.theta).reshape(self.K + 1, n, n, m).transpose(0, 3, 1, 2)

    def end_state(self, u, c):
        """Constant-coefficient data at an end state: M0, M1 (N x N) and a^(k) (K+1, n, n)."""
        Y = self.state(u)
        M0, M1 = self.first_order(Y, c)
        return M0[0], M1[0], self.coefficients(Y, c)[:, 0]

    def top_matrix_condition(self, u, c):
        Y = self.state(u).reshape(self.N, 1)
        return float(np.linalg.cond(self._H(*Y, c, *self.theta).reshape(self.n, self.n)))

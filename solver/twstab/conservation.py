"""Equations in conservation form (DESIGN.md, section 2.14).

    u_t = d/dx F(u, u_x, ..., u^(K-1)).

Every constant is an equilibrium, and the essential spectrum touches lambda = 0.
* The wave.  One integration of the profile equation gives  F(U, U', ...) + c U = q.
  The jump condition  F(u+) - F(u-) + c (u+ - u-) = 0  fixes c for a front between given
  end states; q follows.  The profile is a standing wave of the reduced problem
  v_t = F(v, v_x, ...) + c v - q, of order K - 1.
* The spectrum.  With P = integral of p, the linearised equation  lambda p = (F_lin p)'
  becomes  lambda P = F_lin P'  (the integrated operator).  For a front P = U - u_- does not
  decay: lambda = 0 is no eigenvalue of the integrated operator.
* Two derivatives.  If F is a total derivative itself, u_t = d^2/dx^2 F2 (Cahn-Hilliard), a
  front has c = 0 and the profile equation is F2(U, ...) = q2, of order K - 2; then
  lambda P = F2_lin P''  with P the second integral of p.

antiderivative      recovers F from d/dx F by repeated integration over the jets
ConservativeSystem  the reduced problem for the wave and the integrated operator for the spectrum
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from .problem import Problem
from .symbolic import LocalSystem, StructureError, _vectorised


def antiderivative(expressions, U, orders):
    """Expressions F_i with  expressions[i] = d/dx F_i  in the jet symbols U[j][k]; None if there is none.

    orders[j]: highest derivative of component j that may occur.
    """
    n = len(U)

    def total(expr):
        return sum(sp.diff(expr, U[j][k]) * U[j][k + 1] for j in range(n) for k in range(orders[j]))

    out = []
    for g in expressions:
        F, rest = sp.Integer(0), sp.expand(g)
        for m in range(max(orders), 0, -1):
            level = [U[j][m] for j in range(n) if m <= orders[j]]
            for j in range(n):
                if m > orders[j]:
                    continue
                coefficient = sp.diff(rest, U[j][m])
                if coefficient == 0:
                    continue
                if coefficient.has(*level):
                    return None                        # not affine in the derivatives of order m
                term = sp.integrate(coefficient, U[j][m - 1])
                if term.has(sp.Integral):
                    return None
                F += term
                rest = sp.expand(rest - total(term))
        if sp.simplify(rest) != 0:
            return None
        out.append(F)
    return out


class ConservativeSystem:
    """Reduced problem and integrated operator of a system in conservation form.

    All components must have the same order K, and all equations must be in conservation form.
    depth = 1: u_t = (F)_x; depth = 2: u_t = (F2)_xx.  The state of the wave is that of the reduced
    problem: U, ..., U^(K-depth-1).  The first-order system of the integrated operator has the dimension n K.
    """

    def __init__(self, problem, system):
        if system.quadratic or np.any(system.mass == 0) or len(set(system.orders)) != 1 or system.K < 2:
            raise StructureError("the conservation mode needs equations of first order in time, without "
                                 "constraints, and the same order in every component")
        sym, n, K = system.symbols, system.n, system.K
        # The parameter values are inserted first: a symbolic exponent would give case distinctions.
        values = {p: (sp.Integer(int(v)) if float(v).is_integer() else sp.Float(v))
                  for p, v in problem.parameters.items()}
        U = [[sym["low"][system.offset[j] + k] for k in range(K)] + [sym["T"][j]] for j in range(n)]
        fluxes = antiderivative([g.xreplace(values) for g in sym["G"]], U, [K] * n)
        if fluxes is None:
            raise StructureError("every constant is an equilibrium, but the equations are not in "
                                 "conservation form u_t = (flux)_x")
        spec = problem.wave
        left, right = np.asarray(spec["left"], dtype=float), np.asarray(spec["right"], dtype=float)
        constants = [U[j][0] for j in range(n)]
        rest = {U[j][k]: 0 for j in range(n) for k in range(1, K)}

        def at_constants(F):
            f = _vectorised(constants, [e.xreplace(rest) for e in F])
            return lambda u: f(*u)

        flux = at_constants(fluxes)
        jump_u, jump_f = right - left, flux(right) - flux(left)
        scale = max(1.0, np.abs(left).max(), np.abs(right).max())
        if np.abs(jump_u).max() > 1e-12 * scale:
            speed = float(-(jump_f @ jump_u) / (jump_u @ jump_u))
            residual = float(np.abs(jump_f + speed * jump_u).max())
            if residual > 1e-8 * max(1.0, np.abs(jump_f).max(), abs(speed) * np.abs(jump_u).max()):
                raise StructureError("the end states do not satisfy the jump condition "
                                     f"F(u+) - F(u-) + c (u+ - u-) = 0 (residual {residual:.3g})")
            if spec["speed"] is not None and abs(spec["speed"] - speed) > 1e-8 * max(1.0, abs(speed)):
                raise StructureError(f"wave['speed'] differs from the speed {speed:.10g} of the jump condition")
        elif spec["speed"] is not None:
            speed, residual = float(spec["speed"]), 0.0
        else:
            # A pulse: the jump condition holds for every speed.  Without a given speed the standing
            # pulse is taken (convention, DESIGN 2.14).
            speed, residual, self.default_speed = 0.0, 0.0, True
        if abs(speed) < 1e-12 * scale:
            speed = 0.0

        self.default_speed = getattr(self, "default_speed", False)
        # A second integration: the flux is a total derivative, and the wave does not move.
        self.depth = 1
        if speed == 0.0 and K >= 3:
            second = antiderivative(fluxes, U, [K - 1] * n)
            if second is not None:
                level = at_constants(second)
                residual = float(np.abs(level(right) - level(left)).max())
                if residual > 1e-8 * max(1.0, np.abs(level(left)).max()):
                    raise StructureError("the end states do not satisfy the second jump condition "
                                         f"F2(u+) = F2(u-) (residual {residual:.3g})")
                fluxes, flux, self.depth = second, level, 2
        depth = self.depth
        self.speed, self.jump_residual = speed, residual
        self.constant = flux(left) + (speed * left if depth == 1 else 0.0)            # q (depth 1) or q2

        self.base, self.n, self.K = system, n, K
        self.N, self.mu, self.orders = n * K, [K] * n, [K] * n
        self.offset = np.arange(n) * K
        self.top_rows = self.offset + K - 1
        self.mass, self.inertia = np.ones(n), np.zeros(n)
        self.first_order_components, self.quadratic = [], False
        self.terms = K - depth + 1                                         # F_k, k = 0, ..., K - depth
        jets = [U[j][k] for j in range(n) for k in range(self.terms)]
        self._F = _vectorised(jets, [sp.diff(F, U[j][k]) for k in range(self.terms) for F in fluxes for j in range(n)])

        # --- reduced problem: v_t = F(v, v_x, ...) + c v - q, a standing wave --------------------------
        x, us = problem.x, problem.unknowns
        back = {U[j][k]: (us[j] if k == 0 else sp.Derivative(us[j], (x, k))) for j in range(n) for k in range(K)}
        equations = [sp.Eq(sp.Derivative(us[j], problem.t),
                           fluxes[j].xreplace(back) + (sp.Float(speed) * us[j] if depth == 1 else 0)
                           - sp.Float(float(self.constant[j]))) for j in range(n)]
        self.reduced_problem = Problem(path=problem.path, name=problem.name, x=x, t=problem.t, equations=equations,
                                       unknowns=us, parameters=problem.parameters, wave=dict(spec, speed=0.0),
                                       options=problem.options)
        self.reduced = LocalSystem(self.reduced_problem)
        if len(set(self.reduced.orders)) != 1 or self.reduced.orders[0] != K - depth:
            raise StructureError("the reduced profile equation does not have the order K - depth in every component")
        self.state_rows = self.reduced.N                                   # rows of the state of the wave

    def state(self, u):
        return self.reduced.state(u)

    def equilibrium_residual(self, u):
        return self.reduced.equilibrium_residual(u)

    def flux_coefficients(self, Y):
        """F_k = dF/dU^(k) on the wave (depth 1: the speed is added to F_0); shape (terms, m, n, n)."""
        Y = np.asarray(Y, dtype=float).reshape(self.state_rows, -1)
        n, terms, m = self.n, self.terms, Y.shape[1]
        top = self.reduced._solve_top(Y, 0.0)[1]                           # the highest derivative in F
        jets = np.concatenate((Y.reshape(n, terms - 1, m), top[:, None, :]), axis=1).reshape(n * terms, m)
        F = self._F(*jets).reshape(terms, n, n, m).transpose(0, 3, 1, 2)
        if self.depth == 1:
            F[0] = F[0] + self.speed * np.eye(n)[None]
        return F

    def coefficients(self, Y, c=None):
        """Coefficients of the integrated operator  lambda P = sum_k F_k P^(k+depth); shape (K + 1, m, n, n)."""
        F = self.flux_coefficients(Y)
        return np.concatenate((np.zeros((self.depth,) + F.shape[1:]), F))

    def first_order(self, Y, c=None):
        """M0, M1 of the first-order system for (P, P', ..., P^(K-1)); shape (m, n K, n K)."""
        a = self.coefficients(Y)
        n, K, m, N = self.n, self.K, a.shape[1], self.N
        inverse = np.linalg.inv(a[K])
        M0, M1 = np.zeros((m, N, N)), np.zeros((m, N, N))
        for j in range(n):
            for k in range(K - 1):
                M0[:, self.offset[j] + k, self.offset[j] + k + 1] = 1.0
        rows = self.top_rows[:, None]
        for k in range(1, K):                                              # a[0] = 0
            M0[:, rows, (self.offset + k)[None, :]] += -(inverse @ a[k])
        M1[:, rows, self.offset[None, :]] = inverse
        return M0, M1

    pencil = first_order

    def damping(self, Y, c=None):
        m = np.asarray(Y).reshape(self.state_rows, -1).shape[1]
        return np.broadcast_to(np.eye(self.n), (m, self.n, self.n)).copy()

    def transport(self, u):
        """Eigenvalues of the first-order coefficient of the integrated operator at the constant state u:
        minus the characteristic speeds relative to the wave.  None if one of them is zero or not real."""
        values = np.linalg.eigvals(self.coefficients(self.state(u))[self.depth, 0])
        size = max(np.abs(values).max(), 1e-300)
        if np.abs(values.imag).max() > 1e-9 * size or np.abs(values.real).min() <= 1e-6 * size:
            return None
        return sorted(float(v) for v in values.real)

    def end_state(self, u, c=None):
        Y = self.state(u)
        M0, M1 = self.first_order(Y)
        return M0[0], M1[0], self.coefficients(Y)[:, 0]

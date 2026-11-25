"""Multi-pulse waves (DESIGN.md, sections 2.8 and 3.6).

An N-pulse is a wave that passes N times near a primary pulse.  The relative
position of two weakly interacting pulses is almost free, so the plain profile
problem is ill-conditioned.  This module solves the pinned problem instead:

    G(U) + (c + sum_j sigma_j chi_j(z)) U' = 0,

every pulse has its own phase condition (it is pinned at a prescribed position),
chi_j = 1 around pulse j >= 2, and the speed defects sigma_j are unknowns.
A true N-pulse has sigma = 0.  For two pulses at distance D the reduced motion
is dD/dt = -sigma(D): the zeros of sigma are the admissible distances, and
-sigma'(D) is the leading-order interaction eigenvalue.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from .asymptotic import reference_rows, split
from .symbolic import StructureError
from .wave import Wave, WaveError, merged_grid

CUTOFF_WIDTH = 1.5            # width of the transition of chi_j, in units of z
MAX_NODES = 30000


class _Pinned:
    """The pinned profile problem for copies of one primary pulse."""

    def __init__(self, system, single, rest):
        self.system, self.single, self.rest = system, single, np.asarray(rest, dtype=float)
        self.Yr = system.state(rest)
        zs = merged_grid(single.mesh, np.linspace(-single.L, single.L, 4001))
        self._dU = CubicSpline(zs, system.rhs(single.Y(zs), single.c)[system.offset], axis=1)
        deviation = np.abs(single.Y(zs)[system.offset] - self.Yr[system.offset][:, None])
        self.component = int(np.argmax(deviation.max(axis=1)))       # the component that shows the pulses
        self.amplitude = float(deviation[self.component].max())
        k, Pu, *_ = split(system.end_state(rest, single.c)[0])
        eye = np.eye(system.N)
        self.k = k
        self.C_left = reference_rows(eye - Pu, system.N - k).real
        self.C_right = reference_rows(Pu, k).real

    def weight(self, z):
        """U'(z) of the primary pulse, zero outside its domain.  Shape (n, len(z))."""
        z = np.asarray(z, dtype=float)
        L = self.single.L
        return np.where(np.abs(z) <= L, self._dU(np.clip(z, -L, L)), 0.0)

    def positions(self, distances):
        distances = np.atleast_1d(np.asarray(distances, dtype=float))
        return distances.sum() / 2 - np.concatenate(([0.0], np.cumsum(distances)))   # leading pulse first

    def solve(self, distances, tol, start=None):
        """Pulses at the given distances.  Returns (solution, positions) or raises WaveError."""
        system, single, Yr = self.system, self.single, self.Yr
        N, off, eye = system.N, system.offset, np.eye(system.N)
        s = self.positions(distances)
        count = s.size
        L = single.L + (s[0] - s[-1]) / 2
        bounds = (s[:-1] + s[1:]) / 2                                 # midpoints between neighbours
        step = lambda y: 0.5 * (1 + np.tanh(y / CUTOFF_WIDTH))

        def speed(z, p):
            c = p[0] + 0.0 * z
            for j in range(1, count):                                 # region of pulse j + 1
                chi = step(bounds[j - 1] - z) - (step(bounds[j] - z) if j < count - 1 else 0.0)
                c = c + p[j] * chi
            return c

        def reference(z):
            return Yr[:, None] + sum(single.Y(z - sj) - Yr[:, None] for sj in s)

        def fun(z, Y, p):
            deviation = Y[off] - reference(z)[off]
            rows = [system.rhs(Y[:N], speed(z, p))]
            rows += [(deviation * self.weight(z - sj)).sum(axis=0) for sj in s]
            return np.vstack(rows)

        def bc(Ya, Yb, p):
            km, Pm, *_ = split(system.end_state(self.rest, p[0] + p[-1] * (count > 1))[0])
            kp, Pp, *_ = split(system.end_state(self.rest, p[0])[0])
            if km != self.k or kp != self.k:
                raise WaveError("the end-state splitting changed during the iteration")
            return np.concatenate((self.C_left @ ((eye - Pm) @ (Ya[:N] - Yr)),
                                   self.C_right @ (Pp @ (Yb[:N] - Yr)), Ya[N:], Yb[N:]))

        if start is None:
            mesh = merged_grid(*[single.mesh + sj for sj in s], np.array([-L, L]))
            mesh = mesh[(mesh >= -L) & (mesh <= L)]
            Y0 = np.vstack((reference(mesh), np.zeros((count, mesh.size))))
            p0 = np.concatenate(([single.c], np.zeros(count - 1)))
        else:
            mesh, Y0, p0 = start.x, start.y, start.p
        try:
            sol = solve_bvp(fun, bc, mesh, Y0, p=p0, tol=tol, max_nodes=MAX_NODES)
        except (np.linalg.LinAlgError, StructureError) as error:
            raise WaveError(f"the pinned profile problem broke down: {error}") from error
        if not sol.success:
            raise WaveError(f"the pinned profile problem did not converge: {sol.message}")
        # Every pinned position must carry a pulse (a pulse can die in the wake of its neighbour).
        signal = np.abs(sol.y[off[self.component]] - Yr[off[self.component]])
        for sj in s:
            near = np.abs(sol.x - sj) < 0.25 * np.min(np.diff(-s)) if count > 1 else np.abs(sol.x - sj) < single.L
            if not near.any() or signal[near].max() < 0.5 * self.amplitude:
                raise WaveError("a pulse did not survive at its pinned position")
        return sol, s, L

    def defect(self, D, tol):
        """Speed defect sigma(D) of two pulses at distance D; NaN if the problem has no solution."""
        try:
            return float(self.solve([D], tol)[0].p[1])
        except WaveError:
            return np.nan


def _admissible(pinned, guess, period, tol):
    """Zero of sigma(D) nearest to `guess`, and the slope sigma'(D) there with an error estimate."""
    step = period / 12
    values = {}

    def sigma(i):
        if i not in values:
            values[i] = pinned.defect(guess + i * step, tol)
        return values[i]

    bracket = None
    for i in range(1, 16):                                            # search outwards, both sides
        for a, b in ((i - 1, i), (-i, 1 - i)):
            if np.isfinite(sigma(a)) and np.isfinite(sigma(b)) and sigma(a) * sigma(b) <= 0:
                bracket = (guess + a * step, guess + b * step)
                break
        if bracket:
            break
    if bracket is None:
        raise WaveError(f"no admissible pulse distance was found near {guess:g}: the speed defect "
                        "does not change sign (multi-pulses need oscillatory tails)")
    D = brentq(lambda d: pinned.defect(d, tol), *bracket, xtol=1e-9, rtol=1e-12)
    slopes = []
    for h in (period / 40, period / 80):
        slopes.append((pinned.defect(D + h, tol) - pinned.defect(D - h, tol)) / (2 * h))
    slope = slopes[1] + (slopes[1] - slopes[0]) / 3                   # Richardson
    return float(D), float(slope), float(abs(slopes[1] - slopes[0]))


def solve_multipulse(problem, system, single, tol=1e-8, check_tol=5e-10, tail_tol=1e-9):
    """N copies of the primary pulse `single` at admissible distances near wave['separation'].

    Returns a Wave.  Its info holds the distances, the remaining speed defects and
    the interaction eigenvalues predicted by the reduction, -sigma'(D_j).
    """
    spec = problem.wave
    copies = spec["copies"]
    rest = spec["left"]
    pinned = _Pinned(system, single, rest)
    ev = np.linalg.eigvals(system.end_state(rest, single.c)[0])
    lead = ev[np.argmin(np.where(ev.real > 0, ev.real, np.inf))]      # slowest rate behind the pulse
    oscillatory = abs(lead.imag) > 1e-9 * abs(lead)
    period = np.pi / abs(lead.imag) if oscillatory else 0.3 * float(np.mean(spec["separation"]))
    guesses = np.broadcast_to(np.asarray(spec["separation"], dtype=float), (copies - 1,))
    found = [_admissible(pinned, float(g), period, tol) for g in guesses]
    distances = np.array([f[0] for f in found])
    slopes = np.array([f[1] for f in found])

    sol, s, L = pinned.solve(distances, tol)
    for _ in range(6):                                                # more than two pulses: correct the gaps
        defects = sol.p[1:]
        if np.abs(defects).max() <= 1e-10 or copies == 2:
            break
        distances = distances - defects / slopes      # sigma_{j+1} depends on the gap D_j ahead of pulse j+1
        sol, s, L = pinned.solve(distances, tol)
    fine, _, _ = pinned.solve(distances, check_tol, start=sol)
    N, Yr = system.N, pinned.Yr
    probe = np.linspace(-L, L, 4001)
    profile_error = float(np.abs(fine.sol(probe)[:N] - sol.sol(probe)[:N]).max())
    tails = float(max(np.abs(fine.y[:N, 0] - Yr).max(), np.abs(fine.y[:N, -1] - Yr).max()))
    scale = max(1.0, float(np.abs(fine.y[:N]).max()))
    if tails > 1e3 * tail_tol * scale:
        raise WaveError(f"the multi-pulse has not reached its rest state at the ends of the domain "
                        f"(distance {tails:.2g})")
    defects = np.abs(fine.p[1:])
    info = dict(speed_is_unknown=True, copies=int(copies), positions=s, distances=distances,
                tail_rate=float(lead.real), tail_frequency=float(abs(lead.imag)),
                speed_defects=fine.p[1:], defect_slopes=slopes,
                predicted_interaction_eigenvalues=-slopes,
                prediction_errors=np.array([f[2] for f in found]),
                primary_speed=single.c, nodes=int(fine.x.size),
                residual_rms_max=float(fine.rms_residuals.max()),
                speed_error=abs(float(fine.p[0] - sol.p[0])),
                # The remaining defect acts like an error of the profile equation.
                profile_error=profile_error + float(defects.max()), tail_error=tails,
                amplitude=pinned.amplitude, slowest_rate=float(lead.real), started_from="pinned copies")
    return Wave(c=float(fine.p[0]), L=float(L), N=N, Y_minus=Yr, Y_plus=Yr, interpolant=fine.sol,
                mesh=fine.x, info=info, theta=system.theta)

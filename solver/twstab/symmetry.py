"""Waves with a rotation symmetry (DESIGN.md, section 2.16).

The equations u_t = G(u, u_x, ...) commute with the rotation of one pair of components
(a complex field psi = u_i + i u_j):  G(exp(theta J) u, ...) = exp(theta J) G(u, ...).
A wave is
    u(x, t) = exp(omega t J) U(x - c t),        G(U, U', ...) + c U' - omega J U = 0:
a standing wave in the frame that moves and rotates with it.  In that frame the linearisation
is an ordinary class-L operator; its kernel contains U' and J U.

rotation_pairs   the pairs of components with this symmetry
RotatingFrame    the problem in the co-rotating frame; the wave with two phase conditions
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from .problem import Problem
from .symbolic import LocalSystem, StructureError
from .wave import Wave, WaveError, solve_wave

FREQUENCY = sp.Symbol("frequency_of_the_wave", real=True)
UNFOLDING = sp.Symbol("unfolding_of_the_frequency", real=True)
ZERO = 1e-6           # an unfolding parameter above this (relative) means: no wave with the given values


def _jets(system):
    """Jet symbols U[j][k], k = 0, ..., order of component j."""
    sym = system.symbols
    return [[sym["low"][system.offset[j] + k] for k in range(system.mu[j])] + [sym["T"][j]]
            for j in range(system.n)]


def rotation_pairs(system, samples=4, seed=0):
    """Pairs (i, j) of components such that G commutes with their rotation.

    Infinitesimal test  DG(U)[J U] = J G(U), evaluated at random points with the parameter
    values of the problem.  Equations of first order in time without constraints only.
    """
    if getattr(system, "quadratic", False) or np.any(system.mass == 0):
        return []
    n, G, U = system.n, system.symbols["G"], _jets(system)
    arguments = [s for row in U for s in row] + list(system.params)
    rng = np.random.default_rng(seed)
    points = [np.concatenate((rng.uniform(0.5, 1.5, len(arguments) - len(system.params))
                              * rng.choice([-1.0, 1.0], len(arguments) - len(system.params)), system.theta))
              for _ in range(samples)]
    size = sp.lambdify(arguments, G, "numpy")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if system.mu[i] != system.mu[j]:
                continue
            defect = []
            for m in range(n):
                d = sum(sp.diff(G[m], U[j][k]) * U[i][k] - sp.diff(G[m], U[i][k]) * U[j][k]
                        for k in range(len(U[i])))
                defect.append(d + (G[j] if m == i else -G[i] if m == j else 0))
            f = sp.lambdify(arguments, defect, "numpy")
            with np.errstate(all="ignore"):
                values = [(np.abs(np.asarray(f(*p), dtype=float)).max(),
                           1.0 + np.abs(np.asarray(size(*p), dtype=float)).max()) for p in points]
            if all(np.isfinite(v) and v <= 1e-9 * s for v, s in values):
                pairs.append((i, j))
    return pairs


class RotatingFrame:
    """The problem of a wave with a frequency, in the frame that moves and rotates with the wave.

    `problem`, `system`: the equation in that frame, u_t = G - omega J u (omega is a parameter).
    The spectral part of the pipeline works on them without change.

    The wave.  The speed c and the frequency omega are unknowns, or they are given.  A given value is
    replaced by an unfolding parameter, which must come out as zero:
        G + c U' - omega J U + d1 S U' + d2 S J U = 0,        S = I + J.
    S makes the unfolded problem regular for gradient-like and for Hamiltonian equations
    (DESIGN 3.2).  Two phase conditions fix the translation and the rotation.
    """

    def __init__(self, problem, system):
        spec = problem.wave
        if getattr(system, "quadratic", False) or np.any(system.mass == 0) or problem.boundary is not None:
            raise StructureError("a frequency needs equations of first order in time, without constraints "
                                 "and without a free boundary")
        pairs = rotation_pairs(system)
        names = [u.func.__name__ for u in problem.unknowns]
        if not pairs:
            raise StructureError("the equations do not commute with the rotation of a pair of components")
        if len(pairs) > 1:
            raise StructureError("more than one rotation symmetry (pairs "
                                 + ", ".join(f"({names[i]}, {names[j]})" for i, j in pairs) + ") is not supported")
        (i, j), n = pairs[0], system.n
        self.pair, self.names = (i, j), (names[i], names[j])
        self.J = np.zeros((n, n))
        self.J[i, j], self.J[j, i] = -1.0, 1.0
        for side in ("left", "right"):
            if np.abs(self.J @ np.asarray(spec[side], dtype=float)).max() > 1e-12:
                raise StructureError(f"the {side} end state is not invariant under the rotation "
                                     "(dark solitons and fronts to a state of non-zero amplitude are not supported)")
        self.base_problem, self.base = problem, system
        self.given_speed = spec["speed"] is not None
        self.given_frequency = spec.get("frequency") is not None
        self.speed = float(spec["speed"] if self.given_speed else spec["speed_guess"])
        self.frequency = float(spec["frequency"] if self.given_frequency else spec["frequency_guess"])

        us, x, t = problem.unknowns, problem.x, problem.t
        U = _jets(system)
        back = {U[a][k]: (us[a] if k == 0 else sp.Derivative(us[a], (x, k)))
                for a in range(n) for k in range(len(U[a]))}
        G = sp.Matrix([g.xreplace(back) for g in system.symbols["G"]])
        J = sp.Matrix(n, n, lambda a, b: int(self.J[a, b]))
        u, ux = sp.Matrix(us), sp.Matrix([sp.Derivative(v, x) for v in us])
        rotating = G - FREQUENCY * J * u

        def build(rhs, parameters, wave):
            equations = [sp.Eq(sp.Derivative(us[a], t), sp.expand(rhs[a])) for a in range(n)]
            return Problem(path=problem.path, name=problem.name, x=x, t=t, equations=equations, unknowns=us,
                           parameters=parameters, wave=wave, options=problem.options)

        parameters = dict(problem.parameters)
        parameters[FREQUENCY] = self.frequency
        self.problem = build(rotating, parameters, dict(spec))
        self.system = LocalSystem(self.problem)

        # --- the problem that gives the wave ---------------------------------------------------
        S = sp.eye(n) + J
        rhs = S.inv() * (rotating + sp.Float(self.speed) * ux) if self.given_speed else rotating
        if self.given_frequency:
            rhs = rhs + UNFOLDING * (J if self.given_speed else S * J) * u
            parameters = dict(parameters)
            parameters[UNFOLDING] = 0.0
        wave = dict(spec, speed=None, speed_guess=0.0 if self.given_speed else self.speed)
        self.wave_problem = build(rhs, parameters, wave)
        self.wave_system = LocalSystem(self.wave_problem)
        self.unknown = str(UNFOLDING if self.given_frequency else FREQUENCY)

    def retarget(self, problem):
        """The same frame for another guess: `problem` must give the same values in `wave` (speed, frequency).

        The compiled systems are shared; only the wave specifications change.
        """
        import copy
        spec = problem.wave
        other = copy.copy(self)
        other.base_problem = problem
        other.speed = float(spec["speed"] if self.given_speed else spec["speed_guess"])
        other.problem = Problem(**{**self.problem.__dict__, "wave": dict(spec)})
        other.wave_problem = Problem(**{**self.wave_problem.__dict__, "wave": dict(
            spec, speed=None, speed_guess=0.0 if self.given_speed else other.speed)})
        return other

    def _solve(self, start_value, **kwargs):
        ws = self.wave_system
        ws.set_parameters({self.unknown: start_value})
        extra = dict(index=ws.parameter_index(self.unknown), J=self.J)
        wave = solve_wave(self.wave_problem, ws, extra=extra, **kwargs)
        return wave, float(ws.theta[extra["index"]])

    def solve_wave(self, **kwargs):
        """The wave as a Wave of `self.system`; the frequency is stored in the system."""
        start = 0.0 if self.given_frequency else self.frequency
        wave, value = self._solve(start, **kwargs)
        c_scale, w_scale = max(1.0, abs(self.speed)), max(1.0, abs(self.frequency))
        given = " and ".join(f"{name} {v:g}" for name, v, on in (("speed", self.speed, self.given_speed),
                                                                 ("frequency", self.frequency, self.given_frequency)) if on)
        unfolding = dict(speed=wave.c if self.given_speed else None, frequency=value if self.given_frequency else None)
        if any(v is not None and abs(v) > ZERO * s for v, s in ((unfolding["speed"], c_scale),
                                                                 (unfolding["frequency"], w_scale))):
            raise WaveError(f"there is no wave with the given {given} (unfolding parameters: "
                            f"{unfolding['speed']}, {unfolding['frequency']})")
        c = self.speed if self.given_speed else wave.c
        omega = self.frequency if self.given_frequency else value
        # An unknown must be isolated.  In a family of waves (Hamiltonian equations) the iteration returns
        # the member nearest to the guess, whatever the start value.  Test: is there also a wave with a
        # shifted value?  The unfolded problem answers this: its parameters vanish only on a wave.
        for name, unknown, scale in (("frequency", not self.given_frequency, w_scale),
                                     ("speed", not self.given_speed, c_scale)):
            if not unknown:
                continue
            spec = dict(self.base_problem.wave, speed=c, frequency=omega, frequency_guess=None)
            spec[name] += 0.05 * scale
            try:
                shifted = RotatingFrame(Problem(**{**self.base_problem.__dict__, "wave": spec}), self.base)
                other, d2 = shifted._solve(0.0, **kwargs)
            except (WaveError, StructureError, np.linalg.LinAlgError):
                continue                                   # no wave there: nothing against the one found
            if abs(other.c) <= ZERO * c_scale and abs(d2) <= ZERO * w_scale:
                raise WaveError(f"the {name} is not isolated (a family of waves): give both wave['speed'] "
                                "and wave['frequency']")
        Y = wave.Y(np.linspace(-wave.L, wave.L, 801))[self.system.offset]
        if np.abs(self.J @ Y).max() < 1e-6 * max(1.0, np.abs(Y).max()):
            raise WaveError("the rotating components vanish on the wave: it has no frequency")
        self.system.set_parameters({str(FREQUENCY): omega})
        self.frequency_found = omega
        info = dict(wave.info, speed_is_unknown=not self.given_speed, frequency=omega,
                    frequency_is_unknown=not self.given_frequency, unfolding_parameters=unfolding)
        return Wave(c=c, L=wave.L, N=wave.N, Y_minus=wave.Y_minus, Y_plus=wave.Y_plus,
                    interpolant=wave.interpolant, mesh=wave.mesh, info=info, theta=self.system.theta)

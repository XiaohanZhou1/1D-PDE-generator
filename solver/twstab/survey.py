"""Search for the waves of an equation and label each of them (DESIGN.md, sections 2.11 and 3.9).

Input: a problem file with the equations and the numerical values of all parameters,
and no `wave`.  The search is finite and explicit:

1. All constant solutions (equilibria) in a given range, each with the supremum of its
   essential spectrum: an equilibrium is an admissible end state only if that is negative.
2. Every wave that has an unstable equilibrium as an end state is unstable.  This is one
   statement for each unstable equilibrium; no wave is computed.
3. Candidates: a front between every pair of stable equilibria, a pulse on every stable
   equilibrium (with a free boundary: a semi-wave from every stable equilibrium).
4. Every candidate is attempted with a small family of generic guesses.  A wave that is
   found is labelled by label.analyse.  A candidate that is not found is reported as
   "not found"; this is not a statement that the wave does not exist.
5. Partner waves: a wave with exactly one unstable eigenvalue, a real one, can be the
   unstable half of a fold pair; the other half is then a stable wave.  The branch of such
   a wave is followed in every parameter, in both directions.  If it passes a fold and
   returns to the present parameter value, the wave there is a second wave of the same
   equation with the same end states.  (`partners="all"`: for every wave found.)

The list is not complete: waves outside the candidates, and waves that the generic
guesses do not reach, are missing.
"""
from __future__ import annotations

import time

import numpy as np
import sympy as sp

from . import __version__
from . import delay as delay_solver
from . import kernel as kernel_solver
from .asymptotic import essential_bound, quadratic_essential_bound
from .classify import classify
from .continuation import trace
from .freeboundary import solve_semi_wave
from .label import _clean, analyse
from .problem import ProblemError, load_problem, source_name
from .symbolic import LocalSystem, StructureError
from .symmetry import RotatingFrame, rotation_pairs
from .taxis import TaxisSystem
from .wave import WaveError, solve_wave, solve_wave_at_speed

DEFAULT_RANGE = 100.0         # equilibria are searched in [-range, range] for every component
SEARCH = dict(tol=1e-6, check_tol=1e-7, max_nodes=4000)     # a search attempt is cheap; a failing one ends quickly
PARTNER = dict(max_steps=16, growth=1.5, min_step=0.2)      # small budget for each parameter and direction


def find_equilibria(system, lower, upper, tol=1e-12):
    """Constant solutions G(u) = 0 inside the box [lower, upper], by Newton iteration from a lattice.

    The lattice is dense near zero and sparse far away (sinh spacing).  For one component
    the sign changes of G on a fine grid are used as well.  The result is every root that
    the iteration reaches; a root far from all starting points can be missed.
    """
    n = system.n
    lower, upper = np.broadcast_to(lower, (n,)).astype(float), np.broadcast_to(upper, (n,)).astype(float)
    per_axis = {1: 4001, 2: 61, 3: 17, 4: 9}.get(n, 5)

    def axis(lo, hi):
        scale = max(abs(lo), abs(hi), 1e-12) / 100
        t = np.linspace(np.arcsinh(lo / scale), np.arcsinh(hi / scale), per_axis)
        return scale * np.sinh(t)

    axes = [axis(lower[j], upper[j]) for j in range(n)]
    U = np.array(np.meshgrid(*axes, indexing="ij")).reshape(n, -1)

    def residual(U):
        Y = np.zeros((system.N, U.shape[1]))
        Y[system.offset] = U
        return system._r(*Y, 0.0, *system.theta), Y

    if n == 1:                                        # brackets from the sign changes
        r = residual(U)[0][0]
        i = np.nonzero(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
        exact = U[0][r == 0.0]
        U = np.concatenate((U[0], (U[0][i] + U[0][i + 1]) / 2, exact))[None, :]
    span = (upper - lower)[:, None]
    step = np.full_like(U, np.inf)
    for _ in range(60):
        r, Y = residual(U)
        zero = np.zeros((n, U.shape[1]))
        J = system._a_k(*Y, *zero, 0.0, *system.theta).reshape(system.K + 1, n, n, -1)[0].transpose(2, 0, 1)
        try:
            step = np.linalg.solve(J + 1e-14 * np.eye(n)[None], r.T[..., None])[..., 0].T
        except np.linalg.LinAlgError:
            break
        step = np.clip(np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0), -span / 4, span / 4)
        U = U - step
    size = np.maximum(1.0, np.abs(U).max(axis=0))
    converged = np.abs(step).max(axis=0) < tol * size
    inside = np.all((U >= lower[:, None] - 1e-9) & (U <= upper[:, None] + 1e-9), axis=0)
    # A point where the functions overflow gives a step of zero: it must also solve the equations.
    with np.errstate(all="ignore"):
        r = residual(U)[0]
    solved = np.all(np.isfinite(r), axis=0) & (np.abs(np.nan_to_num(r)).max(axis=0) < 1e-8 * size)
    good = converged & inside & solved
    roots = []
    for u in U[:, good].T:
        if all(np.abs(u - v).max() > 1e-7 * max(1.0, np.abs(u).max()) for v in roots):
            roots.append(u)
    roots = [np.where(np.abs(u) < 1e-12, 0.0, u) for u in roots]
    return sorted(roots, key=lambda u: tuple(u))


class _Kit:
    """What the survey needs from the sub-solver of the problem."""

    def __init__(self, problem, sub_solver):
        self.sub = sub_solver
        if sub_solver == "D":
            self.full = delay_solver.DelaySystem(problem)
            self.local = self.full.system0
        elif sub_solver == "N":
            self.full = kernel_solver.KernelSystem(problem)
            self.local = self.full.system0
        else:
            self.full = self.local = LocalSystem(problem)
        self._frames = {}
        self.taxis = None          # transport towards a component without diffusion: system in flux variables
        self.rotation = []         # pairs of components with a rotation symmetry: waves with the frequency 0 only
        if sub_solver == "L" and problem.boundary is None and not self.local.conservative:
            try:
                self.taxis = self.full = TaxisSystem(problem, self.local)
            except StructureError:
                self.rotation = rotation_pairs(self.local)

    @property
    def state_system(self):
        """The system whose state the waves of the search have."""
        return self.taxis if self.taxis is not None else self.local

    def constant_coefficients(self, u, c):
        system = self.state_system
        return system.coefficients(system.state(u), c)[:, 0]

    def attempt(self, problem):
        """Compute the wave of `problem` (it carries one guess) with the compiled system.  Raises WaveError."""
        if self.rotation:
            # The frame depends on the wave only through the pattern of given and unknown values:
            # it is compiled once and reused for the other guesses.
            spec = problem.wave
            key = (spec["speed"], spec.get("frequency"), spec.get("frequency_guess") is None)
            if key not in self._frames:
                self._frames[key] = RotatingFrame(problem, self.local)
            return self._frames[key].retarget(problem).solve_wave(**SEARCH)
        if getattr(self.full, "quadratic", False) and problem.wave["speed"] is not None:
            return solve_wave_at_speed(problem, self.full, **SEARCH)      # no damping: the standing wave
        if self.sub == "D":                              # coarse grid: the full pipeline solves again
            self.full.problem0.wave = problem.wave
            return delay_solver.solve_wave(problem, self.full, h_factor=0.12, max_nodes=SEARCH["max_nodes"])
        if self.sub == "N":
            self.full.problem0.wave = problem.wave
            return kernel_solver.solve_wave(problem, self.full, h_factor=0.12, max_nodes=SEARCH["max_nodes"])
        if problem.boundary is not None:
            return solve_semi_wave(problem, self.full, **SEARCH)
        return solve_wave(problem, self.full, **SEARCH)

    def signature(self, wave, base):
        """Speed and signed extremum of every component: equal for two computations of one wave."""
        U = wave.Y(wave.mesh)[self.state_system.offset] - np.asarray(base, dtype=float)[:, None]
        extremes = U[np.arange(U.shape[0]), np.abs(U).argmax(axis=1)]
        return tuple(np.round(np.concatenate(([wave.c], extremes)), 5))

    def sup_re(self, u):
        """Supremum of the real part of the essential spectrum of the constant state u."""
        spec = dict(left=list(u), right=list(u))
        if self.sub == "D":
            return delay_solver.end_bounds(self.full, spec, 1.0)[0]
        if self.sub == "N":
            return kernel_solver.end_bounds(self.full, spec, 1.0)[0]
        if getattr(self.local, "quadratic", False):                   # speed 0: c^2 must stay below the wave speeds
            Y = self.local.state(u)
            end = (self.local.coefficients(Y, 0.0)[:, 0], self.local.damping(Y, 0.0)[0], self.local.inertia)
            s, sides = quadratic_essential_bound(end, end, 0.0)
        else:
            a = self.constant_coefficients(u, 1.0)                        # the speed does not change real parts
            s, sides = essential_bound(a, a, first_order=self.state_system.first_order_components,
                                       mass=self.local.mass)
        # A value within the rounding error of the dispersion curves is zero (no dissipation: marginal).
        return 0.0 if abs(s) <= max(v["rounding"] for v in sides.values()) else s

    def scales(self, u):
        """Length and speed scales of the linearisation at u: sqrt(D / rate) and sqrt(D rate)."""
        a = self.constant_coefficients(u, 0.0 if getattr(self.local, "quadratic", False) else 1.0)
        rate = max(np.linalg.norm(a[0], 2), 1e-12)
        if self.sub == "N" and not self.full.diffusive:               # no diffusion: the kernels set the length
            return min(self.full.length), min(self.full.length) * rate
        if a.shape[0] <= 2:
            return 1.0 / np.sqrt(rate), np.sqrt(rate)
        top = a.shape[0] - 1                                          # highest order K: width (|a_K| / rate)^(1/K)
        width = (max(np.linalg.norm(a[top], 2), 1e-12) / rate) ** (1.0 / top)
        return width, width * rate

    def reflection_symmetric(self):
        """True if x -> -x maps the equations to themselves (then a wave and its mirror image are one wave)."""
        if self.sub == "F":
            return False                                  # the domain x < s(t) is not symmetric
        if self.sub == "N":                               # every kernel must be even
            s = np.linspace(0.0, max(self.full.radius), 201)
            if any(np.abs(K(s) - K(-s)).max() > 1e-12 * np.abs(K(s)).max() for K in self.full.kernels):
                return False
        local = self.local if self.sub == "L" else self.full.local    # the delayed values do not depend on x
        sym = local.symbols
        flip, k = {}, 0
        for j in range(local.n):
            for order in range(local.mu[j]):
                if order % 2:
                    flip[sym["low"][k]] = -sym["low"][k]
                k += 1
            if local.mu[j] % 2:
                flip[sym["T"][j]] = -sym["T"][j]
        flip[sym["c"]] = -sym["c"]                     # the mirror image moves with -c (u_t = -c U' is in G with u_tt)
        try:
            return all(sp.expand(g.xreplace(flip) - g) == 0 for g in sym["G"])
        except Exception:                             # a comparison that SymPy cannot decide
            return False


def _partner(kit, wave, a, b):
    """A second wave with the same end states at the same parameter values, or (None, None).

    The branch of `wave` is followed in each parameter, upwards and downwards, within
    100 percent of its value.  A branch that passes a fold and comes back to the present
    value ends at another wave.  The first such wave is returned.  Class L only.
    """
    system = kit.full
    theta0 = system.theta
    try:
        for i, parameter in enumerate(system.params):
            p0 = float(theta0[i])
            span = abs(p0) if p0 != 0 else 1.0
            for direction in (1.0, -1.0):
                system.theta = theta0
                try:
                    branch = trace(system, a, a if b is None else b, wave, str(parameter), direction=direction,
                                   until=p0, step=0.05 * span, value_range=(p0 - span, p0 + span), **PARTNER)
                except (WaveError, StructureError, np.linalg.LinAlgError, ArithmeticError):
                    continue
                if branch.final is not None and branch.folds:
                    return branch.final, dict(parameter=str(parameter), fold=branch.folds[0]["value"],
                                              steps=len(branch.points) - 1)
    finally:
        system.theta = theta0
    return None, None


def _single_real_instability(record):
    """True if the complete count shows exactly one unstable eigenvalue and it is real."""
    first = record.get("pass_1") or {}
    unstable = [r for r in record.get("roots", []) if r["status"] == "unstable"]
    return bool(first.get("complete_count")) and len(unstable) == 1 and unstable[0]["value"][1] == 0


def _named(problem, u):
    return {v.func.__name__: float(x) for v, x in zip(problem.unknowns, u)}


def _attempts(kind, kit, a, b, others, symmetric):
    """Groups of wave specifications (file syntax, without the end states) for one candidate.

    Every group aims at one wave: the search of a group stops at the first success.
    A front or a semi-wave has one group.  A pulse has one group for each direction in
    state space: towards each other equilibrium, or along the axes if there is none.
    """
    width, speed = np.mean([kit.scales(u) for u in ((a, b) if b is not None else (a,))], axis=0)
    zero_speed = not kit.state_system.first_order_components   # c = 0 is singular with a diffusion-free component
    speeds = ([0.0] if zero_speed else []) + [0.5 * speed, -0.5 * speed]
    if kind in ("front", "semi"):
        return [[dict(speed_guess=c, width=w) for w in (1.5 * width, 3.0 * width, 0.7 * width) for c in speeds]]
    if symmetric and zero_speed:
        speeds = [0.0, 0.5 * speed]
    directions = [v - a for v in others] or [np.eye(len(a))[j] * sign for j in range(len(a)) for sign in (1.0, -1.0)]
    return [[dict(speed_guess=c, width=w, amplitude=alpha * direction)
             for alpha in (1.0, 0.5) for w in (1.5 * width, 3.0 * width) for c in speeds[:2]]
            for direction in directions]


def survey(path, **overrides):
    """Search for the waves of the equation in `path` and label them.  Returns a JSON-ready dictionary.

    Keyword arguments: `parameters={name: value}`, `equilibria_range=number`, `partners` (True: partner
    search for waves with one real unstable eigenvalue; "all": for every wave; False: none), and every
    option of label.analyse.
    """
    start = time.perf_counter()
    given = overrides.pop("parameters", None)
    out = dict(solver_version=__version__, file=source_name(path), waves=[], families=[], not_found=[], equilibria=[])

    def finish(note=None):
        out.update(note=note, seconds=time.perf_counter() - start)
        return _clean(out)

    try:
        problem = load_problem(path, given)
    except ProblemError as error:
        return finish(f"input file: {error}")
    if problem.wave is not None:
        return finish("the file names a wave: use label.analyse for it")
    out.update(name=problem.name, parameters={str(k): v for k, v in problem.parameters.items()})
    cls = classify(problem)
    out["classification"] = cls.as_dict()
    if not cls.supported:
        return finish(f"sub-solver {cls.sub_solver} is not available: " + "; ".join(cls.reasons))
    try:
        kit = _Kit(problem, cls.sub_solver)
    except StructureError as error:
        return finish(f"structure: {error}")
    if kit.local.conservative:
        return finish("conservation form: every constant is an equilibrium, so the waves are not determined by the "
                      "equation alone; give the end states: wave = dict(kind='front', left={...}, right={...})")
    names = [u.func.__name__ for u in problem.unknowns]
    if len(kit.rotation) > 1:
        return finish("more than one rotation symmetry: the automatic search is not available")
    rotating = set(kit.rotation[0]) if kit.rotation else set()
    if rotating:
        # Travelling waves only: the frequency of every wave of the search is 0 (DESIGN 2.16).
        out["rotation"] = dict(components=[names[k] for k in kit.rotation[0]], frequency=0.0)
    undamped = getattr(kit.local, "quadratic", False)

    # --- 1. equilibria and their essential spectrum -----------------------------------------
    limit = float(overrides.pop("equilibria_range", problem.options.get("equilibria_range", DEFAULT_RANGE)))
    states = find_equilibria(kit.local, -limit, limit)
    out["search"] = dict(equilibria_range=limit, method="Newton iteration from a lattice")
    stable, marginal, points, circles = [], [], [], {}
    for u in states:
        if rotating and max(abs(u[k]) for k in rotating) > 1e-9:
            # The lattice finds many points of one circle of equilibria.  One representative is kept:
            # it gives a direction for the guess of a pulse, and it is not an end state (DESIGN 2.16).
            i, j = kit.rotation[0]
            v = np.array(u, dtype=float)
            v[i], v[j] = float(np.hypot(u[i], u[j])), 0.0
            key = tuple(np.round(v, 6))
            if key not in circles:
                circles[key] = v
                points.append(v)
                out["equilibria"].append(dict(state=_named(problem, v), sup_re=None,
                                              kind="circle of equilibria (rotation), not an end state"))
            continue
        points.append(u)
        try:
            s = float(kit.sup_re(u))
        except (np.linalg.LinAlgError, ArithmeticError):
            s = np.nan
        scale = max(np.linalg.norm(kit.local.reaction_jacobian(u), 2), 1e-300)
        kind = "stable" if s < -1e-9 * scale else "unstable" if s > 1e-9 * scale else "marginal"
        out["equilibria"].append(dict(state=_named(problem, u), sup_re=s, kind=kind))
        if kind == "stable":
            stable.append(u)
        elif kind == "unstable":
            out["families"].append(dict(state=_named(problem, u), label="UNSTABLE", sup_re=s,
                                        statement="every wave that has this state as an end state is unstable "
                                                  "(essential spectrum)"))
        elif np.isfinite(s):
            # The essential spectrum touches the imaginary axis (no damping, or a component that does not
            # relax): the state can be an end state; its waves get a label in the neutral mode.
            marginal.append(u)
        else:
            out["families"].append(dict(state=_named(problem, u), label="UNCERTAIN", sup_re=s,
                                        statement="the essential spectrum of this state could not be computed"))
    usable = stable + marginal
    is_marginal = lambda u: any(u is v for v in marginal)

    # --- 2. candidates --------------------------------------------------------------------------
    symmetric = kit.reflection_symmetric()
    out["search"]["reflection_symmetric"] = bool(symmetric)
    candidates = []
    if problem.boundary is not None:
        candidates += [("semi", a, None) for a in stable]
    else:
        for i, a in enumerate(usable):
            for j, b in enumerate(usable):
                if i != j and (i < j or not symmetric):
                    candidates.append(("front", a, b))
        candidates += [("pulse", a, None) for a in usable]

    # --- 3. attempts ---------------------------------------------------------------------------
    partners = overrides.pop("partners", True)
    doubles = int(overrides.pop("double_pulses", problem.options.get("double_pulses", 3)))
    plain = cls.sub_solver == "L" and kit.taxis is None and not rotating     # continuation and multi-pulses
    for kind, a, b in candidates:
        others = [u for u in points if not np.array_equal(u, a)]
        groups = _attempts(kind, kit, a, b, others, symmetric)
        # Second order in time without damping at an end state: the waves form a family in the speed
        # (Lorentz boost).  The standing wave is its representative (DESIGN 2.15).
        standing = bool(undamped and (is_marginal(a) or (b is not None and is_marginal(b))))
        found, reasons = {}, set()
        for group in groups:
            for trial in group:
                if kind == "front":
                    wave = dict(kind="front", left=_named(problem, a), right=_named(problem, b))
                elif kind == "semi":
                    wave = dict(kind="semi", far=_named(problem, a))
                else:
                    wave = dict(kind="pulse", rest=_named(problem, a),
                                amplitude=_named(problem, trial["amplitude"]))
                wave.update(speed_guess=trial["speed_guess"], width=trial["width"])
                if standing:
                    if trial["speed_guess"] != group[0]["speed_guess"]:
                        continue                              # one trial for each width and amplitude
                    del wave["speed_guess"]
                    wave["speed"] = 0.0
                if rotating:
                    wave["frequency"] = 0.0
                try:
                    computed = kit.attempt(load_problem(path, given, wave=wave))
                except (WaveError, StructureError, ProblemError, np.linalg.LinAlgError, ArithmeticError) as error:
                    reasons.add(str(error)[:100])
                    continue
                found.setdefault(kit.signature(computed, a), (wave, computed, None))   # two directions, one pulse
                break
        entry = dict(kind=kind, left=_named(problem, a), right=_named(problem, b if b is not None else a))
        if not found:
            out["not_found"].append(dict(entry, attempts=sum(len(g) for g in groups), reasons=sorted(reasons)))
        queue = list(found.items())
        while queue:
            signature, (wave, computed, how) = queue.pop(0)
            # The full pipeline, started from the wave of the search where the sub-solver allows it.
            extra = dict(start=computed) if cls.sub_solver == "L" and not rotating and not standing else {}
            record = analyse(path, parameters=given, wave=wave, **extra, **overrides)
            out["waves"].append(dict(entry, extremum=_named(problem, np.asarray(a) + np.asarray(signature[1:])),
                                     found_by="direct search" if how is None else "continuation through a fold",
                                     fold=how, label=record["label"], qualifier=record.get("qualifier"),
                                     reason=record["reason"],
                                     speed=record.get("wave", {}).get("speed", computed.c), record=record))
            if standing:
                out["waves"][-1]["note"] = "the standing wave stands for the family of waves in the speed"
            if kind == "pulse" and plain and not undamped and doubles > 0:
                _double_pulses(out, path, given, overrides, kit, entry, wave, computed, a, record, doubles)
            if not plain or standing or not partners or how is not None:
                continue                              # the partner of a partner is the wave we started from
            if partners != "all":
                if record["label"] != "UNSTABLE":
                    continue
                if record["pass_1"].get("shortcut"):  # the shortcut does not count: count now
                    record = analyse(path, parameters=given, wave=wave, start=computed,
                                     **{**overrides, "parity_shortcut": False})
                if not _single_real_instability(record):
                    continue
            other, how = _partner(kit, computed, a, b)
            if other is not None and kit.signature(other, a) not in found:
                found[kit.signature(other, a)] = (wave, other, how)
                queue.append((kit.signature(other, a), found[kit.signature(other, a)]))
    return finish()


def _double_pulses(out, path, given, overrides, kit, entry, wave, single, rest, record, count):
    """Double pulses made of two copies of a pulse with an oscillatory tail (DESIGN 2.8, 2.11).

    There are infinitely many multi-pulses.  The list holds the double pulses at the `count` smallest
    admissible distances that the search finds.  Copies of an unstable pulse are unstable: one statement.
    """
    ev = np.linalg.eigvals(kit.full.end_state(rest, single.c)[0])
    behind = ev[ev.real > 0]
    if behind.size == 0:
        return
    lead = behind[np.argmin(behind.real)]                    # slowest rate behind the pulse
    if abs(lead.imag) <= 1e-9 * abs(lead):
        return                                               # monotone tail: no multi-pulses of this kind
    if record["label"] != "STABLE":
        if record["label"] == "UNSTABLE":
            out["families"].append(dict(state=entry["left"], label="UNSTABLE", sup_re=None,
                                        text=f"every multi-pulse made of copies of the unstable pulse (c = {single.c:.6g})",
                                        statement="a multi-pulse inherits the unstable eigenvalue of each copy"))
        return
    period = np.pi / abs(lead.imag)
    z = single.mesh
    U = np.abs(single.Y(z)[kit.state_system.offset] - np.asarray(rest, dtype=float)[:, None]).max(axis=0)
    core = z[U > 0.02 * U.max()]
    width = float(core.max() - core.min())
    found = []
    for step in range(2 * count + 2):                        # from the width of the pulse upwards
        if len(found) >= count:
            break
        guess = width + step * period
        if guess <= 0:
            continue
        result = analyse(path, parameters=given, wave=dict(wave, copies=2, separation=guess), start=single,
                         **overrides)
        distances = (result.get("wave") or {}).get("distances")
        if not distances:
            continue                                         # no admissible distance near this guess
        distance = float(distances[0])
        if any(abs(distance - d) < 0.25 * period for d in found):
            continue
        found.append(distance)
        out["waves"].append(dict(entry, copies=2, distance=distance, extremum=None,
                                 found_by="two copies of the pulse above", fold=None, label=result["label"],
                                 qualifier=result.get("qualifier"), reason=result["reason"],
                                 speed=result.get("wave", {}).get("speed", single.c), record=result))


def summary(result):
    """One line for each equilibrium, wave, family and candidate that was not found."""
    def show(state):
        return "(" + ", ".join(f"{k} = {v:.6g}" for k, v in state.items()) + ")"

    lines = []
    if result.get("note"):
        return [f"{result.get('name', result['file'])}: {result['note']}"]
    lines.append(f"{result['name']}: {len(result['equilibria'])} equilibria in the range "
                 f"+-{result['search']['equilibria_range']:g}: "
                 + "; ".join(f"{show(e['state'])} {e['kind']}" for e in result["equilibria"]))
    for w in result["waves"]:
        where = (f"{show(w['left'])} -> {show(w['right'])}" if w["kind"] == "front"
                 else f"on {show(w['left'])}, extremum {show(w['extremum'] or {})}" if w["kind"] == "pulse"
                 else f"behind the free boundary, far state {show(w['left'])}")
        note = f"   ({w['reason']})" if w["label"] == "UNCERTAIN" else ""
        if w.get("fold"):
            note += f"   [second wave of this branch; fold at {w['fold']['parameter']} = {w['fold']['fold']:.6g}]"
        if w.get("copies"):
            where = f"two copies on {show(w['left'])}, distance {w['distance']:.6g}"
        if w.get("qualifier") and w["label"] == "STABLE":
            note += f" ({w['qualifier']})"                    # a weaker sense of STABLE: always printed
        if w.get("note"):
            note += f"   [{w['note']}]"
        speed = 0.0 if abs(w["speed"]) < 1e-9 else w["speed"]
        lines.append(f"  {w['kind']:6s} {where}, c = {speed:.6g}: {w['label']}{note}")
    for f in result["families"]:
        lines.append(f"  {f.get('text') or 'every wave that ends at ' + show(f['state'])}: {f['label']}")
    for c in result["not_found"]:
        where = (f"{show(c['left'])} -> {show(c['right'])}" if c["kind"] == "front"
                 else f"on {show(c['left'])}" if c["kind"] == "pulse" else f"far state {show(c['left'])}")
        lines.append(f"  {c['kind']:6s} {where}: not found ({c['attempts']} attempts)")
    return lines

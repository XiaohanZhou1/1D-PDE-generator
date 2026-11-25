"""Two-pass labelling of one travelling wave (DESIGN.md, sections 3.1 and 3.3).

Pass 1 runs for every wave on two mesh resolutions.  A confirmed unstable root
ends the analysis with UNSTABLE.  Every other outcome goes to pass 2, which adds
a third resolution, two changes of the domain, an independent finite-difference
discretisation, the rank certificate of the boundary rows and the parity index.
"""
from __future__ import annotations

import time

import numpy as np
import sympy as sp
from scipy import sparse

from . import __version__
from .asymptotic import axis_tolerance, essential_bound, quadratic_essential_bound
from .classify import classify
from .conservation import ConservativeSystem
from .conservative_front import exact_wave, continued_origin
from .conservative_unfolding import unfold_wave
from .continuation import trace
from .contour import ContourError, box, circle, isolate, locate, winding
from .evans import Asymptotics, Evans, SplittingError, TouchingAsymptotics, graded_cells, wave_variation
from . import delay as delay_solver
from . import kernel as kernel_solver
from .multipulse import solve_multipulse
from .freeboundary import (BoundaryAsymptotics, boundary_bounds, boundary_region, build_engine,
                           fd_operator_boundary, solve_semi_wave)
from .problem import ProblemError, load_problem, source_name
from .scout import coefficient_bounds, fd_companion, fd_operator, fd_probe, scaled_bounds, search_region
from .symbolic import LocalSystem, StructureError
from .symmetry import RotatingFrame, rotation_pairs
from .taxis import TaxisSystem
from .wave import WaveError, solve_wave, solve_wave_at_speed
from .weighted import WeightedSystem, find_rate, translation_mode, weighted_wave

DEFAULTS = dict(wave_tol=1e-8, wave_check_tol=5e-10, tail_tol=1e-9, L_max=2000.0, h_factor=0.4, max_cells=8000,
                margin_factor=10.0, phase_limit=0.55, initial_panels=64, fd_unknowns=2400,
                region=None, two_pass=True, max_refinements=2,
                max_nodes=40000,         # node limit of the collocation solve of the wave
                mesh_cap=4.0,            # largest step of the graded Evans mesh, in units of h; 1 = uniform
                parity_shortcut=True,    # a confirmed real root from the parity index ends pass 1
                weighted=True,           # essential spectrum in Re > 0: also give the label in a weighted space
                neutral_threshold=None,  # essential spectrum on the axis: growth rates above it are examined
                conservation_method="auto",  # "weights", "axis" (determinant on the closed half-plane), or "auto"
                conservation_origin_radius=None,  # opt-in continuation for supported exact undercompressive fronts
                default_region=True,     # no estimate of the search rectangle: use the default rule and say so
                seeds="fd")              # "fd": difference probe, then subdivision; "evans": subdivision only

SPACE = "unweighted L2(R; C^n)"


def _clean(obj):
    """Convert numpy and complex values to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (complex, np.complexfloating)):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, sp.Basic):
        return _clean(complex(obj) if obj.is_number and not obj.is_real else float(obj)) if obj.is_number else str(obj)
    return obj


class _Run:
    """Data shared by the two passes."""

    expected = 1              # zeros of D in the origin disk: the translation zero

    def __init__(self, system, wave, asym, opts, delta, x_max, y_max, lam0, h1):
        self.system, self.wave, self.asym, self.opts = system, wave, asym, opts
        self.delta, self.r0 = delta, delta / 2
        self.x_max, self.y_max, self.lam0, self.h1 = x_max, y_max, lam0, h1
        self.wave_error = lam0 * wave.info["profile_error"]
        # Multi-pulse: interaction eigenvalues predicted by the reduction, with their numerical errors.
        self.predicted = wave.info.get("predicted_interaction_eigenvalues")
        self.predicted_error = wave.info.get("prediction_errors")
        self.cap = float(opts["mesh_cap"])
        self.variation = wave_variation(system, wave) if self.cap > 1.0 else None
        self._engines = {}
        self.evaluations = 0
        self.subdivision = None

    def cells(self, h):
        """Number of cells of the Evans mesh with step h on the full domain."""
        if self.variation is None:
            return 2 * self.wave.L / h
        return graded_cells(self.variation, self.wave.L, self.wave.L, h, self.cap)[2]

    def engine(self, h, La=None, Lb=None):
        La = self.wave.L if La is None else La
        Lb = self.wave.L if Lb is None else Lb
        key = (round(h, 12), La, Lb)
        if key not in self._engines:
            self._engines[key] = Evans.build(self.system, self.wave, self.asym, La, Lb, h,
                                             self.cap, self.variation)
        return self._engines[key]

    def box_count(self, engine):
        out = winding(engine, box(-self.delta, self.x_max, self.y_max, self.delta),
                      self.opts["initial_panels"], self.opts["phase_limit"])
        self.evaluations += out["evaluations"]
        return out

    def disk_count(self, engine, center, radius):
        out = winding(engine, circle(center, radius), 16, self.opts["phase_limit"])
        self.evaluations += out["evaluations"]
        return out

    def fd_matrix(self, factor=1.0):
        # At least fd_unknowns unknowns; on a long domain a step of 0.6 h1 (h1: Evans step).
        points = max(self.opts["fd_unknowns"] / self.system.n, 2 * self.wave.L / (0.6 * self.h1))
        nz = int(np.clip(factor * points, 160, 20000))
        if getattr(self.system, "quadratic", False):   # second order in time: companion form (A, B)
            return nz, fd_companion(self.system, self.wave, self.wave.L, self.wave.L, nz)
        return nz, fd_operator(self.system, self.wave, self.wave.L, self.wave.L, nz)

    def probe(self, A, shifts):
        """Difference probe.  Constraint components make the eigenvalue problem a generalised one."""
        if isinstance(A, tuple):
            return fd_probe(A[0], shifts, mass=A[1])
        if np.all(self.system.mass != 0):
            return fd_probe(A, shifts)
        nz = A.shape[0] // self.system.n
        return fd_probe(A, shifts, mass=sparse.diags(np.repeat(self.system.mass, nz), format="csc"))

    def domain_changes(self):
        """Shorter domains for the audit, one side at a time."""
        L = self.wave.L
        return (L, 0.75 * L), (0.75 * L, L)

    def shift_groups(self):
        """Probe shifts: a real group (right to left) and a group on the imaginary axis."""
        real, x = [], self.x_max
        while x > 2 * self.delta:
            real.append(x)
            x /= 4
        real.append(self.r0)
        imag, y = [], 4 * self.delta
        while y < self.y_max:
            imag.append(1j * y)
            y *= 4
        return real, imag


class _RunBoundary(_Run):
    """Free boundary: the domain is [-L, 0] and there is no translation zero."""

    expected = 0

    def cells(self, h):
        if self.variation is None:
            return self.wave.L / h
        return graded_cells(self.variation, self.wave.L, 0.0, h, self.cap)[2]

    def engine(self, h, La=None, Lb=None):
        La = self.wave.L if La is None else La
        key = (round(h, 12), La)
        if key not in self._engines:
            self._engines[key] = build_engine(self.system, self.wave, self.asym, La, h,
                                              self.cap, self.variation)
        return self._engines[key]

    def fd_matrix(self, factor=1.0):
        points = max(self.opts["fd_unknowns"] / self.system.n, self.wave.L / (0.6 * self.h1))
        nz = int(np.clip(factor * points, 160, 20000))
        return nz, fd_operator_boundary(self.system, self.wave, self.wave.L, nz)

    def probe(self, pair, shifts):
        """The boundary rows make the eigenvalue problem a generalised one: pair = (A, B)."""
        return fd_probe(pair[0], shifts, mass=pair[1])

    def domain_changes(self):
        return ((0.75 * self.wave.L, None),)


class _RunBand(_Run):
    """Sub-solvers D and N: uniform grids, the band-determinant engine, a sparse matrix as probe.

    `kit` is the module of the sub-solver (delay.py or kernel.py).
    """

    def __init__(self, kit, system, wave, opts, delta, x_max, y_max, lam0, h1):
        super().__init__(system, wave, None, dict(opts, mesh_cap=1.0), delta, x_max, y_max, lam0, h1)
        self.kit = kit

    def engine(self, h, La=None, Lb=None):
        La = self.wave.L if La is None else La
        Lb = self.wave.L if Lb is None else Lb
        key = (round(h, 12), La, Lb)
        if key not in self._engines:
            self._engines[key] = self.kit.engine(self.system, self.wave, La, Lb, h)
        return self._engines[key]

    def fd_matrix(self, factor=1.0):
        points = max(self.opts["fd_unknowns"] / (4 * self.system.n), 2 * self.wave.L / (0.6 * self.h1))
        nz = int(np.clip(factor * points, 160, 4000))
        return nz, self.kit.probe_matrix(self.system, self.wave, self.wave.L, self.wave.L, nz)

    def probe(self, A, shifts):
        return fd_probe(A, shifts)


def _polish(run, coarse, fine, seeds, known):
    """Polish seeds with the Evans function.  Returns new roots in the upper half-plane."""
    roots = []
    for s in sorted(seeds, key=lambda v: -v.real):
        step = 0.02 * max(abs(s), run.delta)
        try:
            z = locate(fine, s, step)
        except (ContourError, SplittingError, ArithmeticError):
            continue
        if abs(z - s) > max(0.3 * abs(s), 2 * run.delta):
            continue                                  # the iteration left the neighbourhood
        if z.real <= -run.delta or z.real > run.x_max or abs(z.imag) > run.y_max:
            continue
        if abs(z) <= run.r0:
            continue                                  # the translation zero
        if abs(z.imag) < 1e-8 * max(1.0, abs(z)):
            z = complex(z.real, 0.0)
        if z.imag < 0:
            z = z.conjugate()
        if any(abs(z - r["value"]) < 1e-6 * max(1.0, abs(z)) for r in roots + known):
            continue
        try:
            z_coarse = locate(coarse, z, step)
        except (ContourError, SplittingError, ArithmeticError):
            z_coarse = None
        roots.append(dict(value=z, coarse=z_coarse, seed=complex(s)))
    return roots


def _qualify(run, coarse, fine, roots):
    """Local winding numbers, error estimates and status of each located root."""
    margin = run.opts["margin_factor"]
    centres = [0j]
    for r in roots:
        centres.append(r["value"])
        if r["value"].imag > 0:
            centres.append(r["value"].conjugate())
    for r in roots:
        z = r["value"]
        gap = min(abs(z - other) for other in centres if other != z)
        radius = min(run.r0, 0.25 * gap)
        r["radius"] = radius
        try:
            r["winding_fine"] = run.disk_count(fine, z, radius)["count"]
            r["winding_coarse"] = run.disk_count(coarse, z, radius)["count"]
        except (ContourError, SplittingError):
            r["winding_fine"] = r["winding_coarse"] = None
        r["error"] = (abs(z - r["coarse"]) if r["coarse"] is not None else np.inf) + run.wave_error
        simple = r["winding_fine"] == 1 and r["winding_coarse"] == 1
        if simple and z.real > margin * r["error"]:
            r["status"] = "unstable"
        elif simple and z.real < -margin * r["error"]:
            r["status"] = "stable"
        else:
            r["status"] = "unresolved"
        r["multiplicity_in_count"] = (r["winding_fine"] or 0) * (2 if z.imag > 0 else 1)
    return roots


def _origin_zeros(run, coarse, fine):
    """The zeros of D in the origin disk, when there are more than expected (DESIGN 3.3).

    The zero nearest to the origin is the translation zero (exactly 0 in the model);
    its computed value measures the error of the others.  Returns (translation zero,
    other zeros) with error and status, or (None, None) when the disk cannot be resolved.
    """
    r, margin = run.r0, run.opts["margin_factor"]
    try:
        seeds = isolate(fine, (-r, r, -r, r), (), min_size=1e-4 * r, initial=16,
                        limit=run.opts["phase_limit"])
    except (ContourError, SplittingError):
        return None, None
    zeros = []
    for s in seeds:
        z = complex(s)
        if abs(z) > r or z.imag < -1e-9 * r:
            continue                                  # corner of the square, or the conjugate of a kept zero
        if abs(z.imag) <= 1e-9 * r:
            z = complex(z.real, 0.0)
        if any(abs(z - q["value"]) <= 1e-4 * r for q in zeros):
            return None, None                         # a cluster that the subdivision did not separate
        try:
            z_coarse = locate(coarse, z, 0.1 * max(abs(z), 1e-3 * r))
        except ArithmeticError:
            z_coarse = None
        zeros.append(dict(value=z, coarse=z_coarse, seed=z, source="origin disk", near_origin=True,
                          winding_fine=1, winding_coarse=1, radius=None,
                          multiplicity_in_count=2 if z.imag > 0 else 1))
    zeros.sort(key=lambda q: abs(q["value"]))
    translation = zeros[0] if run.expected and zeros else None
    others = zeros[run.expected:]
    # The symmetry zeros are exactly 0 in the model: the largest of their computed values is the error.
    shift = max((abs(q["value"]) for q in zeros[:run.expected]), default=0.0)
    for q in others:
        z = q["value"]
        q["error"] = (abs(z - q["coarse"]) if q["coarse"] is not None else np.inf) + shift + run.wave_error
        if z.real > margin * q["error"]:
            q["status"] = "unstable"
        elif z.real < -margin * q["error"]:
            q["status"] = "stable"
        else:
            q["status"] = "unresolved"
    if run.predicted is not None:
        # Multi-pulse: compare with the reduction -sigma'(D).  It decides where the Evans
        # function cannot resolve the zero; a contradiction makes the zero unresolved.
        for q in others:
            j = int(np.argmin(np.abs(np.asarray(run.predicted) - q["value"].real)))
            value, error = float(run.predicted[j]), float(run.predicted_error[j])
            q["predicted"] = value
            if abs(value) <= margin * error:
                continue
            if q["status"] == "unresolved":
                q.update(status="unstable" if value > 0 else "stable", source="weak-interaction reduction")
            elif (q["status"] == "unstable") != (value > 0):
                q.update(status="unresolved", conflict="the Evans zero and the reduction disagree in sign")
    return translation, others


def _parity(run, coarse, fine):
    """Parity index: the signs of the real function D at r0 and at x_max (DESIGN 3.2).

    Different signs mean an odd number of real zeros in (r0, x_max).  One of them
    is then located by bisection and qualified like every other root.
    """
    def sign(engine, x):
        log_abs, phase, _ = engine(x)
        if not np.isfinite(log_abs) or abs(phase.imag) > 1e-6:
            return 0
        return 1 if phase.real > 0 else -1

    signs = {name: [sign(e, run.r0), sign(e, run.x_max)] for name, e in (("coarse", coarse), ("fine", fine))}
    out = dict(signs=signs, odd=None, root=None)
    if 0 in signs["fine"] + signs["coarse"]:
        return out                                    # D is not real here: no statement
    out["odd"] = signs["fine"][0] != signs["fine"][1]
    if not out["odd"] or signs["coarse"][0] == signs["coarse"][1]:
        return out
    a, b, sa = run.r0, run.x_max, signs["fine"][0]
    z = None
    for sweep in range(2):
        for _ in range(60):
            if b - a < (1e-3 if sweep == 0 else 1e-11) * max(1.0, b):
                break
            m = 0.5 * (a + b)
            if sign(fine, m) == sa:
                a = m
            else:
                b = m
        try:
            z = locate(fine, 0.5 * (a + b), 0.25 * (b - a))
        except ArithmeticError:
            z = None
        if z is not None and a <= z.real <= b and abs(z.imag) < 1e-8 * max(1.0, abs(z)):
            break
        z = complex(0.5 * (a + b))                    # the secant left the bracket: bisect further
    z = complex(z.real, 0.0)
    try:
        z_coarse = locate(coarse, z, 0.02 * max(abs(z), run.delta))
    except ArithmeticError:
        z_coarse = None
    out["root"] = _qualify(run, coarse, fine, [dict(value=z, coarse=z_coarse, seed=z, source="parity")])[0]
    return out


def _subdivide(run, coarse, fine, roots):
    """Independent root search: bisection of the box with winding numbers of the Evans function."""
    known = [0j] * run.expected + [z for r in roots for z in {r["value"], r["value"].conjugate()}]
    try:
        seeds = isolate(coarse, (-run.delta, run.x_max, -run.y_max, run.y_max), known,
                        min_size=run.r0 / 2, limit=run.opts["phase_limit"])
    except (ContourError, SplittingError) as error:
        run.subdivision = dict(seeds=0, error=str(error))
        return roots
    run.subdivision = dict(seeds=len(seeds))
    new = _polish(run, coarse, fine, seeds, roots)
    return _qualify(run, coarse, fine, roots + new) if new else roots


def _search(run, coarse, fine, target, known, factor=1.0):
    """Roots in the box.  Seeds come from finite-difference probes; the Evans function polishes them.

    The search stops when the located zeros account for `target` zeros of the box count.
    Zeros that the probes did not reach are found by subdivision of the box.
    """
    roots = list(known)
    located = lambda: sum(r["multiplicity_in_count"] for r in roots)
    if run.opts["seeds"] == "fd":
        _, A = run.fd_matrix(factor)
        for shifts in run.shift_groups():
            ev = run.probe(A, shifts)
            seeds = ev[(ev.real > -run.delta) & (np.abs(ev) > run.r0) & (ev.imag >= -1e-9)]
            new = _polish(run, coarse, fine, seeds, roots)
            if new:
                roots = _qualify(run, coarse, fine, roots + new)
            if target is not None and located() >= target:
                break
    if target is not None and located() < target:
        roots = _subdivide(run, coarse, fine, roots)
    return roots


def _pass_one(run):
    opts = run.opts
    h = run.h1
    for level in range(1 + opts["max_refinements"]):
        coarse, fine = run.engine(h), run.engine(h / 2)
        parity = _parity(run, coarse, fine)
        known = [] if parity["root"] is None else [parity["root"]]
        del parity["root"]
        if opts["parity_shortcut"] and known and known[0]["status"] == "unstable":
            return dict(h_coarse=h, h_fine=h / 2, cells_fine=fine.cells, refinement_level=level,
                        parity=parity, shortcut="parity", W=None, m0=None, counts_consistent=None,
                        translation_root=None, roots=known, complete_count=False, open_rhp_count=None,
                        outcome="UNSTABLE", reason="parity index: the number of real positive roots is odd; "
                        "one of them is located and confirmed")
        wc, wf = run.box_count(coarse), run.box_count(fine)
        oc, of = run.disk_count(coarse, 0j, run.r0), run.disk_count(fine, 0j, run.r0)
        consistent = (wc["resolved"] and wf["resolved"] and oc["resolved"] and of["resolved"]
                      and wc["count"] == wf["count"] and oc["count"] == of["count"])
        W, m0 = wf["count"], of["count"]
        origin, near, disk_resolved = None, [], m0 == run.expected
        if consistent and m0 > run.expected:
            # More zeros than expected next to the origin: locate and classify each of them.
            translation, others = _origin_zeros(run, coarse, fine)
            if others is not None and sum(q["multiplicity_in_count"] for q in others) == m0 - run.expected:
                near = others
                origin = translation["value"] if translation else None
                disk_resolved = all(q["status"] != "unresolved" for q in near)
        elif run.expected:
            try:
                origin = locate(fine, run.r0 / 4, run.r0 / 8)
            except (ContourError, ArithmeticError):
                pass
        roots = known
        if not consistent or W != m0:
            roots = _search(run, coarse, fine, W - m0 if consistent else None, known)
        located = sum(r["multiplicity_in_count"] for r in roots)
        complete = bool(consistent and W == m0 + located)
        roots = roots + near
        unstable = [r for r in roots if r["status"] == "unstable"]
        result = dict(h_coarse=h, h_fine=h / 2, cells_fine=fine.cells, refinement_level=level,
                      box=dict(coarse=wc, fine=wf), origin_disk=dict(coarse=oc, fine=of),
                      W=W, m0=m0, counts_consistent=bool(consistent), translation_root=origin,
                      roots=roots, complete_count=complete, parity=parity, subdivision=run.subdivision)
        if unstable:
            result.update(outcome="UNSTABLE", reason="a root with positive real part is confirmed",
                          open_rhp_count=len([r for r in roots if r["status"] == "unstable"
                                              for _ in range(2 if r["value"].imag > 0 else 1)])
                          if complete else None)
            return result
        if consistent and disk_resolved and complete and all(r["status"] == "stable" for r in roots):
            result.update(outcome="STABLE", reason="only the translation zero lies in Re >= 0"
                          if run.expected == 1 else "no zero lies in Re >= 0" if not run.expected else
                          f"only the {run.expected} symmetry zeros (translation, rotation) lie in Re >= 0")
            return result
        refine = (not consistent) or not disk_resolved or any(
            r["status"] == "unresolved" and r["value"].real > run.r0 for r in roots)
        if refine and level < opts["max_refinements"] and run.cells(h / 4) <= opts["max_cells"]:
            h /= 2
            continue
        reasons = []
        if not consistent:
            reasons.append("the counts differ between the two resolutions or are unresolved")
        if not disk_resolved:
            reasons.append(f"the origin disk contains {m0} zeros instead of {run.expected}, "
                           "and they are not all resolved")
        if consistent and not complete:
            reasons.append(f"the box count {W} exceeds the located zeros ({m0 + located})")
        if any(r["status"] == "unresolved" for r in roots):
            reasons.append("a root is too close to the imaginary axis or not simple")
        result.update(outcome="UNCERTAIN", reason="; ".join(reasons) or "undecided")
        return result


def _pass_two(run, first):
    """Audit of a result that pass 1 did not label UNSTABLE."""
    h = first["h_fine"]
    W, m0 = first["W"], first["m0"]
    fine, coarse = run.engine(h), run.engine(2 * h)
    checks = []

    def add(name, ok, **data):
        checks.append(dict(name=name, passed=bool(ok), **data))

    # 1. third resolution
    if run.cells(h / 2) <= run.opts["max_cells"]:
        finest = run.engine(h / 2)
        w3, o3 = run.box_count(finest), run.disk_count(finest, 0j, run.r0)
        origin = None
        near = [r for r in first["roots"] if r.get("near_origin")]
        if run.expected:
            # With other zeros in the disk, start the iteration at the translation zero of pass 1.
            seed = first["translation_root"] if near and first["translation_root"] is not None else run.r0 / 4
            try:
                origin = locate(finest, seed, run.r0 / 8 if not near else max(abs(seed), 1e-6 * run.r0))
            except (ContourError, ArithmeticError):
                pass
        add("third_resolution", w3["resolved"] and w3["count"] == W and o3["count"] == m0,
            h=h / 2, box=w3, origin_disk=o3, translation_root=origin)
        for r in near:
            # A zero next to the origin must keep its sign on the finest mesh.
            try:
                z3 = locate(finest, r["value"], 0.1 * abs(r["value"]))
            except (ContourError, ArithmeticError):
                z3 = None
            add("origin_disk_zero", z3 is not None and abs(z3 - r["value"]) < 0.5 * abs(r["value"])
                and (z3.real > 0) == (r["value"].real > 0), pass_1=r["value"], finest=z3)
        if run.expected:
            add("translation_root_small", origin is not None and abs(origin) < run.r0 / 10,
                value=origin, limit=run.r0 / 10)
    else:
        add("third_resolution", False, note="the cell limit max_cells does not allow a third resolution")

    # 2. change of the domain length, one side at a time
    for La, Lb in run.domain_changes():
        w = run.box_count(run.engine(h, La, Lb))
        add("domain_change", w["resolved"] and w["count"] == W, lengths=[La, Lb], box=w)

    # 3. independent discretisation: no finite-difference eigenvalue may be unaccounted for
    known = list(first["roots"])
    new_roots = []
    real_shifts, imag_shifts = run.shift_groups()
    for factor in (1.0, 1.5):
        nz, A = run.fd_matrix(factor)
        ev = run.probe(A, real_shifts + imag_shifts)
        ev = ev[(ev.real > -run.delta / 2) & (np.abs(ev) > run.r0) & (ev.imag >= -1e-9)]
        zone = []
        for s in ev:                                  # the probes overlap: remove duplicates
            if all(abs(s - q) > 1e-7 * max(1.0, abs(s)) for q in zone):
                zone.append(complex(s))
        unmatched = []
        for s in zone:
            if any(abs(s - r["value"]) < 0.25 * max(abs(r["value"]), run.delta) for r in known + new_roots):
                continue
            found = _polish(run, coarse, fine, [s], known + new_roots)
            if found:
                new_roots += _qualify(run, coarse, fine, found)
                continue
            try:
                z = locate(fine, s, 0.02 * max(abs(s), run.delta))
                if abs(z) <= run.r0 or any(abs(z - r["value"]) < 1e-6 * max(1.0, abs(z))
                                           or abs(z.conjugate() - r["value"]) < 1e-6 * max(1.0, abs(z))
                                           for r in known + new_roots):
                    continue                          # it is the translation zero or a known root
            except (ContourError, SplittingError, ArithmeticError):
                pass
            unmatched.append(complex(s))
        add("independent_discretisation", not unmatched and not new_roots, points=nz,
            probe_shifts=len(real_shifts) + len(imag_shifts), eigenvalues_in_zone=len(zone),
            unmatched=unmatched)

    # 4. rank certificate of the boundary rows
    g = first["box"]["fine"]
    add("boundary_rank_certificate", g["gamma_count"] == 0 and g["gamma_max_step"] < 1.0,
        gamma_count=g["gamma_count"], gamma_max_step=g["gamma_max_step"])

    # 5. wave accuracy relative to the origin disk
    add("wave_accuracy", run.wave_error < run.r0 / 10, spectral_error_scale=run.wave_error,
        limit=run.r0 / 10)

    # 6. parity index: the sign change of D on (r0, x_max) must match the located real roots
    parity = first.get("parity") or {}
    if parity.get("odd") is not None:
        real_roots = sum((r["winding_fine"] or 0) for r in first["roots"] + new_roots
                         if r["value"].imag == 0 and r["value"].real > run.r0)
        add("parity_index", parity["odd"] == (real_roots % 2 == 1), odd=parity["odd"],
            located_real_roots=real_roots)

    result = dict(checks=checks, new_roots=new_roots)
    unstable = [r for r in new_roots if r["status"] == "unstable"]
    failed = sorted({c["name"] for c in checks if not c["passed"]})
    if unstable:
        result.update(outcome="UNSTABLE", reason="the audit found and confirmed an unstable root")
    elif first["outcome"] == "STABLE" and not failed:
        result.update(outcome="STABLE", reason="all audit checks passed")
    elif first["outcome"] == "STABLE":
        result.update(outcome="UNCERTAIN", reason="audit checks failed: " + ", ".join(failed))
    else:
        result.update(outcome="UNCERTAIN", reason=first["reason"]
                      + ("; audit checks failed: " + ", ".join(failed) if failed else ""))
    return result


def _continued(path, given, problem, system, opts):
    """Wave at the start value of the continuation parameter, followed to the value of the file."""
    cont = problem.wave["continuation"]
    name = cont["parameter"]
    target = {str(k): v for k, v in problem.parameters.items()}[name]
    first = load_problem(path, {**(given or {}), name: cont["start"]})
    system.set_parameters(first.parameters)
    try:
        wave = solve_wave(first, system, tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                          tail_tol=opts["tail_tol"], L_max=opts["L_max"])
        branch = trace(system, first.wave["left"], first.wave["right"], wave, name,
                       direction=cont["direction"], step=cont["step"], until=target,
                       max_steps=cont["max_steps"], tail_tol=opts["tail_tol"], L_max=opts["L_max"],
                       fixed_L=first.wave["L"] is not None)
    finally:
        system.set_parameters(problem.parameters)
    if branch.final is None:
        raise WaveError(f"continuation in {name} from {cont['start']:g}: {branch.status}")
    return branch.final, branch.summary()


def _analyse_band(kit, problem, opts, record, finish, clock, frequency=None, rebuilds=0):
    """Sub-solvers D and N (DESIGN 2.9, 2.10): the same two passes with the band-determinant engine.

    frequency: the quadrature of the distributed delays is built for this frequency (delay.py).
    The analysis starts again, at most twice, if the search region needs a larger one.
    """
    tick = time.perf_counter()
    try:
        system = kit.System(problem) if frequency is None else kit.System(problem, frequency=frequency)
    except StructureError as error:
        return finish("UNCERTAIN", f"structure: {error}")
    tick = clock("symbolic", tick)
    spec = problem.wave
    record["classification"]["details"].update(kit.details(system))
    for side in ("left", "right"):
        res = np.abs(system.equilibrium_residual(spec[side])).max()
        if res > 1e-9 * max(1.0, np.abs(spec[side]).max()):
            return finish("UNCERTAIN", f"the {side} end state is not an equilibrium (residual {res:.3g})")
    c_guess = spec["speed"] if spec["speed"] is not None else spec["speed_guess"]
    s_ess, sides, scale0 = kit.end_bounds(system, spec, c_guess)
    record["essential"] = dict(s_ess=s_ess, sides=sides)
    record["wave"] = dict(kind=spec["kind"], left=spec["left"], right=spec["right"])
    tick = clock("essential_spectrum", tick)
    if any(v["growing_at_large_kappa"] for v in sides.values()):
        return finish("UNCERTAIN", "the dispersion curves still rise at the largest sampled wavenumber")
    if s_ess > 1e-9 * scale0:
        record["wave"]["computed"] = False
        return finish("UNSTABLE", "the essential spectrum enters the right half-plane "
                      f"(sup Re = {s_ess:.6g}); this holds for every wave with these end states", 1)
    if s_ess > -1e-9 * scale0:
        return finish("UNCERTAIN", "the essential spectrum touches the imaginary axis")
    try:
        wave = kit.solve_wave(problem, system, tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                              tail_tol=opts["tail_tol"], L_max=opts["L_max"])
    except (WaveError, StructureError, np.linalg.LinAlgError) as error:
        return finish("UNCERTAIN", f"wave: {error}")
    record["wave"].update(computed=True, speed=wave.c, L=wave.L, **wave.info)
    tick = clock("wave", tick)
    core = kit.core_spectrum(system, wave)
    if core is not None and core > s_ess:
        # Standing wave without diffusion: the values of the local coefficient along the wave are essential spectrum.
        s_ess = core
        record["essential"].update(s_ess=s_ess, core_of_the_wave=core)
        if s_ess > 1e-9 * scale0:
            return finish("UNSTABLE", "the essential spectrum enters the right half-plane "
                          f"(sup Re = {s_ess:.6g}, from the local coefficient along the standing wave)", 1)
        if s_ess > -1e-9 * scale0:
            return finish("UNCERTAIN", "the essential spectrum touches the imaginary axis")

    bounds = coefficient_bounds(system, wave)
    extra_norm, extra_bound = kit.coupling(system, wave, None)
    lam0 = bounds["lam0"] + extra_norm
    delta = min(-s_ess / 4, 0.1 * lam0, kit.delta_cap(system))
    if bounds["applies"]:
        # The delayed or nonlocal term is a bounded operator: one more constant in the energy estimate.
        coupling = float(extra_bound(delta))
        bounds.update(c0=bounds["c0"] + coupling, c2=bounds["c2"] + coupling, bounded_term=coupling)
    region, origin = (search_region(bounds, delta) if bounds["applies"] else None), "energy estimate"
    if region is None:
        own = kit.region(system, wave, delta)          # estimate of the sub-solver (no diffusion)
        if own is not None:
            region, origin = own[0], own[1]
            bounds = dict(bounds, **own[2])
    if opts["region"] is not None:
        (x_max, y_max), source = map(float, opts["region"]), "given in the problem file"
    elif region is not None:
        (x_max, y_max), source = region, origin
    else:
        record["region"] = dict(bounds=bounds)
        return finish("UNCERTAIN", "no bound on the search region: "
                      + (bounds["reason"] or "the energy estimate does not close for this system"))
    record["region"] = dict(x_min=-delta, x_max=x_max, y_max=y_max, origin_radius=delta / 2,
                            source=source, bounds=bounds)
    needed = kit.resolution(system, y_max, wave.c)
    if needed is not None:
        if rebuilds >= 2:
            return finish("UNCERTAIN", "the quadrature of the distributed delay does not reach the "
                          f"frequency {needed:.4g} that the search region needs")
        record["quadrature_rebuilds"] = rebuilds + 1
        return _analyse_band(kit, problem, opts, record, finish, clock, needed, rebuilds + 1)
    try:
        nu_max = kit.rate(system, spec, wave)
        run = _RunBand(kit, system, wave, opts, delta, x_max, y_max, lam0,
                       max(opts["h_factor"] / nu_max, kit.min_step(system, wave)))
        finest = run.cells(run.h1 / 4)
        if finest > opts["max_cells"]:
            run.h1 *= finest / opts["max_cells"]
        first = _pass_one(run)
        tick = clock("pass_1", tick)
        record.update(pass_1=first, roots=first["roots"], evaluations=run.evaluations)
        if first["outcome"] == "UNSTABLE" or not opts["two_pass"]:
            return finish(first["outcome"], first["reason"], 1)
        second = _pass_two(run, first)
        clock("pass_2", tick)
        record.update(pass_2=second, roots=first["roots"] + second["new_roots"], evaluations=run.evaluations)
        return finish(second["outcome"], second["reason"], 2)
    except ArithmeticError as error:              # contour through a zero, or lambda in the essential spectrum
        return finish("UNCERTAIN", f"spectral computation: {error}")


def _weighted_label(problem, system, spec, opts):
    """Label in an exponentially weighted space (DESIGN 2.13), for a wave that is unstable in L2
    through its essential spectrum.  Returns the dictionary of record['weighted']."""
    out = dict(label="UNCERTAIN", reason=None, space="L2 with the weight exp(phi), phi' -> the rates at the two ends")
    if system.quadratic or system.K != 2 or spec.get("copies", 1) > 1 or np.any(np.asarray(system.orders) > 2):
        out["reason"] = "the weighted analysis covers equations of second order in space and first order in time"
        return out
    try:
        wave = solve_wave(problem, system, tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                          tail_tol=opts["tail_tol"], L_max=opts["L_max"], max_nodes=opts["max_nodes"])
    except (WaveError, StructureError, np.linalg.LinAlgError) as error:
        out["reason"] = f"wave: {error}"
        return out
    out["wave"] = dict(speed=wave.c, L=wave.L, **wave.info)
    ends, scale0 = {}, 1e-300
    for side in ("left", "right"):
        a_k = system.end_state(spec[side], wave.c)[2]
        ends[side] = find_rate(a_k, system.first_order_components, system.mass)
        scale0 = max(scale0, float(np.linalg.norm(a_k[0], 2)))
    s_ess = max(end["supremum"] for end in ends.values())
    rates = [ends["left"]["rate"], ends["right"]["rate"]]
    out.update(ends=ends, rates=rates, s_ess=s_ess)
    if s_ess > 1e-9 * scale0:
        out.update(label="UNSTABLE", reason="no exponential weight moves the essential spectrum into the left "
                   f"half-plane (smallest supremum {s_ess:.6g})")
        return out
    if s_ess > -1e-9 * scale0:
        out["reason"] = "in the best weight the essential spectrum touches the imaginary axis"
        return out
    mode = translation_mode(system, wave, *rates)
    if mode is None:
        out["reason"] = "the decay rate of the wave equals the rate of the weight: the translation mode is marginal"
        return out
    out["translation_mode_in_the_space"] = bool(mode)
    return _weighted_passes(system, wave, spec["left"], spec["right"], rates, s_ess, 1 if mode else 0, opts, out)


def _weighted_passes(system, wave, left, right, rates, s_ess, expected, opts, out):
    """The two passes for `system` in the weight with the end rates `rates`; fills and returns `out`.

    expected: number of zeros in the origin disk (the translation mode, if it lies in the space).
    """
    wsystem, wwave = WeightedSystem(system), weighted_wave(wave, *rates)
    bounds = coefficient_bounds(wsystem, wwave)
    lam0 = bounds["lam0"]
    delta = min(-s_ess / 4, 0.1 * lam0) if lam0 > 0 else -s_ess / 4
    if bounds.get("beta", 0.0) < 0:
        delta = min(delta, -bounds["beta"] / 2)
    region = search_region(bounds, delta) if bounds["applies"] else None
    if region is None:
        out["reason"] = ("no bound on the search region in the weighted space: "
                         + (bounds["reason"] or "the energy estimate does not close"))
        return out
    x_max, y_max = region
    out["region"] = dict(x_min=-delta, x_max=x_max, y_max=y_max, origin_radius=delta / 2, bounds=bounds)
    try:
        asym = Asymptotics(wsystem, list(left) + rates[:1], list(right) + rates[1:], wave.c, x_max)
        nu_max = max(np.abs(np.linalg.eigvals(asym.M0m)).max(), np.abs(np.linalg.eigvals(asym.M0p)).max())
        run = _Run(wsystem, wwave, asym, opts, delta, x_max, y_max, lam0, opts["h_factor"] / nu_max)
        run.expected = expected
        finest = run.cells(run.h1 / 4)
        if finest > opts["max_cells"]:
            run.h1 *= finest / opts["max_cells"]
        first = _pass_one(run)
        out.update(pass_1=first, roots=first["roots"], evaluations=run.evaluations)
        if first["outcome"] == "UNSTABLE" or not opts["two_pass"]:
            out.update(label=first["outcome"], reason=first["reason"], decided_in_pass=1)
            return out
        second = _pass_two(run, first)
        out.update(pass_2=second, roots=first["roots"] + second["new_roots"], evaluations=run.evaluations,
                   label=second["outcome"], reason=second["reason"], decided_in_pass=2)
    except (ContourError, SplittingError) as error:
        out["reason"] = f"spectral computation: {error}"
    return out


def _default_region(system, wave):
    """Search rectangle (x_max, y_max) for an equation without an estimate (DESIGN 3.2).

    Four and eight times the size of the lower-order coefficients along the wave:
    sup |a0| + sup |a1| + sup |a1'| / 2.  It is a rule, not a bound: a label that rests on it
    carries a qualifier.
    """
    z = np.linspace(-wave.L, wave.hi, 2001)
    a = system.coefficients(wave.Y(z), wave.c)
    norm = lambda A: float(np.linalg.norm(A, 2, axis=(1, 2)).max())
    size = norm(a[0])
    if a.shape[0] > 1:
        size += norm(a[1]) + 0.5 * norm(np.gradient(a[1], z, axis=0))
    size = max(size, 1e-6)
    return 4.0 * size, 8.0 * size


DEFAULT_RULE = "default rule (no estimate)"


def _real_root(engine, a, b):
    """A real zero of the determinant in (a, b), by bisection in the logarithm; None without a sign change.

    The search rectangle of an equation of higher order can be many orders of magnitude larger than
    the eigenvalue.  A subdivision of the rectangle does not reach that scale; a bisection does.
    """
    base = []

    def sign(x):
        # On the real axis the determinant is real up to a constant factor (rounding of the
        # normalisation, when the reference point lies far away): signs are taken relative to the start.
        log_abs, phase, _ = engine(x)
        if not np.isfinite(log_abs):
            return 0
        base.append(phase)
        ratio = phase / base[0]
        if abs(ratio.imag) > 1e-3:
            return 0
        return 1 if ratio.real > 0 else -1

    # March upwards from a: the first change of sign brackets the smallest real zero.  The sign at
    # the far end b alone is not used: the determinant can be unreliable there.
    sa, lower, upper = sign(a), a, None
    if sa == 0:
        return None
    x = a
    while x < b:
        x = min(4.0 * x, b)
        s = sign(x)
        if s == -sa:
            upper = x
            break
        if s == sa:
            lower = x
    if upper is None:
        return None
    a, b = lower, upper
    for _ in range(200):
        if b / a < 1 + 1e-9:
            break
        m = float(np.sqrt(a * b))
        s = sign(m)
        if s == 0:
            return None
        a, b = (m, b) if s == sa else (a, m)
    return complex(np.sqrt(a * b), 0.0)


def _neutral_label(system, wave, left, right, opts, axis=False):
    """Label when the essential spectrum touches or lies on the imaginary axis (DESIGN 2.15).

    axis=True (DESIGN 2.14, shocks of conservation laws): the rectangle starts at Re(lambda) = 0.  The
    determinant of the integrated operator is continuous on the closed half-plane; its end-state
    subspaces come from TouchingAsymptotics.

    Only Re(lambda) > r0 is examined: the zeros of the determinant are counted in the rectangle
    [r0, x_max] x [-y_max, y_max] on two meshes and on a shorter domain.  r0 is the resolution
    threshold; a growth rate below r0 is not excluded.  Returns the dictionary of record['neutral'].
    """
    out = dict(label="UNCERTAIN", reason=None)
    bounds = coefficient_bounds(system, wave)
    region = search_region(bounds, 0.0) if bounds["applies"] else None
    if opts["region"] is not None:
        (x_max, y_max), source = map(float, opts["region"]), "given in the problem file"
    elif region is not None:
        (x_max, y_max), source = region, "energy estimate"
    elif opts["default_region"]:
        (x_max, y_max), source = _default_region(system, wave), DEFAULT_RULE
    else:
        out["reason"] = ("no bound on the search region: "
                         + (bounds["reason"] or "the energy estimate does not close for this system")
                         + "; give options['region'] = (x_max, y_max)")
        return out
    scale = max(float(np.linalg.norm(a, 2)) for u in (left, right) for a in system.end_state(u, wave.c)[2])
    r0 = float(opts["neutral_threshold"]) if opts["neutral_threshold"] is not None else 0.01 * scale
    y_scale = r0                                     # spacing of the contour next to the real axis
    if axis:
        r0 = 0.0
    out.update(threshold=r0, region=dict(x_min=r0, x_max=x_max, y_max=y_max, source=source, bounds=bounds),
               qualifier="algebraic decay" if axis else f"neutral: no spectrum with Re(lambda) > {r0:.3g}")
    if source == DEFAULT_RULE:
        out["qualifier"] += f"; default search rectangle {x_max:.3g} x {y_max:.3g}"
    if x_max <= r0:
        out.update(label="STABLE", reason=f"the estimate of the search region excludes Re(lambda) > {r0:.3g}")
        return out
    try:
        asym = (TouchingAsymptotics if axis else Asymptotics)(system, left, right, wave.c, x_max)
        nu_max = max(np.abs(np.linalg.eigvals(asym.M0m)).max(), np.abs(np.linalg.eigvals(asym.M0p)).max())
        h = opts["h_factor"] / nu_max
        cap = float(opts["mesh_cap"])
        variation = wave_variation(system, wave) if cap > 1.0 else None
        build = lambda step, L: Evans.build(system, wave, asym, L, L, step, cap, variation)
        contour = box(r0, x_max, y_max, y_scale)
        out["splitting_dimension"] = asym.k
        wave_error = bounds["lam0"] * wave.info["profile_error"]
        out["evaluations"] = 0
        for level in range(1 + opts["max_refinements"]):
            # A zero at the origin of higher multiplicity splits on a mesh; the pieces move towards
            # the origin when the mesh is refined.  So an unconfirmed result is tried again on finer meshes.
            coarse, fine, short = build(h, wave.L), build(h / 2, wave.L), build(h / 2, 0.75 * wave.L)
            counts = [winding(e, contour, opts["initial_panels"], opts["phase_limit"]) for e in (coarse, fine, short)]
            out.update(counts=[c["count"] for c in counts], resolved=[bool(c["resolved"]) for c in counts],
                       h_fine=h / 2, refinement_level=level, roots=[],
                       evaluations=out["evaluations"] + sum(c["evaluations"] for c in counts))
            agree = all(c["resolved"] for c in counts) and len({c["count"] for c in counts}) == 1
            if agree and counts[1]["count"] == 0:
                out.update(label="STABLE", reason=f"no zero of the determinant in {r0:.3g} <= Re(lambda) <= "
                           f"{x_max:.4g}, |Im(lambda)| <= {y_max:.4g} (two meshes and a shorter domain)")
                return out
            out["reason"] = "the counts differ between the meshes or the domains, or are not resolved"
            if agree:
                roots, seeds, radius = [], [], {}
                if counts[1]["count"] % 2:
                    # An odd count: a real zero.  It can lie far below the size of the rectangle.
                    real = _real_root(fine, max(r0, 1e-9 * y_scale), x_max)
                    if real is not None:
                        seeds.append(real)
                        radius[real] = 0.02 * abs(real)
                if len(seeds) < counts[1]["count"]:
                    try:
                        seeds += list(isolate(fine, (r0, x_max, -y_max, y_max),
                                              min_size=1e-3 * max(x_max - r0, 1e-300)))
                    except (ContourError, SplittingError):
                        if not seeds:
                            raise
                for seed in seeds:
                    z = locate(fine, seed, radius.get(seed, 0.02 * max(abs(seed), y_scale)))
                    if z.imag < -1e-9 * max(1.0, abs(z)) or any(abs(z - q["value"]) < 1e-8 * max(1.0, abs(z))
                                                              for q in roots):
                        continue                       # the conjugate, or a zero found twice
                    zc = locate(coarse, z, 0.02 * (abs(z) if seed in radius else max(abs(z), y_scale)))
                    error = abs(z - zc) + wave_error
                    status = "unstable" if z.real > opts["margin_factor"] * error else "unresolved"
                    roots.append(dict(value=z, coarse=zc, error=error, status=status))
                out["roots"] = roots
                if roots and all(r["status"] == "unstable" for r in roots):
                    out.update(label="UNSTABLE", reason="a root with positive real part is confirmed")
                    return out
                out["reason"] = "the count is positive, but a root is not confirmed on both meshes"
            h /= 2
    except (ContourError, SplittingError) as error:
        out["reason"] = f"spectral computation: {error}"
    return out


def _neutral_finish(neutral, record, finish, prefix, key="neutral"):
    """Common end of an analysis in the neutral mode."""
    record.update({key: neutral}, roots=neutral.get("roots", []), evaluations=neutral.get("evaluations"))
    if neutral["label"] == "STABLE":
        record["qualifier"] = neutral["qualifier"]
        return finish("STABLE", prefix + neutral["reason"], 2)
    return finish(neutral["label"], prefix + neutral["reason"], 1 if neutral["label"] == "UNSTABLE" else None)


def _conservative_label(problem, system, opts, record, finish, clock):
    """Equations in conservation form (DESIGN 2.14).

    The integrated operator is analysed.  Its essential spectrum touches the origin; transport at the
    end states (characteristic speeds relative to the wave) allows weights that move it away.
    """
    tick = time.perf_counter()
    spec = problem.wave
    if problem.boundary is not None or spec.get("copies", 1) > 1:
        return finish("UNCERTAIN", "conservation form with a free boundary or a multi-pulse is not supported")
    try:
        cons = ConservativeSystem(problem, system)
    except StructureError as error:
        return finish("UNCERTAIN", f"conservation form: {error}")
    record["conservation"] = dict(speed=cons.speed, flux_constant=[float(q) for q in cons.constant],
                                  jump_residual=cons.jump_residual, operator="integrated", depth=cons.depth)
    record["wave"] = dict(kind=spec["kind"], left=spec["left"], right=spec["right"])
    supplied_wave = None
    if spec.get("exact_profile") is not None:
        try:
            supplied_wave = exact_wave(problem, cons, opts)
        except (WaveError, StructureError, ValueError, NotImplementedError, ArithmeticError) as error:
            return finish("UNCERTAIN", f"exact profile: {error}")
    ends = {side: cons.end_state(spec[side])[2] for side in ("left", "right")}
    scale0 = max(max(float(np.linalg.norm(a, 2)) for a in end[1:]) for end in ends.values())
    s_ess, sides = essential_bound(ends["left"], ends["right"])
    record["essential"] = dict(s_ess=s_ess, sides=sides)
    tick = clock("essential_spectrum", tick)
    if any(v["growing_at_large_kappa"] for v in sides.values()):
        return finish("UNCERTAIN", "the dispersion curves still rise at the largest sampled wavenumber")
    if s_ess > axis_tolerance(sides, scale0):
        record["wave"]["computed"] = False
        return finish("UNSTABLE", "the essential spectrum enters the right half-plane "
                      f"(sup Re = {s_ess:.6g}): the constant end states are unstable", 1)
    try:
        wave = supplied_wave
        if spec.get("unfolding") is not None:
            wave = unfold_wave(problem,cons,opts)
        elif wave is None:
            wave = solve_wave(cons.reduced_problem, cons.reduced, tol=opts["wave_tol"],
                              check_tol=opts["wave_check_tol"], tail_tol=opts["tail_tol"],
                              L_max=opts["L_max"], max_nodes=opts["max_nodes"])
    except (WaveError, StructureError, np.linalg.LinAlgError, ArithmeticError, ValueError) as error:
        hint = ("; no speed was given and the standing pulse (speed 0) was tried: a pulse of a conservation "
                "law can exist for a family of speeds, give wave['speed']") if cons.default_speed else ""
        return finish("UNCERTAIN", f"wave: {error}{hint}")
    record["wave"].update(computed=True, speed=cons.speed, L=wave.L,
                          **{k: v for k, v in wave.info.items() if k != "speed_is_unknown"})
    tick = clock("wave", tick)
    method = opts["conservation_method"]
    front = np.abs(np.subtract(spec["left"], spec["right"])).max() > 0
    # The determinant on the closed half-plane needs a front (no zero at the origin), one integration,
    # and transport at both end states: all characteristic speeds differ from the speed of the wave.
    transport = [cons.transport(spec[side]) if cons.depth == 1 else None for side in ("left", "right")]
    axis_applies = front and cons.depth == 1 and all(t is not None for t in transport)
    record["conservation"].update(transport=dict(left=transport[0], right=transport[1]))
    if opts["conservation_origin_radius"] is not None:
        if ((supplied_wave is None and spec.get("unfolding") is None) or not axis_applies or method=="weights"):
            return finish("UNCERTAIN", "continued origin requires a validated conservative front and axis analysis")
        try:
            closed = continued_origin(cons,wave,spec["left"],spec["right"],opts)
        except (WaveError, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            return finish("UNCERTAIN", f"continued origin: {error}")
        clock("continued_origin", tick)
        record["conservation"]["method"] = "continued determinant around the origin"
        return _neutral_finish(closed, record, finish, "conservative integrated operator: ",
                               key="closed_half_plane")
    if cons.K == 2 and method != "axis":
        found = {side: find_rate(ends[side], [], cons.mass) for side in ("left", "right")}
        s_weighted = max(end["supremum"] for end in found.values())
        if s_weighted < -1e-9 * scale0:
            rates = [found["left"]["rate"], found["right"]["rate"]]
            front = np.abs(np.subtract(spec["left"], spec["right"])).max() > 0
            # For a front the integral of the translation mode, U - u_-, does not decay: no zero at the origin.
            mode = False if front else translation_mode(cons.reduced, wave, *rates)
            out = dict(label="UNCERTAIN", reason=None, ends=found, rates=rates, s_ess=s_weighted)
            if mode is None:
                out["reason"] = "the decay rate of the pulse equals the rate of the weight"
            else:
                out = _weighted_passes(cons, wave, spec["left"], spec["right"], rates, s_weighted,
                                       1 if mode else 0, opts, out)
            record.update(weighted=out, roots=out.get("roots", []), evaluations=out.get("evaluations"))
            clock("weighted_space", tick)
            where = f"in the weighted space with the rates {rates[0]:.4g} and {rates[1]:.4g} at the two ends"
            if out["label"] == "STABLE":
                record["qualifier"] = "algebraic decay"
                return finish("STABLE", "conservation form: the essential spectrum touches the origin and nothing else "
                              f"lies in Re >= 0; {where} the integrated operator has no spectrum in Re >= 0", 2)
            if out["label"] == "UNSTABLE":
                return finish("UNSTABLE", f"conservation form, integrated operator {where}: {out['reason']}", 1)
            return finish("UNCERTAIN", f"conservation form, integrated operator {where}: {out['reason']}")
    if axis_applies and method != "weights":
        # Shocks of systems and of equations of higher order: no single weight exists, but the
        # determinant of the integrated operator is continuous on Re(lambda) >= 0 (DESIGN 2.14).
        closed = _neutral_label(cons, wave, spec["left"], spec["right"], opts, axis=True)
        clock("closed_half_plane", tick)
        record["conservation"]["method"] = "determinant on the closed half-plane"
        return _neutral_finish(closed, record, finish, "conservation form, integrated operator on the closed "
                               "half-plane Re(lambda) >= 0: ", key="closed_half_plane")
    # No transport at an end state, or a pulse: no weight separates the essential spectrum from the
    # origin, and the determinant vanishes or is not continuous there.  Only growth rates above a
    # threshold are examined.
    neutral = _neutral_label(cons, wave, spec["left"], spec["right"], opts)
    clock("neutral", tick)
    return _neutral_finish(neutral, record, finish, "conservation form, integrated operator: ")


def _weighted_summary(weighted):
    """One clause for the reason and the command line."""
    if weighted["label"] != "STABLE":
        return f"in an exponentially weighted space: {weighted['label']} ({weighted['reason']})"
    parts = []
    for side, where in (("left", "behind"), ("right", "ahead")) if weighted["wave"]["speed"] >= 0 else (
            ("left", "ahead"), ("right", "behind")):
        end = weighted["ends"][side]
        if end["rate"] != 0.0:
            lo, hi = end["interval"]
            parts.append(f"exp({end['rate']:.4g} z) {where} of the wave (rates from {lo:.4g} to {hi:.4g})")
    return "in the weighted space with " + " and ".join(parts) + ": STABLE"


def analyse(path, **overrides):
    """Analyse the wave described in a problem file.  Returns a JSON-ready record.

    Keyword arguments replace entries of DEFAULTS.  `parameters={name: value}`
    replaces parameter values of the file.  `start=Wave` replaces the guess of
    the file (for example a point of a branch computed by continuation.trace).
    """
    start = time.perf_counter()
    timings = {}
    record = dict(solver_version=__version__, file=source_name(path), label=None, decided_in_pass=None,
                  reason=None, space=SPACE)

    def finish(label, reason, decided=None):
        timings["total"] = time.perf_counter() - start
        record.update(label=label, reason=reason, decided_in_pass=decided, timings=timings)
        return _clean(record)

    def clock(name, since):
        timings[name] = time.perf_counter() - since
        return time.perf_counter()

    tick = time.perf_counter()
    given = overrides.pop("parameters", None)
    start_wave = overrides.pop("start", None)
    try:
        problem = load_problem(path, given, wave=overrides.pop("wave", None))
    except ProblemError as error:
        return finish("UNCERTAIN", f"input file: {error}")
    if problem.wave is None:
        return finish("UNCERTAIN", "the file names no wave: use twstab.survey.survey(path), or the command "
                      "line, to search for the waves of the equation")
    opts = {**DEFAULTS, **problem.options, **overrides}
    record.update(name=problem.name, parameters={str(k): v for k, v in problem.parameters.items()})
    cls = classify(problem)
    record["classification"] = cls.as_dict()
    if not cls.supported:
        return finish("UNCERTAIN", f"sub-solver {cls.sub_solver} is not available: " + "; ".join(cls.reasons))
    with_frequency = (problem.wave.get("frequency") is not None
                      or problem.wave.get("frequency_guess") is not None)
    if cls.sub_solver in ("D", "N"):
        if (problem.wave.get("exact_profile") is not None or problem.wave.get("unfolding") is not None
                or opts["conservation_origin_radius"] is not None):
            return finish("UNCERTAIN", "exact profile, unfolding and continued origin require a local conservative front")
        if with_frequency:
            return finish("UNCERTAIN", "a wave with a frequency is not supported together with a delay or a convolution")
        kit = delay_solver if cls.sub_solver == "D" else kernel_solver
        return _analyse_band(kit, problem, opts, record, finish, clock)
    try:
        system = LocalSystem(problem)
    except StructureError as error:
        return finish("UNCERTAIN", f"structure: {error}")
    tick = clock("symbolic", tick)
    record["classification"]["details"].update(orders=system.orders, first_order_dimension=system.N,
                                               constraints=int((system.mass == 0).sum()))
    if system.conservative:
        if with_frequency:
            return finish("UNCERTAIN", "a wave with a frequency is not supported for equations in conservation form")
        return _conservative_label(problem, system, opts, record, finish, clock)
    if (problem.wave.get("exact_profile") is not None or problem.wave.get("unfolding") is not None
            or opts["conservation_origin_radius"] is not None):
        return finish("UNCERTAIN", "exact profile, unfolding and continued origin require a local conservative front")

    spec = problem.wave
    free = problem.boundary is not None           # sub-solver F: only the left state is an end state
    sides = ("left",) if free else ("left", "right")
    rotating = None
    if spec.get("frequency") is not None or spec.get("frequency_guess") is not None:
        # A wave with a frequency (rotation symmetry, DESIGN 2.16): the analysis is done in the frame
        # that moves and rotates with the wave.  There the wave is an ordinary standing wave.
        if spec["copies"] > 1 or start_wave is not None or spec.get("continuation") is not None:
            return finish("UNCERTAIN", "a wave with a frequency cannot be combined with a multi-pulse or a continuation")
        try:
            rotating = RotatingFrame(problem, system)
        except StructureError as error:
            return finish("UNCERTAIN", f"rotation symmetry: {error}")
        problem, system = rotating.problem, rotating.system
        record["symmetry"] = dict(rotation=list(rotating.names), expected_zeros_at_the_origin=2)
        tick = clock("rotating_frame", tick)
    for side in sides:
        res = np.abs(system.equilibrium_residual(spec[side])).max()
        if res > 1e-9 * max(1.0, np.abs(spec[side]).max()):
            return finish("UNCERTAIN", f"the {side} end state is not an equilibrium (residual {res:.3g})")
    c_guess = spec["speed"] if spec["speed"] is not None else spec["speed_guess"]
    try:
        cond = max(system.top_matrix_condition(spec[s], c_guess) for s in sides)
        if (not cond < 1e12 and not free and rotating is None and spec["copies"] == 1
                and isinstance(system, LocalSystem)):
            # The second derivative of a component without diffusion enters another equation (taxis):
            # first-order form in flux variables (DESIGN 2.17).
            try:
                taxis = TaxisSystem(problem, system)
            except StructureError:
                taxis = None
            if taxis is not None and max(taxis.top_matrix_condition(spec[s], c_guess) for s in sides) < 1e12:
                names = [u.func.__name__ for u in problem.unknowns]
                record["taxis"] = dict(with_diffusion=[names[j] for j in taxis.diffusive],
                                       without_diffusion=[names[j] for j in taxis.first_order_components],
                                       form="flux variables")
                system, cond = taxis, 1.0
        if not cond < 1e12:
            record["classification"].update(sub_solver="G", supported=False)
            return finish("UNCERTAIN", "the highest-derivative matrix is singular at an end state "
                          "(degenerate diffusion, or zero speed with a first-order component)")
        a_minus = system.end_state(spec["left"], c_guess)[2]
        a_plus = a_minus if free else system.end_state(spec["right"], c_guess)[2]
    except np.linalg.LinAlgError:
        return finish("UNCERTAIN", "the highest-derivative matrix is singular at an end state")

    # --- step 3: essential spectrum (no wave needed) -----------------------------
    try:
        if system.quadratic:
            if free or spec.get("copies", 1) > 1:
                return finish("UNCERTAIN", "second-order time derivatives are not supported together with "
                              "a free boundary or a multi-pulse")
            ends = [(a_k, system.damping(system.state(spec[side]), c_guess)[0], system.inertia)
                    for side, a_k in (("left", a_minus), ("right", a_plus))]
            s_ess, sides = quadratic_essential_bound(ends[0], ends[1], c_guess)
        else:
            s_ess, sides = essential_bound(a_minus, a_plus, first_order=system.first_order_components,
                                           mass=system.mass)
    except np.linalg.LinAlgError as error:
        return finish("UNCERTAIN", f"essential spectrum: {error}")
    scale0 = max(np.linalg.norm(a_minus[0], 2), np.linalg.norm(a_plus[0], 2), 1e-300)
    record["essential"] = dict(s_ess=s_ess, sides=sides)
    record["wave"] = dict(kind=spec["kind"], left=spec["left"], right=spec["right"])
    tick = clock("essential_spectrum", tick)
    if any(v["growing_at_large_kappa"] for v in sides.values()):
        return finish("UNCERTAIN", "the dispersion curves still rise at the largest sampled wavenumber")
    tolerance = axis_tolerance(sides, scale0)        # includes the rounding error of the dispersion curves
    if s_ess > tolerance:
        record["wave"]["computed"] = False
        reason = ("the essential spectrum enters the right half-plane "
                  f"(sup Re = {s_ess:.6g}); this holds for every wave with these end states")
        if opts["weighted"] and not free and isinstance(system, LocalSystem):
            # The instability can be convective: the same wave can be stable in a weighted space.
            record["weighted"] = _weighted_label(problem, system, spec, opts)
            clock("weighted_space", tick)
            reason += "; " + _weighted_summary(record["weighted"])
        return finish("UNSTABLE", reason, 1)
    touching = s_ess > -tolerance                    # essential spectrum on the axis: neutral mode after the wave
    if touching and (free or spec.get("copies", 1) > 1):
        return finish("UNCERTAIN", "the essential spectrum touches the imaginary axis")

    # --- step 4: the wave ---------------------------------------------------------
    try:
        if free:
            if start_wave is not None or spec.get("continuation") is not None:
                raise WaveError("continuation is not available for free-boundary waves")
            wave = solve_semi_wave(problem, system, tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                                   tail_tol=opts["tail_tol"], L_max=opts["L_max"])
        else:
            if start_wave is None and spec.get("continuation") is not None:
                start_wave, record["wave"]["continuation"] = _continued(path, given, problem, system, opts)
                tick = clock("continuation", tick)
            # With u_tt and a given speed: the speed can be a symmetry parameter (no damping).
            at_speed = system.quadratic and spec["speed"] is not None and start_wave is None
            wave = (rotating.solve_wave(tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                                        tail_tol=opts["tail_tol"], L_max=opts["L_max"], max_nodes=opts["max_nodes"])
                    if rotating is not None else
                    solve_wave_at_speed(problem, system, tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                                        tail_tol=opts["tail_tol"], L_max=opts["L_max"], max_nodes=opts["max_nodes"])
                    if at_speed else
                    solve_wave(problem, system, tol=opts["wave_tol"], check_tol=opts["wave_check_tol"],
                               tail_tol=opts["tail_tol"], L_max=opts["L_max"], max_nodes=opts["max_nodes"],
                               start=start_wave))
            if spec["copies"] > 1:                    # the wave so far is the primary pulse
                record["wave"]["primary_pulse"] = dict(speed=wave.c, L=wave.L)
                wave = solve_multipulse(problem, system, wave, tol=opts["wave_tol"],
                                        check_tol=opts["wave_check_tol"], tail_tol=opts["tail_tol"])
                tick = clock("primary_pulse_and_distances", tick)
    except (WaveError, StructureError, np.linalg.LinAlgError) as error:
        hint = ""
        if rotating is None and not free and isinstance(system, LocalSystem) and rotation_pairs(system):
            hint = ("; the equations have a rotation symmetry: a wave with a frequency needs "
                    "wave['frequency'] or wave['frequency_guess']")
        return finish("UNCERTAIN", f"wave: {error}{hint}")
    record["wave"].update(computed=True, speed=wave.c, L=wave.L, **wave.info)
    if rotating is not None:
        record["symmetry"]["frequency"] = wave.info["frequency"]
    tick = clock("wave", tick)
    if touching:
        neutral = _neutral_label(system, wave, spec["left"], spec["right"], opts)
        clock("neutral", tick)
        return _neutral_finish(neutral, record, finish, "the essential spectrum touches the imaginary axis; ")

    # --- step 5: search region and engines ---------------------------------------
    bounds = coefficient_bounds(system, wave)
    if free:
        bounds = boundary_bounds(system, wave, bounds)
    lam0 = bounds["lam0"]
    delta = min(-s_ess / 4, 0.1 * lam0) if lam0 > 0 else -s_ess / 4
    if bounds.get("beta", 0.0) < 0:
        delta = min(delta, -bounds["beta"] / 2)       # the estimate needs -delta > beta (DESIGN 2.4)
    if bounds.get("indefinite") and not free:
        scaled = scaled_bounds(system, wave, delta)   # cross-diffusion: estimate in scaled variables
        if scaled is not None:
            bounds = dict(scaled, lam0=lam0)
    region = (boundary_region if free else search_region)(bounds, delta) if bounds["applies"] else None
    if opts["region"] is not None:
        x_max, y_max = map(float, opts["region"])
        source = "given in the problem file"
    elif region is not None:
        x_max, y_max = region
        source = "energy estimate"
    elif opts["default_region"] and not free:
        x_max, y_max = _default_region(system, wave)
        source = DEFAULT_RULE
    else:
        record["region"] = dict(bounds=bounds)
        return finish("UNCERTAIN", "no bound on the search region: "
                      + (bounds["reason"] or "the energy estimate does not close for this system")
                      + "; give options['region'] = (x_max, y_max)")
    record["region"] = dict(x_min=-delta, x_max=x_max, y_max=y_max, origin_radius=delta / 2,
                            source=source, bounds=bounds)
    try:
        if free:
            asym = BoundaryAsymptotics(system, spec["left"], wave, x_max)
            nu_max = np.abs(np.linalg.eigvals(asym.M0m)).max()
        else:
            asym = Asymptotics(system, spec["left"], spec["right"], wave.c, x_max)
            nu_max = max(np.abs(np.linalg.eigvals(asym.M0m)).max(), np.abs(np.linalg.eigvals(asym.M0p)).max())
        run = (_RunBoundary if free else _Run)(system, wave, asym, opts, delta, x_max, y_max, lam0,
                                               opts["h_factor"] / nu_max)
        if rotating is not None:
            run.expected = 2                          # the translation zero and the rotation zero
        finest = run.cells(run.h1 / 4)
        if finest > opts["max_cells"]:
            run.h1 *= finest / opts["max_cells"]      # three resolutions must fit into max_cells
        record["splitting_dimension"] = asym.k

        # --- step 6: pass 1 ---------------------------------------------------------
        first = _pass_one(run)
        tick = clock("pass_1", tick)
        record["pass_1"] = first
        record["roots"] = first["roots"]
        record["evaluations"] = run.evaluations
        def limited(outcome):
            # With the default rectangle a STABLE label holds for this rectangle only: a printed qualifier.
            if source == DEFAULT_RULE and outcome == "STABLE":
                record["qualifier"] = f"default search rectangle {x_max:.3g} x {y_max:.3g}"

        if first["outcome"] == "UNSTABLE" or not opts["two_pass"]:
            limited(first["outcome"])
            return finish(first["outcome"], first["reason"], 1)

        # --- pass 2: audit of everything that is not UNSTABLE -------------------------
        second = _pass_two(run, first)
        clock("pass_2", tick)
        record["pass_2"] = second
        record["roots"] = first["roots"] + second["new_roots"]
        record["evaluations"] = run.evaluations
        limited(second["outcome"])
        return finish(second["outcome"], second["reason"], 2)
    except (ContourError, SplittingError) as error:
        return finish("UNCERTAIN", f"spectral computation: {error}")
